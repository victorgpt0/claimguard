"""Load a claims set into the platform (#65), as if providers had lodged it over time.

    python -m sha_cdp.seed <live.parquet> [--seed 2026] [--replace]

Each claim gets the processing times SHA's rules allow:
  - lodged within 7 days of discharge (LN 49/2024, reg. 58(1));
  - approved after the Claims Management Office's review (reg. 59), here 1-14 days;
  - paid within 90 days of approval (Social Health Insurance (Amendment) Regulations 2024),
    here 14-60 days, inside that limit.

Only the platform's own columns are kept (KEEP). The ground-truth columns (is_fraud,
fraud_typology, label, the typology flags, original_icd11_code, the noise flags) are dropped
here, so they never reach the platform's database, its API, or the audit system."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import delete, func, insert, select

from . import db

KEEP = ["claim_id", "provider_id", "facility_level", "member_id", "beneficiary_type", "patient_age", "patient_sex",
        "icd11_code", "claim_amount", "service_date", "discharge_date", "clinical_notes"]
LODGE_DAYS = (0, 7)        # reg. 58(1)
APPROVE_DAYS = (1, 14)     # reg. 59 review
PAY_DAYS = (14, 60)        # within the 90-day limit


def schedule(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    """The claims with lodged_at, approved_at and paid_at, and nothing else beyond KEEP."""
    missing = [c for c in KEEP if c not in frame.columns]
    if missing:
        raise ValueError(f"the claims set has no {', '.join(missing)} column")
    rng = np.random.default_rng(seed)
    out = frame[KEEP].copy()
    out["service_date"] = pd.to_datetime(out["service_date"])
    out["discharge_date"] = pd.to_datetime(out["discharge_date"], errors="coerce")
    n = len(out)

    def days(bounds):
        # Whole days plus a time within office hours (08:00-17:00), so payments spread over the day.
        return (pd.to_timedelta(rng.integers(bounds[0], bounds[1] + 1, n), unit="D")
                + pd.to_timedelta(rng.integers(8 * 3600, 17 * 3600, n), unit="s"))

    out["lodged_at"] = out["discharge_date"].fillna(out["service_date"]) + days(LODGE_DAYS)
    out["approved_at"] = out["lodged_at"].dt.normalize() + days(APPROVE_DAYS)
    out["paid_at"] = out["approved_at"].dt.normalize() + days(PAY_DAYS)
    out["service_date"] = out["service_date"].dt.date
    out["discharge_date"] = out["discharge_date"].dt.date.astype(object).where(out["discharge_date"].notna(), None)
    return out.sort_values(["paid_at", "claim_id"]).reset_index(drop=True)


def load(frame: pd.DataFrame, eng, seed: int = 2026, replace: bool = False) -> int:
    scheduled = schedule(frame, seed)
    records = [{k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in row.items()}
               for row in scheduled.to_dict(orient="records")]
    for row in records:
        for key in ("lodged_at", "approved_at", "paid_at"):
            row[key] = row[key].to_pydatetime()
    with eng.begin() as conn:
        if replace:
            conn.execute(delete(db.claims))
        elif conn.execute(select(func.count()).select_from(db.claims)).scalar():
            raise ValueError("the platform already holds claims; pass --replace to start over")
        # Restart the clock before the first payment. Also on a first load: a read of the empty
        # platform (the ingest worker polls it) creates the clock at today's date, which would
        # count every claim as already paid.
        conn.execute(delete(db.sim_clock))
        conn.execute(insert(db.claims), records)
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Load a claims set into the simulated claims platform")
    parser.add_argument("claims", type=Path, help="a parquet or CSV claims set, e.g. live.parquet")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--replace", action="store_true", help="delete the platform's claims and clock first")
    args = parser.parse_args()
    frame = pd.read_parquet(args.claims) if args.claims.suffix == ".parquet" else pd.read_csv(args.claims)
    eng = db.engine()
    n = load(frame, eng, args.seed, args.replace)
    with eng.connect() as conn:
        first, last = conn.execute(select(func.min(db.claims.c.paid_at), func.max(db.claims.c.paid_at))).one()
    print(f"{n} claims loaded into {eng.url.render_as_string(hide_password=True)}; payments from {first:%Y-%m-%d} "
          f"to {last:%Y-%m-%d}")


if __name__ == "__main__":
    main()
