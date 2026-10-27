"""SHAP coherence check (Section 3.2.4, task 7.6).

`sample` draws a typology-stratified sample of flagged claims from a pipeline run log
(src/pipeline.py: `claims` entries with claim_ref, flagged, reported and shap; reported
claims only by default, `--all-flagged` for every stage-1 flag) joined to the
dataset split for the typology, and writes a sheet with each claim's flag sources, the
claim fields the features are computed from, its top-3 SHAP features and blank
`coherent` (y/n) and `reason` columns. Marking guide: docs/shap_coherence_guide.md. Allocation is as equal as
possible across typologies; a typology with too few flagged claims gives its remainder
to the others. `summarise` reports the coherent share per typology.

    python -m src.evaluation.shap_coherence sample --log experiments/pipeline_runs/<run>.json \\
        --data data/processed/v0.6-20k
    python -m src.evaluation.shap_coherence summarise --sheet experiments/shap_coherence/sheet.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.evaluation.faithfulness import top_features

# Claim fields shown beside the features, so the rater can see what each one refers to.
CONTEXT_COLUMNS = ["facility_level", "beneficiary_type", "patient_age", "patient_sex", "claim_type",
                   "procedure", "claim_amount", "tariff", "stay_days"]


def allocate(available: dict[str, int], n: int) -> dict[str, int]:
    """Equal shares, capped at what each typology has, leftovers spread over the rest."""
    quota = {k: 0 for k in available}
    remaining = min(n, sum(available.values()))
    while remaining:
        open_ = sorted(k for k in available if quota[k] < available[k])
        share = max(remaining // len(open_), 1)
        for k in open_:
            take = min(share, available[k] - quota[k], remaining)
            quota[k] += take
            remaining -= take
            if not remaining:
                break
    return quota


def _eligible(entry: dict, reported_only: bool) -> bool:
    """Logs that record `reported` (D2: a report per pipeline positive) are sampled on it,
    so the sheet holds what an auditor sees; older logs on stage-1 `flagged`."""
    if reported_only and "reported" in entry:
        return bool(entry["reported"])
    return bool(entry.get("flagged"))


def sample_sheet(log: dict, claims: pd.DataFrame, n: int = 100, seed: int = 0,
                 reported_only: bool = True) -> pd.DataFrame:
    flagged = pd.DataFrame([c for c in log["claims"] if _eligible(c, reported_only)])
    flagged["claim_ref"] = flagged["claim_ref"].astype(str)
    truth = claims.assign(claim_id=claims["claim_id"].astype(str)).set_index("claim_id")
    if {"service_date", "discharge_date"} <= set(truth.columns):
        truth["stay_days"] = (pd.to_datetime(truth["discharge_date"]) - pd.to_datetime(truth["service_date"])).dt.days
    context = [c for c in CONTEXT_COLUMNS if c in truth.columns]
    flagged = flagged.join(truth[["fraud_typology", "icd11_code", *context]], on="claim_ref", how="inner")
    quota = allocate(flagged.groupby("fraud_typology").size().to_dict(), n)
    picked = pd.concat([group.sample(n=quota[t], random_state=seed)
                        for t, group in flagged.groupby("fraud_typology") if quota[t]])
    rows = []
    for c in picked.itertuples(index=False):
        shap = c.shap if isinstance(c.shap, dict) else {}
        row = {"claim_ref": c.claim_ref, "report_id": getattr(c, "report_id", None),
               "fraud_typology": c.fraud_typology, "icd11_code": c.icd11_code,
               "verdict": getattr(c, "verdict", None), "lane": getattr(c, "lane", None),
               "flag_sources": " ".join(getattr(c, "flag_sources", None) or []),
               **{col: getattr(c, col) for col in context}}
        features = top_features(shap)
        for i in range(3):
            row[f"feature_{i + 1}"] = features[i] if i < len(features) else ""
            row[f"shap_{i + 1}"] = round(float(shap[features[i]]), 4) if i < len(features) else ""
        rows.append(row | {"coherent": "", "reason": ""})
    return pd.DataFrame(rows).sort_values(["fraud_typology", "claim_ref"]).reset_index(drop=True)


def summarise(sheet: pd.DataFrame) -> dict:
    """Unmarked rows are left out; marks are y/n (case-insensitive)."""
    marks = sheet["coherent"].fillna("").astype(str).str.strip().str.lower()
    if not marks.isin(["", "y", "n"]).all():
        raise ValueError("coherent must be y, n or blank")
    marked = sheet[marks != ""].assign(coherent=marks[marks != ""] == "y")

    def block(frame):
        return {"marked": len(frame), "coherent": int(frame["coherent"].sum()),
                "share": round(float(frame["coherent"].mean()), 3) if len(frame) else None}
    return {"overall": block(marked),
            "by_typology": {t: block(g) for t, g in marked.groupby("fraud_typology")}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--log", type=Path, required=True, help="pipeline run log JSON")
    s.add_argument("--data", type=Path, required=True, help="processed dataset directory")
    s.add_argument("--split", help="default: the run's --batch setting")
    s.add_argument("--n", type=int, default=100)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", type=Path, default=Path("experiments/shap_coherence/sheet.csv"))
    s.add_argument("--force", action="store_true", help="overwrite an existing (possibly marked) sheet")
    s.add_argument("--all-flagged", action="store_true",
                   help="sample every stage-1 flag, including ones the text screen dropped")
    m = sub.add_parser("summarise")
    m.add_argument("--sheet", type=Path, default=Path("experiments/shap_coherence/sheet.csv"))
    args = parser.parse_args()

    if args.command == "sample":
        if args.out.exists() and not args.force:
            parser.error(f"{args.out} exists and may hold marks; pass --force to overwrite")
        log = json.loads(args.log.read_text())
        split = args.split or log.get("settings", {}).get("batch")
        if not split:
            parser.error("pass --split: the log does not record its batch")
        claims = pd.read_parquet(args.data / f"{split}.parquet")
        sheet = sample_sheet(log, claims, args.n, args.seed, reported_only=not args.all_flagged)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        sheet.to_csv(args.out, index=False)
        print(sheet["fraud_typology"].value_counts().to_string())
        print(f"{len(sheet)} claims -> {args.out}")
    else:
        result = {"sheet": str(args.sheet), **summarise(pd.read_csv(args.sheet, dtype={"coherent": str}))}
        out = args.sheet.with_name(args.sheet.stem + "_summary.json")
        out.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        print(f"-> {out}")


if __name__ == "__main__":
    main()
