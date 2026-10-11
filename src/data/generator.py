"""Synthetic SHA claims generator, stage A of dataset construction (Section 3.2.1).

Produces the structured claims (no clinical notes yet) plus one note-generation request per
claim. Notes are written by an LLM in stage B (src/data/note_synthesis.py, run on Colab),
and claim-level fraud injection, text noise and splitting happen in stage C
(src/data/build_dataset.py). Notes must exist before claim-level injection because
upcoding, ghost billing and fabrication all work by leaving or swapping a note that no
longer matches the billed code.

The two provider-level typologies of Table 3.1 are generated here, because they change
claim volumes rather than editing existing claims:
  - row 4, inflated caesarean-section rates: flagged providers bill extra caesarean
    deliveries until their caesarean share far exceeds the WHO-recommended 10-15%;
  - row 6, anomalous dependent counts: flagged providers bill for fictitious dependents
    attached to real member households.
Only the excess claims (the extra caesareans, the fictitious dependents' claims) are
labelled, so the positive class stays tied to specific fraudulent claims.

Tariffs and ICD-11 codes come from sha_catalogue.yaml (LN 56/2025; WHO ICD-11 2026-01).

Usage: python -m src.data.generator --n-claims 5000 --seed 42 --out data/raw
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CATALOGUE_PATH = Path(__file__).with_name("sha_catalogue.yaml")

# KEPH facility levels and the share of synthetic providers at each. Lower levels are
# more numerous, as in the Kenyan facility register; Level 6 is national referral.
FACILITY_TYPES = {
    2: "dispensary",
    3: "health_centre",
    4: "sub_county_hospital",
    5: "county_referral_hospital",
    6: "national_referral_hospital",
}
LEVEL_SHARES = {2: 0.30, 3: 0.30, 4: 0.25, 5: 0.12, 6: 0.03}
# Relative claim volume per provider at each level (a referral hospital sees more claims).
LEVEL_VOLUME = {2: 1.0, 3: 1.5, 4: 3.0, 5: 5.0, 6: 8.0}

DELIVERY_CODES = {"normal": "JB20.0", "caesarean": "JB22.1"}
CAESAREAN_CODES = {"JB22.0", "JB22.1"}


@dataclass
class GeneratorConfig:
    n_claims: int = 5000
    n_providers: int = 60
    seed: int = 42
    # Service window: the first six months after LN 56/2025 tariffs took effect.
    start_date: str = "2025-04-01"
    end_date: str = "2025-09-30"
    # Overall fraud prevalence, split evenly across the six Table 3.1 typologies. This
    # stage generates the two provider-level ones; stage C injects the other four.
    fraud_prevalence: float = 0.10
    n_typologies: int = 6
    # Caesarean share a flagged provider is pushed to (WHO recommended max is 0.15).
    inflated_caesarean_share: tuple[float, float] = (0.50, 0.70)
    # Real household sizes stay small; a flagged provider adds this many fictitious
    # dependents to a household it bills for.
    realistic_max_dependents: int = 6
    fictitious_dependents: tuple[int, int] = (10, 30)

    @property
    def per_typology_target(self) -> int:
        return round(self.n_claims * self.fraud_prevalence / self.n_typologies)


def load_catalogue(path: Path = CATALOGUE_PATH) -> dict:
    with open(path) as f:
        catalogue = yaml.safe_load(f)
    codes = [c["code"] for c in catalogue["conditions"]]
    if len(codes) != len(set(codes)):
        raise ValueError("Duplicate ICD-11 codes in catalogue")
    for condition in catalogue["conditions"]:
        for target in condition.get("upcode_to", []):
            if target not in codes:
                raise ValueError(f"{condition['code']} upcodes to {target}, which is not in the catalogue")
    return catalogue


def generate_providers(n_providers: int, rng: np.random.Generator) -> pd.DataFrame:
    levels = rng.choice(list(LEVEL_SHARES), size=n_providers, p=list(LEVEL_SHARES.values()))
    # Guarantee at least two providers at every level so every catalogue condition has
    # somewhere to be billed.
    for i, level in enumerate(LEVEL_SHARES):
        levels[2 * i: 2 * i + 2] = level
    providers = pd.DataFrame({
        "provider_id": [f"PRV{i:04d}" for i in range(1, n_providers + 1)],
        "facility_level": levels.astype(int),
    })
    providers["facility_type"] = providers["facility_level"].map(FACILITY_TYPES)
    return providers


class _ClaimFactory:
    """Builds individual claim rows from catalogue conditions."""

    def __init__(self, catalogue: dict, providers: pd.DataFrame, config: GeneratorConfig,
                 rng: np.random.Generator):
        self.conditions = {c["code"]: c for c in catalogue["conditions"]}
        self.per_diem = {int(k): float(v) for k, v in catalogue["per_diem_rates_kes"].items()}
        self.providers = providers
        self.config = config
        self.rng = rng
        self.start = pd.Timestamp(config.start_date)
        self.window_days = (pd.Timestamp(config.end_date) - self.start).days
        self._next_member = 1

    def new_member_id(self) -> str:
        member_id = f"MBR{self._next_member:06d}"
        self._next_member += 1
        return member_id

    def eligible_providers(self, code: str) -> pd.DataFrame:
        levels = self.conditions[code]["levels"]
        return self.providers[self.providers["facility_level"].isin(levels)]

    def pick_provider(self, code: str) -> pd.Series:
        eligible = self.eligible_providers(code)
        weights = eligible["facility_level"].map(LEVEL_VOLUME).to_numpy()
        return eligible.iloc[self.rng.choice(len(eligible), p=weights / weights.sum())]

    def patient(self, code: str) -> tuple[int, str]:
        condition = self.conditions[code]
        age = int(self.rng.integers(condition["age"][0], condition["age"][1] + 1))
        sex = condition["sex"] if condition["sex"] != "any" else self.rng.choice(["female", "male"])
        return age, str(sex)

    def claim(self, code: str, provider: pd.Series, member_id: str, beneficiary_type: str,
              age: int | None = None, sex: str | None = None) -> dict:
        condition = self.conditions[code]
        if age is None or sex is None:
            age, sex = self.patient(code)
        stay = int(self.rng.integers(condition["los"][0], condition["los"][1] + 1))
        service_date = self.start + pd.Timedelta(days=int(self.rng.integers(0, self.window_days + 1)))
        if condition["payment"] == "capitation":
            amount = 0.0
        elif condition["payment"] == "per_diem":
            amount = self.per_diem[int(provider["facility_level"])] * max(stay, 1)
        else:
            amount = float(condition["tariff_kes"])
        return {
            "provider_id": provider["provider_id"],
            "facility_level": int(provider["facility_level"]),
            "facility_type": provider["facility_type"],
            "member_id": member_id,
            "beneficiary_type": beneficiary_type,
            "patient_age": age,
            "patient_sex": sex,
            "icd11_code": code,
            "package": condition["package"],
            "payment_mechanism": condition["payment"],
            "procedure": condition.get("procedure", ""),
            "claim_type": "outpatient" if stay == 0 else "inpatient",
            "service_date": service_date.date().isoformat(),
            "discharge_date": (service_date + pd.Timedelta(days=stay)).date().isoformat(),
            "claim_amount": amount,
            "tariff": amount,
        }


def _base_claims(factory: _ClaimFactory, n: int, catalogue: dict) -> list[dict]:
    rng = factory.rng
    codes = [c["code"] for c in catalogue["conditions"]]
    weights = np.array([c["weight"] for c in catalogue["conditions"]], dtype=float)
    # A shared pool of households so members recur across claims, as in real data.
    n_households = max(1, n // 3)
    household_ids = [factory.new_member_id() for _ in range(n_households)]
    rows = []
    for code in rng.choice(codes, size=n, p=weights / weights.sum()):
        age, sex = factory.patient(code)
        beneficiary_type = "dependent" if age < 18 or rng.random() < 0.35 else "principal"
        member_id = household_ids[int(rng.integers(0, n_households))]
        rows.append(factory.claim(code, factory.pick_provider(code), member_id,
                                  beneficiary_type, age, sex))
    return rows


def _inflated_caesarean_claims(factory: _ClaimFactory, base: pd.DataFrame,
                               target: int) -> list[dict]:
    """Table 3.1 row 4. Flag providers that can perform caesareans and give each enough
    extra caesarean deliveries to push its caesarean share into the configured inflated
    range, until the extra claims reach `target`."""
    rng = factory.rng
    config = factory.config
    deliveries = base[base["icd11_code"].isin({DELIVERY_CODES["normal"], *CAESAREAN_CODES})]
    candidates = factory.eligible_providers(DELIVERY_CODES["caesarean"])["provider_id"].tolist()
    rng.shuffle(candidates)
    rows: list[dict] = []
    for provider_id in candidates:
        if len(rows) >= target:
            break
        at_provider = deliveries[deliveries["provider_id"] == provider_id]
        n_caesarean = int(at_provider["icd11_code"].isin(CAESAREAN_CODES).sum())
        n_normal = len(at_provider) - n_caesarean
        if n_normal < 3:
            continue
        share = rng.uniform(*config.inflated_caesarean_share)
        # Solve (c + e) / (n + c + e) = share for the number of extra caesareans e.
        extra = int(np.ceil((share * (n_normal + n_caesarean) - n_caesarean) / (1 - share)))
        # A provider capped below its inflated share would be labelled fraud while looking
        # statistically normal, so skip it rather than give it a partial top-up.
        if extra < 1 or extra > target - len(rows):
            continue
        provider = factory.providers.set_index("provider_id").loc[provider_id]
        provider = pd.concat([pd.Series({"provider_id": provider_id}), provider])
        for _ in range(extra):
            row = factory.claim(DELIVERY_CODES["caesarean"], provider,
                                factory.new_member_id(), "principal")
            row["is_inflated_csection"] = True
            rows.append(row)
    return rows


def _fictitious_dependent_claims(factory: _ClaimFactory, catalogue: dict, base: pd.DataFrame,
                                 target: int) -> list[dict]:
    """Table 3.1 row 6. A flagged provider attaches many fictitious dependents to a real
    household it already bills for, far beyond `realistic_max_dependents`, and bills one
    claim per fictitious dependent. Real household members' claims stay unlabelled."""
    rng = factory.rng
    config = factory.config
    rows: list[dict] = []
    provider_ids = base["provider_id"].unique().tolist()
    rng.shuffle(provider_ids)
    for provider_id in provider_ids:
        if len(rows) >= target:
            break
        provider = factory.providers.set_index("provider_id").loc[provider_id]
        provider = pd.concat([pd.Series({"provider_id": provider_id}), provider])
        level = int(provider["facility_level"])
        offered = [c for c in catalogue["conditions"] if level in c["levels"]]
        codes = [c["code"] for c in offered]
        weights = np.array([c["weight"] for c in offered], dtype=float)
        household = rng.choice(base.loc[base["provider_id"] == provider_id, "member_id"].unique())
        n_fictitious = int(rng.integers(*config.fictitious_dependents, endpoint=True))
        for _ in range(min(n_fictitious, target - len(rows))):
            code = rng.choice(codes, p=weights / weights.sum())
            row = factory.claim(code, provider, household, "dependent")
            row["is_anomalous_dependent"] = True
            rows.append(row)
    return rows


def generate_base_claims(catalogue: dict, config: GeneratorConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (claims, providers). `claims` has exactly config.n_claims rows, with the
    provider-level typology flags set and no clinical notes."""
    rng = np.random.default_rng(config.seed)
    providers = generate_providers(config.n_providers, rng)
    factory = _ClaimFactory(catalogue, providers, config, rng)
    target = config.per_typology_target

    base = pd.DataFrame(_base_claims(factory, config.n_claims - 2 * target, catalogue))
    caesarean = _inflated_caesarean_claims(factory, base, target)
    dependents = _fictitious_dependent_claims(factory, catalogue, base, target)
    # Top up with clean claims if either typology fell short of its target, so the
    # dataset always has exactly n_claims rows.
    shortfall = 2 * target - len(caesarean) - len(dependents)
    top_up = _base_claims(factory, shortfall, catalogue) if shortfall > 0 else []

    claims = pd.concat([base, pd.DataFrame(caesarean + dependents + top_up)], ignore_index=True)
    for flag in ("is_inflated_csection", "is_anomalous_dependent"):
        claims[flag] = claims.get(flag, pd.Series(False, index=claims.index)).eq(True)
    claims = claims.sample(frac=1.0, random_state=config.seed).reset_index(drop=True)
    claims.insert(0, "claim_id", [f"CLM{i:06d}" for i in range(1, len(claims) + 1)])
    return claims, providers


def note_requests(claims: pd.DataFrame, catalogue: dict) -> list[dict]:
    """One note-generation request per claim, describing the claim's true (pre-injection)
    clinical situation. Stage B turns each into a clinical justification."""
    titles = {c["code"]: c["title"] for c in catalogue["conditions"]}
    return [
        {
            "claim_id": row.claim_id,
            "icd11_code": row.icd11_code,
            "diagnosis": titles[row.icd11_code],
            "procedure": row.procedure,
            "package": row.package,
            "patient_age": int(row.patient_age),
            "patient_sex": row.patient_sex,
            "facility_type": row.facility_type,
            "length_of_stay": (pd.Timestamp(row.discharge_date) - pd.Timestamp(row.service_date)).days,
        }
        for row in claims.itertuples()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n-claims", type=int, default=5000)
    parser.add_argument("--n-providers", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    config = GeneratorConfig(n_claims=args.n_claims, n_providers=args.n_providers, seed=args.seed)
    catalogue = load_catalogue()
    claims, providers = generate_base_claims(catalogue, config)

    args.out.mkdir(parents=True, exist_ok=True)
    claims.to_parquet(args.out / "base_claims.parquet", index=False)
    providers.to_parquet(args.out / "providers.parquet", index=False)
    with open(args.out / "note_requests.jsonl", "w") as f:
        for request in note_requests(claims, catalogue):
            f.write(json.dumps(request) + "\n")
    (args.out / "generator_config.json").write_text(json.dumps(asdict(config), indent=2))
    print(f"{len(claims)} claims, {len(providers)} providers -> {args.out}")
    print(f"  inflated caesarean claims: {int(claims['is_inflated_csection'].sum())}")
    print(f"  fictitious dependent claims: {int(claims['is_anomalous_dependent'].sum())}")


if __name__ == "__main__":
    main()
