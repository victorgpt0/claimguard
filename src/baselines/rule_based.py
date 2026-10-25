"""Rule-based baseline (Section 3.2.4): the five static rules a conventional claims filter
applies to structured billing fields, reconstructed from the documented SHA typologies.
It is a reconstruction, not SHA's real system.

The thresholds below are fixed before evaluation and never tuned on any split: a static
filter does not learn, and tuning it on the evaluation data would flatter it. A claim is
flagged when any rule fires (logical OR); `score` is the number of rules that fired.

"Provider history" for rules (i), (iv) and (v) is every claim of that provider in the
batch being scored, as a post-payment audit sees a batch of submitted claims.

    python -m src.baselines.rule_based --data data/processed/v0.5-5k --split validation
    python -m src.baselines.rule_based --data data/processed/v0.5-5k --final   # test split
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data.generator import load_catalogue
from src.data.preprocessing import CAESAREAN_CODES, DELIVERY_CODES, repair_icd11_code
from src.evaluation.predictions import PREDICTIONS_DIR, write_predictions

# (i) Upcoding: a claim billed above this multiple of the median amount the same provider
# bills for the same ICD-11 code. KES 0 claims (capitation-paid outpatient visits) are
# never flagged: there is no per-claim amount to inflate.
UPCODING_MEDIAN_MULTIPLE = 1.5

# (ii) Outpatient-to-inpatient conversion: an inpatient claim for a condition paid by
# capitation (Primary Healthcare Fund outpatient care), with a stay this short or shorter.
CONVERSION_MAX_STAY_DAYS = 1
OUTPATIENT_ONLY_PAYMENT = "capitation"

# (iv) Caesarean rate: WHO's recommended population caesarean rate (15%) plus a fixed
# 10-point margin, so only providers far above the guideline are flagged.
WHO_CAESAREAN_RATE = 0.15
CAESAREAN_MARGIN = 0.10
CAESAREAN_SHARE_THRESHOLD = WHO_CAESAREAN_RATE + CAESAREAN_MARGIN

# (v) Dependents: more than this many distinct dependents billed under one principal
# member at one provider. The generator's realistic household assumption
# (GeneratorConfig.realistic_max_dependents) is at most 6 dependents per household, so
# anything above 6 cannot be a real household. Claims carry no dependent identifier, only
# the principal's member_id, so a dependent is identified by (member_id, age, sex).
MAX_DEPENDENTS_PER_PRINCIPAL = 6

RULES = ["rule_upcoding", "rule_conversion", "rule_duplicate", "rule_caesarean_rate",
         "rule_dependents"]
SPLITS = ("train", "validation", "test", "deployment_pool")
OUT_DIR = Path("experiments/baselines/rule_based")


def outpatient_only_codes(catalogue: dict | None = None) -> set[str]:
    catalogue = catalogue or load_catalogue()
    return {c["code"] for c in catalogue["conditions"] if c["payment"] == OUTPATIENT_ONLY_PAYMENT}


def _prepare(claims: pd.DataFrame) -> pd.DataFrame:
    out = claims.copy()
    # Malformed codes (structured noise) are repaired first, as any intake check would.
    out["icd11_code"] = out["icd11_code"].map(repair_icd11_code)
    out["service_date"] = pd.to_datetime(out["service_date"])
    out["discharge_date"] = pd.to_datetime(out["discharge_date"])
    out["claim_amount"] = out["claim_amount"].astype(float)
    return out


def rule_upcoding(claims: pd.DataFrame) -> pd.Series:
    median = claims.groupby(["provider_id", "icd11_code"])["claim_amount"].transform("median")
    return (claims["claim_amount"] > 0) & (claims["claim_amount"] > UPCODING_MEDIAN_MULTIPLE * median)


def rule_conversion(claims: pd.DataFrame, outpatient_only: set[str]) -> pd.Series:
    stay = (claims["discharge_date"] - claims["service_date"]).dt.days
    # A blank discharge date gives no stay, so the rule cannot fire on it.
    return ((claims["claim_type"] == "inpatient") & (stay <= CONVERSION_MAX_STAY_DAYS)
            & claims["icd11_code"].isin(outpatient_only))


def rule_duplicate(claims: pd.DataFrame) -> pd.Series:
    """Every claim that shares member, code and service date with another claim."""
    return claims.duplicated(["member_id", "icd11_code", "service_date"], keep=False)


def rule_caesarean_rate(claims: pd.DataFrame) -> pd.Series:
    deliveries = claims[claims["icd11_code"].isin(DELIVERY_CODES)]
    share = deliveries["icd11_code"].isin(CAESAREAN_CODES).groupby(deliveries["provider_id"]).mean()
    over = share[share > CAESAREAN_SHARE_THRESHOLD].index
    return claims["icd11_code"].isin(CAESAREAN_CODES) & claims["provider_id"].isin(over)


def rule_dependents(claims: pd.DataFrame) -> pd.Series:
    dependent = claims["beneficiary_type"] == "dependent"
    people = claims.loc[dependent, ["provider_id", "member_id", "patient_age", "patient_sex"]]
    counts = (people.drop_duplicates()
              .groupby(["provider_id", "member_id"]).size())
    over = counts[counts > MAX_DEPENDENTS_PER_PRINCIPAL].index
    keys = pd.MultiIndex.from_frame(claims[["provider_id", "member_id"]])
    return dependent & pd.Series(keys.isin(over), index=claims.index)


def apply_rules(claims: pd.DataFrame, outpatient_only: set[str] | None = None) -> pd.DataFrame:
    """Returns claim_id, flagged, score (rules fired) and one boolean column per rule."""
    outpatient_only = outpatient_only if outpatient_only is not None else outpatient_only_codes()
    prepared = _prepare(claims)
    out = pd.DataFrame({"claim_id": prepared["claim_id"]})
    out["rule_upcoding"] = rule_upcoding(prepared)
    out["rule_conversion"] = rule_conversion(prepared, outpatient_only)
    out["rule_duplicate"] = rule_duplicate(prepared)
    out["rule_caesarean_rate"] = rule_caesarean_rate(prepared)
    out["rule_dependents"] = rule_dependents(prepared)
    out[RULES] = out[RULES].fillna(False).astype(bool)
    out["score"] = out[RULES].sum(axis=1).astype(float)
    out["flagged"] = out[RULES].any(axis=1)
    return out[["claim_id", "flagged", "score", *RULES]].reset_index(drop=True)


def rule_counts(results: pd.DataFrame, claims: pd.DataFrame | None = None) -> dict:
    """Claims each rule fired on, and, given the ground truth, how many were fraud."""
    counts = {"claims": len(results), "flagged": int(results["flagged"].sum()),
              "thresholds": {
                  "upcoding_median_multiple": UPCODING_MEDIAN_MULTIPLE,
                  "conversion_max_stay_days": CONVERSION_MAX_STAY_DAYS,
                  "caesarean_share_threshold": CAESAREAN_SHARE_THRESHOLD,
                  "max_dependents_per_principal": MAX_DEPENDENTS_PER_PRINCIPAL},
              "rules": {}}
    truth = None
    if claims is not None and "is_fraud" in claims:
        truth = results[["claim_id"]].merge(claims[["claim_id", "is_fraud", "fraud_typology"]],
                                             on="claim_id", how="left")
    for rule in RULES:
        fired = results[rule].to_numpy()
        row = {"fired": int(fired.sum())}
        if truth is not None:
            row["fraud"] = int(truth.loc[fired, "is_fraud"].sum())
            row["by_typology"] = truth.loc[fired, "fraud_typology"].value_counts().to_dict()
        counts["rules"][rule] = row
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True, help="processed dataset directory")
    parser.add_argument("--split", default="validation", choices=SPLITS)
    parser.add_argument("--final", action="store_true",
                        help="score the held-out test split (final evaluation only)")
    parser.add_argument("--out", type=Path, default=OUT_DIR, help="per-rule counts directory")
    parser.add_argument("--predictions-root", type=Path, default=PREDICTIONS_DIR)
    args = parser.parse_args()
    split = "test" if args.final else args.split
    if split == "test" and not args.final:
        parser.error("the test split is held out: pass --final to score it")

    claims = pd.read_parquet(args.data / f"{split}.parquet")
    results = apply_rules(claims)
    path = write_predictions("rule_based", split, results[["claim_id", "flagged", "score"]],
                             args.predictions_root)

    counts = {"dataset": args.data.name, "split": split, **rule_counts(results, claims)}
    args.out.mkdir(parents=True, exist_ok=True)
    counts_path = args.out / f"{args.data.name}_{split}.json"
    counts_path.write_text(json.dumps(counts, indent=2))

    print(f"{counts['flagged']} of {counts['claims']} claims flagged")
    for rule, row in counts["rules"].items():
        fraud = f"  ({row['fraud']} fraud)" if "fraud" in row else ""
        print(f"  {rule:<22} {row['fired']:>4}{fraud}")
    print(f"-> {path}\n-> {counts_path}")


if __name__ == "__main__":
    main()
