"""Cost-weighted reporting (Section 3.2.4's economic frame, task 7.3).

For one system on one split: amount-weighted recall (fraudulent KES flagged / all
fraudulent KES), fraudulent KES missed, and auditor-hours = flagged claims x minutes per
review. Minutes per review is a stated assumption, so the result is given at several
values (default 20 and 40). Ground truth is always `is_fraud`: this is money lost to fraud.

    python -m src.evaluation.cost --data data/processed/v0.6-20k --split validation \\
        --systems pipeline rule_based --minutes 20 40
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CAPITATION_NOTE = ("Capitation claims carry KES 0, so amount-weighted recall covers only "
                   "case-rate and per-diem fraud.")
DEFAULT_MINUTES = (20, 40)


def cost_summary(amount, is_fraud, flagged, minutes=DEFAULT_MINUTES) -> dict:
    amount = np.asarray(amount, dtype=float)
    fraud = np.asarray(is_fraud).astype(bool)
    flagged = np.asarray(flagged).astype(bool)
    total = float(amount[fraud].sum())
    caught = float(amount[fraud & flagged].sum())
    return {
        "flagged_claims": int(flagged.sum()),
        "fraud_kes_total": round(total, 2),
        "fraud_kes_flagged": round(caught, 2),
        "fraud_kes_missed": round(total - caught, 2),
        "amount_weighted_recall": round(caught / total, 4) if total else None,
        "auditor_hours": {f"{m:g}": round(int(flagged.sum()) * m / 60, 1) for m in minutes},
    }


def main() -> None:
    from src.evaluation.heldout import guard_split, join_predictions, load_split
    from src.evaluation.predictions import PREDICTIONS_DIR

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True, help="processed dataset directory")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--final", action="store_true", help="required to read the test split")
    parser.add_argument("--systems", nargs="+", required=True)
    parser.add_argument("--predictions-root", type=Path, default=PREDICTIONS_DIR)
    parser.add_argument("--minutes", type=float, nargs="+", default=list(DEFAULT_MINUTES),
                        help="assumed minutes per auditor review")
    args = parser.parse_args()
    guard_split(parser, args.split, args.final)

    claims = load_split(args.data, args.split)
    result = {"split": args.split, "minutes_per_review": args.minutes, "note": CAPITATION_NOTE,
              "systems": {}}
    for system in args.systems:
        frame = join_predictions(claims, system, args.split, args.predictions_root)
        result["systems"][system] = cost_summary(frame["claim_amount"], frame["is_fraud"],
                                                 frame["flagged"], args.minutes)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
