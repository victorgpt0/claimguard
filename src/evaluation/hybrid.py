"""Configuration C (decided on validation, 25 Oct, before the test split was read): the
pipeline's text lane together with the five static rules, in place of the anomaly-detector
lane. A claim is reported when the classifier judged its notes inconsistent with the billed
code (pipeline_text_lane) or any rule fired (rule_based).

On v0.6 validation the anomaly-detector lane carried most of the pipeline's false positives
(12.4% false-positive rate, 126 of 401 on claims with noisy structured fields), while the
rules caught the inflated-caesarean typology the pipeline misses. Configuration A (the
pipeline as designed) and C are both reported on the test split. The rules encode the same
SHA typologies the generator injects, so their share of C's result is partly by
construction; Chapter 5 states this.

    python -m src.evaluation.hybrid --split validation --predictions-root <root>
    python -m src.evaluation.hybrid --split test --final --predictions-root <root>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.evaluation.predictions import PREDICTIONS_DIR, read_predictions, write_predictions

SYSTEM = "hybrid_text_rules"
PARTS = ("pipeline_text_lane", "rule_based")


def combine(text_lane: pd.DataFrame, rules: pd.DataFrame) -> pd.DataFrame:
    frame = text_lane[["claim_id", "flagged"]].merge(rules[["claim_id", "flagged"]], on="claim_id",
                                                     suffixes=("_text", "_rules"))
    if not len(frame) == len(text_lane) == len(rules):
        raise ValueError("the text-lane and rule-based predictions do not cover the same claims")
    return pd.DataFrame({"claim_id": frame["claim_id"],
                         "flagged": (frame["flagged_text"] | frame["flagged_rules"]).astype(int)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", default="validation")
    parser.add_argument("--final", action="store_true", help="required for the test split")
    parser.add_argument("--predictions-root", type=Path, default=PREDICTIONS_DIR)
    args = parser.parse_args()
    if args.split == "test" and not args.final:
        parser.error("the test split is reserved for the final evaluation; pass --final")
    parts = [read_predictions(system, args.split, args.predictions_root) for system in PARTS]
    print(f"-> {write_predictions(SYSTEM, args.split, combine(*parts), args.predictions_root)}")


if __name__ == "__main__":
    main()
