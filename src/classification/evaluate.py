"""Stratified evaluation of a trained classifier checkpoint (Section 3.2.4).

Scores one exported split (src.classification.export) and breaks the result down by fraud
typology, by clean versus noisy text and structured fields, and by common versus rare
ICD-11 codes. Writes the per-claim predictions next to the summary, so later analysis
needs no rerun.

The test split is reserved for the final evaluation: scoring it needs `--final`, so it
cannot happen by accident while models are still being developed.

    python -m src.classification.evaluate --checkpoint experiments/classifier/v0.5-5k/<run_id> \\
        --data data/classifier/v0.5-5k --split validation
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from src.classification.classifier import SemanticConsistencyClassifier
from src.classification.train import default_device, predict_probabilities

# A code is "rare" when it accounts for under 1% of training claims.
RARE_CODE_SHARE = 0.01


def _metrics(frame: pd.DataFrame) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        frame["actual"], frame["predicted"], average="binary", zero_division=0)
    return {"claims": len(frame), "inconsistent": int(frame["actual"].sum()),
            "flagged": int(frame["predicted"].sum()), "precision": round(float(precision), 4),
            "recall": round(float(recall), 4), "f1": round(float(f1), 4)}


def stratify(predictions: pd.DataFrame, train_codes: list[str]) -> dict:
    """`predictions` has one row per claim: actual, predicted (0/1), fraud_typology,
    has_text_noise, has_structured_noise, icd11_code."""
    counts = Counter(train_codes)
    rare = {code for code, n in counts.items() if n / len(train_codes) < RARE_CODE_SHARE}
    code_group = predictions["icd11_code"].map(lambda c: "rare" if c in rare or c not in counts else "common")
    by_typology = {
        typology: {"claims": len(group), "flagged": int(group["predicted"].sum()),
                   "rate": round(float(group["predicted"].mean()), 4)}
        for typology, group in predictions.groupby("fraud_typology")
    }
    return {
        "overall": _metrics(predictions),
        # For the three inconsistent typologies the rate is recall; for the others it is
        # the share wrongly flagged (their notes are consistent with the billed code).
        "flagged_by_typology": by_typology,
        "text_noise": {str(k): _metrics(g) for k, g in predictions.groupby("has_text_noise")},
        "structured_noise": {str(k): _metrics(g) for k, g in predictions.groupby("has_structured_noise")},
        "code_frequency": {k: _metrics(g) for k, g in predictions.groupby(code_group)},
        "rare_code_threshold": f"< {RARE_CODE_SHARE:.0%} of training claims ({len(rare)} codes)",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True, help="run folder from train.py")
    parser.add_argument("--data", type=Path, required=True, help="exported JSONL directory")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--final", action="store_true", help="required to score the test split")
    parser.add_argument("--base-model", default="emilyalsentzer/Bio_ClinicalBERT")
    args = parser.parse_args()
    if args.split == "test" and not args.final:
        parser.error("the test split is reserved for the final evaluation; pass --final to score it")

    examples = [json.loads(line) for line in open(args.data / f"{args.split}.jsonl")]
    train_codes = [json.loads(line)["icd11_code"] for line in open(args.data / "train.jsonl")]
    device = default_device()
    classifier = SemanticConsistencyClassifier(str(args.checkpoint), args.base_model, device)
    if classifier.model_version == "untrained-base":
        parser.error(f"no checkpoint found in {args.checkpoint}")
    probabilities = predict_probabilities(classifier.model, examples, classifier.tokenizer, device)

    predictions = pd.DataFrame(examples).drop(columns=["clinical_notes", "icd11_definition"])
    predictions["probability"] = probabilities
    predictions["predicted"] = (predictions["probability"] > 0.5).astype(int)
    predictions["actual"] = (predictions["label"] == "inconsistent").astype(int)
    result = {"checkpoint": args.checkpoint.name, "split": args.split, **stratify(predictions, train_codes)}

    out = args.checkpoint / f"evaluation_{args.split}.json"
    out.write_text(json.dumps(result, indent=2))
    predictions.to_json(args.checkpoint / f"predictions_{args.split}.jsonl", orient="records", lines=True)
    print(json.dumps(result, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
