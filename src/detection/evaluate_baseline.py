"""Isolation Forest baseline on the synthetic claims dataset (Sprint 4; Section 3.2.3).

Fits the preprocessing and the forest on the training split, tunes contamination on the
validation split by F1 (the [0.05, 0.40] grid) unless --contamination fixes it, and reports validation precision, recall
and F1 against `is_fraud`, plus recall per fraud typology. The test split is never read:
it is reserved for the final evaluation (Section 3.2.4).

    python -m src.detection.evaluate_baseline --data data/processed/v0.4-5k --out experiments

With --predictions it also writes the validation predictions in the shared format
(experiments/predictions/isolation_forest/validation.csv, src/evaluation/predictions.py),
as the Isolation-Forest-only baseline of Section 3.2.4. --contamination uses a fixed value
(e.g. the one task 6.2 chose) instead of tuning.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

from src.data.preprocessing import FEATURE_COLUMNS, ClaimsPreprocessor
from src.detection.isolation_forest import IsolationForestDetector
from src.evaluation.predictions import write_predictions


def evaluate(data_dir: Path, seed: int = 42, contamination: float | None = None,
             predictions: bool = False) -> dict:
    """`contamination` fixes the forest's contamination instead of tuning it on validation.
    `predictions` writes the validation predictions (flagged, anomaly score) in the shared
    format."""
    train = pd.read_parquet(data_dir / "train.parquet")
    validation = pd.read_parquet(data_dir / "validation.parquet")

    preprocessor = ClaimsPreprocessor()
    train_x = preprocessor.feature_matrix(preprocessor.align_schema(train), fit=True)
    val_x = preprocessor.feature_matrix(preprocessor.align_schema(validation))
    val_y = validation["is_fraud"].to_numpy()

    if contamination is None:
        contamination, detector = IsolationForestDetector.tune_contamination(
            train_x, val_x, val_y, random_state=seed)
        tuned = True
    else:
        detector = IsolationForestDetector(contamination=contamination, random_state=seed).fit(train_x)
        tuned = False
    flagged = detector.model.predict(val_x) == -1
    scores = detector.score(val_x)
    precision, recall, f1, _ = precision_recall_fscore_support(
        val_y, flagged, average="binary", zero_division=0)

    if predictions:
        write_predictions("isolation_forest", "validation", pd.DataFrame(
            {"claim_id": validation["claim_id"].to_numpy(), "flagged": flagged, "score": scores}))

    by_typology = (pd.DataFrame({"typology": validation["fraud_typology"], "flagged": flagged})
                   .groupby("typology")["flagged"].agg(["size", "sum"]))
    return {
        "model": "IsolationForest",
        "dataset": data_dir.name,
        "seed": seed,
        "features": FEATURE_COLUMNS,
        "contamination": contamination,
        "contamination_tuned": tuned,
        "split": "validation",
        "claims": len(validation),
        "flagged": int(flagged.sum()),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(roc_auc_score(val_y, scores)), 4),
        "average_precision": round(float(average_precision_score(val_y, scores)), 4),
        # For "none" this is the share of clean claims flagged (false positive rate).
        "flagged_by_typology": {
            typology: {"claims": int(row["size"]), "flagged": int(row["sum"]),
                       "rate": round(row["sum"] / row["size"], 4)}
            for typology, row in by_typology.iterrows()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True, help="processed dataset directory")
    parser.add_argument("--out", type=Path, default=Path("experiments"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--contamination", type=float,
                        help="fixed contamination instead of tuning it on validation")
    parser.add_argument("--predictions", action="store_true",
                        help="also write experiments/predictions/isolation_forest/validation.csv")
    args = parser.parse_args()

    results = evaluate(args.data, args.seed, args.contamination, args.predictions)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.out / "isolation_forest" / f"{args.data.name}_{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    print(f"contamination {results['contamination']}: {results['flagged']} of {results['claims']} flagged")
    print(f"precision {results['precision']}  recall {results['recall']}  f1 {results['f1']}  "
          f"roc_auc {results['roc_auc']}  average_precision {results['average_precision']}")
    for typology, row in results["flagged_by_typology"].items():
        print(f"  {typology:<25} {row['flagged']:>3}/{row['claims']:<4} {row['rate']:.0%}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
