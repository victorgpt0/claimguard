"""Stage-1 tuning on the validation split (#28, #50; Section 3.2.3, NFR-05).

Stage 1 is the union of three components: Isolation Forest, the autoencoder and the
text-consistency screen. Each structured detector alone misses the text-only typologies
by design, so the recall floor is applied to stage 1 as a whole: over a grid of the three
settings, pick the combination whose union reaches `--min-recall` on `is_fraud` while
passing on the fewest clean claims (they cost classifier time and, for the structured
detectors, auditor time). Repeated over seeds; the test split is never read.

    python -m src.detection.tune_stage1 --data data/processed/v0.5-5k --cache artifacts/icd11_cache.json

The chosen values go into config.yaml's `thresholds:` block (FR-10), which
src.pipeline reads with `--config`.
"""
from __future__ import annotations

import argparse
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.preprocessing import ClaimsPreprocessor
from src.detection.autoencoder import AutoencoderDetector
from src.detection.isolation_forest import IsolationForestDetector
from src.detection.text_screen import TextConsistencyScreen, threshold_for_pass_rate
from src.icd11.cache import ICD11Cache

CONTAMINATIONS = [round(0.05 * i, 2) for i in range(1, 9)]  # 0.05..0.40, Section 3.2.3
AUTOENCODER_QUANTILES = [0.80, 0.85, 0.90, 0.95, 0.975]      # of training reconstruction errors
SCREEN_PASS_RATES = [0.05, 0.10, 0.15, 0.20]                 # target share of clean validation claims


def _rates(flags: np.ndarray, validation: pd.DataFrame) -> dict:
    fraud = validation["is_fraud"].to_numpy() == 1
    by_typology = (pd.Series(flags, index=validation.index)
                   .groupby(validation["fraud_typology"]).mean().round(3).to_dict())
    return {"recall": round(float(flags[fraud].mean()), 4),
            "clean_pass_rate": round(float(flags[~fraud].mean()), 4),
            "flagged": int(flags.sum()), "by_typology": by_typology}


def tune_seed(train: pd.DataFrame, validation: pd.DataFrame, cache: ICD11Cache, seed: int) -> dict:
    preprocessor = ClaimsPreprocessor()
    train_x = preprocessor.feature_matrix(preprocessor.align_schema(train), fit=True)
    val_x = preprocessor.feature_matrix(preprocessor.align_schema(validation))

    forest_flags = {}
    for contamination in CONTAMINATIONS:
        forest = IsolationForestDetector(contamination=contamination, random_state=seed).fit(train_x)
        forest_flags[contamination] = forest.model.predict(val_x) == -1

    torch.manual_seed(seed)
    autoencoder = AutoencoderDetector(input_dim=train_x.shape[1]).fit(train_x)
    train_errors, val_errors = autoencoder.score(train_x), autoencoder.score(val_x)
    autoencoder_flags = {q: val_errors >= np.quantile(train_errors, q) for q in AUTOENCODER_QUANTILES}

    screen = TextConsistencyScreen(cache).fit(train)
    similarity = screen.similarity(validation)
    clean = validation["is_fraud"].to_numpy() == 0
    screen_thresholds = {r: threshold_for_pass_rate(similarity[clean], r) for r in SCREEN_PASS_RATES}
    screen_flags = {r: similarity <= t for r, t in screen_thresholds.items()}

    grid = {}
    for c, q, r in itertools.product(CONTAMINATIONS, AUTOENCODER_QUANTILES, SCREEN_PASS_RATES):
        grid[(c, q, r)] = _rates(forest_flags[c] | autoencoder_flags[q] | screen_flags[r], validation)
    return {
        "seed": seed,
        "screen_thresholds": screen_thresholds,
        # At most the target; less when a tie straddles it (threshold_for_pass_rate).
        "screen_actual_clean_pass_rate": {r: round(float(f[clean].mean()), 4) for r, f in screen_flags.items()},
        "grid": grid,
        "components": lambda c, q, r: {
            "isolation_forest": _rates(forest_flags[c], validation),
            "autoencoder": _rates(autoencoder_flags[q], validation),
            "text_screen": _rates(screen_flags[r], validation),
            "structured_only": _rates(forest_flags[c] | autoencoder_flags[q], validation)},
        # The previous fixed setting (contamination 0.05, matching AE quantile, no screen).
        "previous_setting": _rates(forest_flags[0.05] | autoencoder_flags[0.95], validation),
    }


def choose(runs: list[dict], min_recall: float) -> tuple[tuple, bool]:
    """One setting for every seed: the grid point whose mean recall across seeds reaches
    `min_recall` with the lowest mean clean pass rate; ties go to the smaller
    contamination, since structured flags can reach the auditor's queue while screen-only
    flags of consistent notes cannot. If no point reaches it, the highest mean recall."""
    def mean(key, point):
        return float(np.mean([run["grid"][point][key] for run in runs]))
    points = list(runs[0]["grid"])
    feasible = [p for p in points if mean("recall", p) >= min_recall]
    if feasible:
        return min(feasible, key=lambda p: (mean("clean_pass_rate", p), p[0], -p[1])), True
    return max(points, key=lambda p: (mean("recall", p), -mean("clean_pass_rate", p))), False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True, help="processed dataset directory")
    parser.add_argument("--cache", type=Path, required=True, help="ICD-11 cache JSON")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024, 7, 31415])
    parser.add_argument("--min-recall", type=float, default=0.80)
    parser.add_argument("--out", type=Path, default=Path("experiments/stage1_tuning"))
    args = parser.parse_args()

    train = pd.read_parquet(args.data / "train.parquet")
    validation = pd.read_parquet(args.data / "validation.parquet")
    cache = ICD11Cache(args.cache)
    runs = [tune_seed(train, validation, cache, seed) for seed in args.seeds]

    point, met = choose(runs, args.min_recall)
    c, q, r = point
    recommended = {"isolation_forest_contamination": c, "autoencoder_train_quantile": q,
                   "text_screen_clean_pass_rate": r,
                   # Deterministic given the training split: the same for every seed.
                   "text_screen_threshold": round(runs[0]["screen_thresholds"][r], 6)}
    per_seed = [{"seed": run["seed"], **run["grid"][point],
                 "components": run["components"](c, q, r), "previous_setting": run["previous_setting"]}
                for run in runs]
    recalls = [s["recall"] for s in per_seed]
    summary = {
        "met_min_recall": met,
        "recall_mean": round(float(np.mean(recalls)), 4),
        "recall_std": round(float(np.std(recalls, ddof=1)), 4) if len(recalls) > 1 else None,
        "clean_pass_rate_mean": round(float(np.mean([s["clean_pass_rate"] for s in per_seed])), 4),
        "screen_actual_clean_pass_rate": runs[0]["screen_actual_clean_pass_rate"][r],
        "previous_recall_mean": round(float(np.mean([s["previous_setting"]["recall"] for s in per_seed])), 4),
    }
    record = {"data": args.data.name, "split": "validation", "min_recall": args.min_recall,
              "seeds": args.seeds, "recommended_thresholds": recommended, "summary": summary,
              "per_seed": per_seed}

    run_dir = args.out / f"{args.data.name}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.json").write_text(json.dumps(record, indent=2))
    print(json.dumps({"recommended_thresholds": recommended, "summary": summary,
                      "first_seed": per_seed[0]}, indent=2))
    print(f"-> {run_dir / 'results.json'}")


if __name__ == "__main__":
    main()
