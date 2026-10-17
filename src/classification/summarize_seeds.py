"""Multi-seed summary for the Semantic Consistency Classifier (Section 3.2.4, task 6.3).

train_all_seeds (src/classification/train.py) writes one experiments/<run_id>/ folder per
seed, holding metrics.json (validation metrics of the saved checkpoint) and config.json
(run_id, seed, hyperparameters). This reads every such folder under a directory and
reports validation precision, recall and F1 as mean +/- sample standard deviation over
seeds (Dodge et al., 2020), lists the per-seed values, and names the best-validation-F1
run, whose adapter becomes models/classifier_lora/active/ and evaluated_v1/ (D5).

Validation only: the held-out test set is not read here (Section 3.2.4).

Usage:
  python -m src.classification.summarize_seeds --runs experiments/classifier [--out summary.json]
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

METRICS = ["precision", "recall", "f1"]
DEFAULT_OUT_NAME = "seed_summary.json"


def load_runs(runs_dir: Path) -> list[dict]:
    """One record per <runs_dir>/*/metrics.json, with run_id and seed from the sibling
    config.json (the folder name and None when it is missing), sorted by run_id."""
    runs = []
    for metrics_path in sorted(runs_dir.glob("*/metrics.json")):
        metrics = json.loads(metrics_path.read_text())
        config_path = metrics_path.parent / "config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        runs.append({
            "run_id": config.get("run_id", metrics_path.parent.name),
            "seed": config.get("seed"),
            "epoch": metrics.get("epoch"),
            **{m: float(metrics[m]) for m in METRICS},
        })
    return sorted(runs, key=lambda r: r["run_id"])


def summarise(runs: list[dict]) -> dict:
    """Mean, sample std (ddof=1; None for a single run) and n per metric, the per-seed
    values, and the run with the highest validation F1 (first run_id on a tie)."""
    if not runs:
        raise ValueError("no */metrics.json runs found")
    stats = {}
    for m in METRICS:
        values = [r[m] for r in runs]
        stats[m] = {"mean": statistics.mean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else None,
                    "n": len(values)}
    best = max(runs, key=lambda r: r["f1"])  # max keeps the first of equal values
    return {"split": "validation", "metrics": stats, "runs": runs,
            "best": {"run_id": best["run_id"], "seed": best["seed"], "f1": best["f1"]}}


def format_summary(summary: dict) -> str:
    lines = ["Validation metrics over seeds (mean ± sample std):"]
    for m, s in summary["metrics"].items():
        std = "n/a" if s["std"] is None else f"{s['std']:.4f}"
        lines.append(f"  {m:<9} {s['mean']:.4f} ± {std}  (n={s['n']})")
    lines.append("Per seed:")
    for r in summary["runs"]:
        lines.append(f"  {r['run_id']} (seed {r['seed']}): precision {r['precision']:.4f}, "
                     f"recall {r['recall']:.4f}, f1 {r['f1']:.4f}")
    best = summary["best"]
    lines.append(f"Best validation F1: {best['run_id']} (seed {best['seed']}), f1 {best['f1']:.4f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", required=True, type=Path,
                        help="directory holding one <run_id>/ folder per seed")
    parser.add_argument("--out", type=Path, default=None,
                        help=f"JSON output (default: <runs>/{DEFAULT_OUT_NAME})")
    args = parser.parse_args(argv)

    summary = summarise(load_runs(args.runs))
    out = args.out or args.runs / DEFAULT_OUT_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(format_summary(summary))
    print(f"-> {out}")
    return summary


if __name__ == "__main__":
    main()
