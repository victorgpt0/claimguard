"""Held-out evaluation of every system (Section 3.2.4, task 7.1).

Joins each system's predictions (src.evaluation.predictions format) to the split's
ground truth and reports precision, recall, F1, false-positive rate and AUC-ROC, 95%
bootstrap confidence intervals, McNemar's test against a reference system, mean ± std
over training seeds, stratified results (src.evaluation.strata) and the cost summary
(src.evaluation.cost).

Ground truth is `is_fraud` for end-to-end comparisons (`--truth is_fraud`) and the text
label for the classifier-versus-zero-shot comparison (`--truth text`: label ==
"inconsistent"). Systems trained with several seeds are named `<base>_seed<N>` and are
grouped automatically for the mean ± std table.

The test split is reserved for the final evaluation: reading it needs `--final`, as in
src/classification/evaluate.py.

A system run on part of a split only (the zero-shot LLM with --limit) is compared with
`--claims-from <system>`: every system is then scored on the claims that system covers,
so the comparison stays paired, and the tables say so.

    python -m src.evaluation.heldout --data data/processed/v0.6-20k --split test --final \\
        --systems pipeline pipeline_text_lane rule_based structured_only_seed42 ... \\
        --reference pipeline --truth is_fraud
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

from src.evaluation.cost import CAPITATION_NOTE, DEFAULT_MINUTES, cost_summary
from src.evaluation.predictions import PREDICTIONS_DIR, read_predictions
from src.evaluation.strata import stratify

METRICS = ["precision", "recall", "f1", "fpr", "auc_roc"]
SEED_NAME = re.compile(r"^(?P<base>.+)_seed(?P<seed>\d+)$")
CLAIM_COLUMNS = ["claim_id", "is_fraud", "label", "fraud_typology", "icd11_code",
                 "has_text_noise", "has_structured_noise", "claim_amount"]


def _rates(tp, fp, fn, tn):
    """Vectorised precision, recall, F1, FPR; 0 where a denominator is 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
        recall = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
        f1 = np.where(precision + recall > 0,
                      2 * precision * recall / np.where(precision + recall > 0, precision + recall, 1), 0.0)
        fpr = np.where(fp + tn > 0, fp / np.maximum(fp + tn, 1), 0.0)
    return precision, recall, f1, fpr


def _auc(y, score) -> float | None:
    if score is None or len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, score))


def metrics(y, flagged, score=None) -> dict:
    """Precision, recall, F1, FPR; AUC-ROC only when a score is given and both classes
    are present (otherwise None)."""
    y = np.asarray(y).astype(int)
    f = np.asarray(flagged).astype(int)
    tp, fp = int(((y == 1) & (f == 1)).sum()), int(((y == 0) & (f == 1)).sum())
    fn, tn = int(((y == 1) & (f == 0)).sum()), int(((y == 0) & (f == 0)).sum())
    precision, recall, f1, fpr = (float(v) for v in _rates(tp, fp, fn, tn))
    auc = _auc(y, None if score is None else np.asarray(score, dtype=float))
    return {"claims": len(y), "positives": tp + fn, "flagged": tp + fp,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
            "fpr": round(fpr, 4), "auc_roc": None if auc is None else round(auc, 4)}


def bootstrap_ci(y, systems: dict, n: int = 1000, seed: int = 0) -> dict:
    """95% percentile intervals per system and metric. `systems` maps name -> (flagged,
    score or None). Every system is scored on the SAME resampled index sets, so the
    intervals are comparable. AUC is skipped on resamples with one class only."""
    y = np.asarray(y).astype(int)
    idx = np.random.default_rng(seed).integers(0, len(y), size=(n, len(y)))
    ys = y[idx]
    both_classes = (ys.min(axis=1) == 0) & (ys.max(axis=1) == 1)
    out = {}
    for name, (flagged, score) in systems.items():
        fs = np.asarray(flagged).astype(int)[idx]
        tp = ((ys == 1) & (fs == 1)).sum(axis=1)
        fp = ((ys == 0) & (fs == 1)).sum(axis=1)
        fn = ((ys == 1) & (fs == 0)).sum(axis=1)
        tn = ((ys == 0) & (fs == 0)).sum(axis=1)
        samples = dict(zip(["precision", "recall", "f1", "fpr"], _rates(tp, fp, fn, tn)))
        if score is not None:
            s = np.asarray(score, dtype=float)[idx]
            samples["auc_roc"] = np.array([roc_auc_score(ys[i], s[i]) if both_classes[i] else np.nan
                                           for i in range(n)])
        out[name] = {}
        for metric, values in samples.items():
            if np.all(np.isnan(values)):
                continue
            lo, hi = np.nanpercentile(values, [2.5, 97.5])
            out[name][metric] = [round(float(lo), 4), round(float(hi), 4)]
    return out


def mcnemar(y, pred_a, pred_b) -> dict:
    """Exact two-sided McNemar test: b = claims A gets right and B wrong, c = the
    reverse; under H0 b ~ Binomial(b + c, 0.5)."""
    y = np.asarray(y).astype(int)
    a_right = np.asarray(pred_a).astype(int) == y
    b_right = np.asarray(pred_b).astype(int) == y
    b = int((a_right & ~b_right).sum())
    c = int((~a_right & b_right).sum())
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return {"b": b, "c": c, "p_value": float(p)}


def seed_summary(per_seed_metrics: list[dict]) -> dict:
    """Mean and sample standard deviation (ddof=1; 0 for one seed) of each numeric metric."""
    out = {"seeds": len(per_seed_metrics)}
    for metric in METRICS:
        values = [m[metric] for m in per_seed_metrics if m.get(metric) is not None]
        if values:
            out[metric] = {"mean": round(float(np.mean(values)), 4),
                           "std": round(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0, 4)}
    return out


def group_seeds(systems: list[str], explicit: list[str] | None = None) -> dict[str, list[str]]:
    """`explicit` entries are `base=sys1,sys2`; otherwise names matching `<base>_seed<N>`
    are grouped by base. Groups of one are kept (std 0)."""
    if explicit:
        return {base: names.split(",") for base, names in (e.split("=", 1) for e in explicit)}
    groups: dict[str, list[str]] = {}
    for name in systems:
        match = SEED_NAME.match(name)
        if match:
            groups.setdefault(match["base"], []).append(name)
    return groups


def guard_split(parser: argparse.ArgumentParser, split: str, final: bool) -> None:
    if split == "test" and not final:
        parser.error("the test split is reserved for the final evaluation; pass --final to read it")


def load_split(data: Path, split: str) -> pd.DataFrame:
    claims = pd.read_parquet(data / f"{split}.parquet")
    claims["claim_id"] = claims["claim_id"].astype(str)
    return claims


def restrict_to(claims: pd.DataFrame, system: str, split: str,
                root: Path = PREDICTIONS_DIR) -> pd.DataFrame:
    """The split's claims that `system` has predictions for, in split order."""
    covered = set(read_predictions(system, split, root)["claim_id"])
    return claims[claims["claim_id"].isin(covered)].reset_index(drop=True)


def join_predictions(claims: pd.DataFrame, system: str, split: str,
                     root: Path = PREDICTIONS_DIR) -> pd.DataFrame:
    """The split's claims, in split order, with the system's flagged and score. Every
    claim needs a prediction, so all systems are compared on the same claims."""
    predictions = read_predictions(system, split, root).set_index("claim_id")
    missing = ~claims["claim_id"].isin(predictions.index)
    if missing.any():
        raise ValueError(f"{system}/{split}: no prediction for {int(missing.sum())} of {len(claims)} claims")
    frame = claims.reindex(columns=[c for c in CLAIM_COLUMNS if c in claims.columns]).copy()
    frame["flagged"] = predictions.loc[frame["claim_id"], "flagged"].to_numpy()
    frame["score"] = pd.to_numeric(predictions.loc[frame["claim_id"], "score"], errors="coerce").to_numpy()
    return frame


def truth(frame: pd.DataFrame, kind: str) -> np.ndarray:
    if kind == "text":
        return (frame["label"] == "inconsistent").astype(int).to_numpy()
    return frame["is_fraud"].astype(int).to_numpy()


def _score_or_none(frame: pd.DataFrame):
    """A score is used for AUC only when every claim has one."""
    return None if frame["score"].isna().any() else frame["score"].to_numpy()


def evaluate(claims: pd.DataFrame, train_codes: list[str], systems: list[str], split: str,
             reference: str | None, kind: str = "is_fraud", root: Path = PREDICTIONS_DIR,
             n_boot: int = 1000, seed: int = 0, minutes=DEFAULT_MINUTES,
             seed_groups: list[str] | None = None) -> dict:
    frames = {s: join_predictions(claims, s, split, root) for s in systems}
    y = truth(claims, kind)
    result = {"split": split, "truth": kind, "claims": len(claims), "positives": int(y.sum()),
              "reference": reference, "bootstrap": {"resamples": n_boot, "seed": seed},
              "cost_note": CAPITATION_NOTE, "minutes_per_review": list(minutes), "systems": {}}
    cis = bootstrap_ci(y, {s: (f["flagged"].to_numpy(), _score_or_none(f)) for s, f in frames.items()},
                       n=n_boot, seed=seed)
    for system, frame in frames.items():
        strata_frame = frame.assign(actual=y, predicted=frame["flagged"].astype(int))
        result["systems"][system] = {
            "metrics": metrics(y, frame["flagged"], _score_or_none(frame)),
            "ci95": cis[system],
            # Cost always uses is_fraud: it measures money lost to fraud.
            "cost": cost_summary(frame["claim_amount"], frame["is_fraud"], frame["flagged"], minutes),
            "strata": stratify(strata_frame, train_codes),
        }
    if reference:
        if reference not in frames:
            raise ValueError(f"reference system {reference!r} is not among --systems")
        result["mcnemar"] = {s: mcnemar(y, frames[reference]["flagged"], f["flagged"])
                             for s, f in frames.items() if s != reference}
    groups = group_seeds(systems, seed_groups)
    result["seed_groups"] = {base: {"systems": names, **seed_summary(
        [result["systems"][n]["metrics"] for n in names])} for base, names in groups.items()}
    return result


def _fmt(value) -> str:
    return "–" if value is None else f"{value:.3f}"


def _with_ci(metrics_: dict, ci: dict, metric: str) -> str:
    if metrics_.get(metric) is None:
        return "–"
    if metric not in ci:
        return _fmt(metrics_[metric])
    lo, hi = ci[metric]
    return f"{metrics_[metric]:.3f} [{lo:.3f}, {hi:.3f}]"


def _table(header: list[str], rows: list[list]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join([" --- "] * len(header)) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return lines + [""]


def tables_markdown(result: dict) -> str:
    systems = result["systems"]
    truth_name = "is_fraud" if result["truth"] == "is_fraud" else 'text label ("inconsistent")'
    out = [f"# Held-out evaluation: {result['split']} split", "",
           f"{result['claims']} claims, {result['positives']} positive (truth: {truth_name}). "
           f"95% percentile bootstrap intervals, {result['bootstrap']['resamples']} resamples "
           "shared by all systems.", ""]
    if result.get("claims_from"):
        out += [f"Only the claims {result['claims_from']} was run on; every system is scored on "
                "those claims.", ""]
    out += ["## Main metrics", ""]
    out += _table(["System", "Precision", "Recall", "F1", "FPR", "AUC-ROC"],
                  [[name] + [_with_ci(s["metrics"], s["ci95"], m) for m in METRICS]
                   for name, s in systems.items()])
    if result.get("mcnemar"):
        out += [f"## McNemar's exact test against {result['reference']}", "",
                f"b = claims {result['reference']} classifies correctly and the other system "
                "does not; c = the reverse.", ""]
        out += _table(["System", "b", "c", "p-value"],
                      [[name, m["b"], m["c"], f"{m['p_value']:.4g}"] for name, m in result["mcnemar"].items()])
    if result.get("seed_groups"):
        out += ["## Mean ± std over training seeds", ""]
        out += _table(["System", "Seeds"] + ["Precision", "Recall", "F1", "FPR", "AUC-ROC"],
                      [[base, g["seeds"]] + [f"{g[m]['mean']:.3f} ± {g[m]['std']:.3f}" if m in g else "–"
                                             for m in METRICS] for base, g in result["seed_groups"].items()])
    minutes = [f"{m:g}" for m in result["minutes_per_review"]]
    out += ["## Cost", "", result["cost_note"] + " Auditor-hours assume the stated minutes per "
            "review. Cost uses is_fraud as truth.", ""]
    out += _table(["System", "Flagged", "Amount-weighted recall", "Fraud KES missed"]
                  + [f"Auditor-hours @ {m} min" for m in minutes],
                  [[name, s["cost"]["flagged_claims"], _fmt(s["cost"]["amount_weighted_recall"]),
                    f"{s['cost']['fraud_kes_missed']:,.0f}"] + [s["cost"]["auditor_hours"][m] for m in minutes]
                   for name, s in systems.items()])
    typologies = sorted({t for s in systems.values() for t in s["strata"]["flagged_by_typology"]})
    out += ["## Flag rate by fraud typology", "",
            "Recall for typologies that are positive under this truth; the share wrongly "
            "flagged for the others.", ""]
    out += _table(["System"] + typologies,
                  [[name] + [_fmt(s["strata"]["flagged_by_typology"].get(t, {}).get("rate")) for t in typologies]
                   for name, s in systems.items()])
    for key, title, labels in [("text_noise", "Clean versus noisy text", {"False": "clean", "True": "noisy"}),
                               ("structured_noise", "Clean versus noisy structured fields",
                                {"False": "clean", "True": "noisy"}),
                               ("code_frequency", "Common versus rare ICD-11 codes",
                                {"common": "common", "rare": "rare"})]:
        out += [f"## {title} (F1 / recall / FPR)", ""]
        if key == "code_frequency":
            first = next(iter(systems.values()))
            out += [f"Rare = {first['strata']['rare_code_threshold']}.", ""]
        out += _table(["System"] + list(labels.values()),
                      [[name] + [(lambda g: f"{g['f1']:.3f} / {g['recall']:.3f} / {g['fpr']:.3f}"
                                  if g else "–")(s["strata"][key].get(k)) for k in labels]
                       for name, s in systems.items()])
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True, help="processed dataset directory")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--final", action="store_true", help="required to read the test split")
    parser.add_argument("--systems", nargs="+", required=True, help="names under --predictions-root")
    parser.add_argument("--reference", default="pipeline", help="system compared with every other (McNemar)")
    parser.add_argument("--truth", choices=["is_fraud", "text"], default="is_fraud")
    parser.add_argument("--predictions-root", type=Path, default=PREDICTIONS_DIR)
    parser.add_argument("--claims-from", help="score every system on this system's claims only")
    parser.add_argument("--seed-groups", nargs="*", help="base=sys1,sys2 ...; default: auto-group <base>_seed<N>")
    parser.add_argument("--minutes", type=float, nargs="+", default=list(DEFAULT_MINUTES))
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("experiments/evaluation"))
    args = parser.parse_args()
    guard_split(parser, args.split, args.final)

    claims = load_split(args.data, args.split)
    if args.claims_from:
        claims = restrict_to(claims, args.claims_from, args.split, args.predictions_root)
    train_codes = pd.read_parquet(args.data / "train.parquet", columns=["icd11_code"])["icd11_code"].tolist()
    reference = args.reference if args.reference in args.systems else None
    if args.reference and reference is None:
        print(f"note: reference {args.reference!r} not in --systems; McNemar skipped")
    result = evaluate(claims, train_codes, args.systems, args.split, reference, args.truth,
                      args.predictions_root, args.resamples, args.seed, args.minutes, args.seed_groups)
    result["data"] = str(args.data)
    result["claims_from"] = args.claims_from

    run_id = f"{args.data.name}_{args.split}_{args.truth}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    out = args.out / run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(result, indent=2))
    (out / "tables.md").write_text(tables_markdown(result))
    print(tables_markdown(result))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
