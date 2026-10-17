"""Stratified results for any system (Section 3.2.4, task 7.2).

Generalises src.classification.evaluate.stratify from the classifier to every system in
the shared prediction format: per fraud typology, clean versus noisy text and
structured fields, and common versus rare ICD-11 codes (rare = under 1% of training
claims, or absent from training).
"""
from __future__ import annotations

from collections import Counter

import pandas as pd

# Same threshold as src/classification/evaluate.py.
RARE_CODE_SHARE = 0.01


def group_metrics(actual: pd.Series, predicted: pd.Series) -> dict:
    """Counts plus precision, recall, F1 and false-positive rate for one stratum."""
    a, p = actual.astype(int), predicted.astype(int)
    tp = int(((a == 1) & (p == 1)).sum())
    fp = int(((a == 0) & (p == 1)).sum())
    fn = int(((a == 1) & (p == 0)).sum())
    tn = int(((a == 0) & (p == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"claims": len(a), "positives": tp + fn, "flagged": tp + fp,
            "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
            "fpr": round(fp / (fp + tn), 4) if fp + tn else 0.0}


def rare_codes(train_codes: list[str]) -> set[str]:
    counts = Counter(train_codes)
    return {code for code, n in counts.items() if n / len(train_codes) < RARE_CODE_SHARE}


def stratify(frame: pd.DataFrame, train_codes: list[str]) -> dict:
    """`frame` has one row per claim: actual, predicted (0/1), fraud_typology,
    has_text_noise, has_structured_noise, icd11_code — a system's predictions joined
    to the split's claims."""
    counts = set(train_codes)
    rare = rare_codes(train_codes)
    code_group = frame["icd11_code"].map(lambda c: "rare" if c in rare or c not in counts else "common")
    by_typology = {
        typology: {"claims": len(group), "flagged": int(group["predicted"].sum()),
                   "rate": round(float(group["predicted"].mean()), 4)}
        for typology, group in frame.groupby("fraud_typology")
    }
    return {
        "overall": group_metrics(frame["actual"], frame["predicted"]),
        # Rate is recall for typologies that are positive under the chosen truth, and the
        # share wrongly flagged for the rest ("none", and under text truth the typologies
        # whose notes stay consistent with the billed code).
        "flagged_by_typology": by_typology,
        "text_noise": {str(k): group_metrics(g["actual"], g["predicted"])
                       for k, g in frame.groupby("has_text_noise")},
        "structured_noise": {str(k): group_metrics(g["actual"], g["predicted"])
                             for k, g in frame.groupby("has_structured_noise")},
        "code_frequency": {k: group_metrics(g["actual"], g["predicted"])
                           for k, g in frame.groupby(code_group)},
        "rare_code_threshold": f"< {RARE_CODE_SHARE:.0%} of training claims ({len(rare)} codes)",
    }
