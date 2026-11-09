"""Shared prediction format for every system compared in Section 3.2.4: the full pipeline
and each baseline write one CSV per split, so the held-out evaluation treats them alike.

    experiments/predictions/<system>/<split>.csv
    claim_id, flagged (0/1), score (float, higher = more suspicious; blank if none)

Predictions carry no labels: evaluation joins them to the split's ground truth by
claim_id.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

PREDICTIONS_DIR = Path("experiments/predictions")
COLUMNS = ["claim_id", "flagged", "score"]


def prediction_path(system: str, split: str, root: Path = PREDICTIONS_DIR) -> Path:
    return root / system / f"{split}.csv"


def write_predictions(system: str, split: str, frame: pd.DataFrame,
                      root: Path = PREDICTIONS_DIR) -> Path:
    """`frame` needs claim_id and flagged; score is optional. Claim ids must be unique."""
    if frame["claim_id"].duplicated().any():
        raise ValueError(f"{system}/{split}: duplicate claim ids in predictions")
    out = frame.reindex(columns=COLUMNS).copy()
    out["flagged"] = out["flagged"].astype(int)
    path = prediction_path(system, split, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return path


def read_predictions(system: str, split: str, root: Path = PREDICTIONS_DIR) -> pd.DataFrame:
    frame = pd.read_csv(prediction_path(system, split, root), dtype={"claim_id": str})
    frame["flagged"] = frame["flagged"].astype(int)
    return frame
