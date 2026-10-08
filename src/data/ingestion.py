"""Claim batch ingestion (FR-01). Loads a batch of synthetic claims — structured billing
fields plus free-text clinical justification — for processing by the auditing pipeline.

Traces to: Objective iv; proposal_revision Section 2.5 (input variables).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.preprocessing import REQUIRED_COLUMNS


def load_claims_batch(path: str | Path) -> pd.DataFrame:
    """Load a batch of raw synthetic claims from the processed dataset (Section 3.2.2).
    Supports .csv and .parquet; picked by file extension."""
    path = Path(path)
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported claims file format: {path.suffix!r} (expected .csv/.parquet)")

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Claims file {path} missing required columns: {sorted(missing)}")
    return df
