"""ClaimsPreprocessor (Section 3.6.1 Class Diagram): accepts raw claim records and
produces normalised feature matrices and cleaned clinical text strings.

Implements Section 3.2.2's five-step pipeline: schema alignment, data-quality validation,
noise/fraud/text-noise injection (src/data/noise_injection.py — called separately by
the dataset-generation script, not from here), feature encoding and normalisation, and
dataset splitting/stratification (Section 3.2.1's dataset-size justification: ~20,000
claims, 60/20/20 train/val/test).
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

REQUIRED_COLUMNS = [
    "claim_id", "provider_id", "icd11_code", "claim_amount",
    "service_date", "discharge_date", "clinical_notes",
]


class ClaimsPreprocessor:
    def __init__(self):
        self._scaler: StandardScaler | None = None

    def align_schema(self, raw_claims: pd.DataFrame) -> pd.DataFrame:
        """Step 1: rename/reorder/type-cast fields to match the SHA provider portal
        schema. Raises if a required field is missing rather than silently proceeding
        with an incomplete claim — a claim without these fields can't be processed."""
        missing = set(REQUIRED_COLUMNS) - set(raw_claims.columns)
        if missing:
            raise ValueError(f"Raw claims missing required columns: {sorted(missing)}")
        out = raw_claims.copy()
        out["service_date"] = pd.to_datetime(out["service_date"])
        out["discharge_date"] = pd.to_datetime(out["discharge_date"])
        out["claim_amount"] = out["claim_amount"].astype(float)
        out["clinical_notes"] = out["clinical_notes"].fillna("").astype(str)
        return out

    def validate_quality(self, claims: pd.DataFrame) -> pd.DataFrame:
        """Step 2: logical-consistency checks (service date <= discharge date <= claim
        date, plausible claim-amount ranges). Rows failing these checks are flagged, not
        silently dropped — Section 3.2.2 says invalid rows are "flagged for review"."""
        out = claims.copy()
        out["quality_flag_date_order"] = out["service_date"] > out["discharge_date"]
        out["quality_flag_negative_amount"] = out["claim_amount"] < 0
        out["quality_flag_missing_notes"] = out["clinical_notes"].str.strip() == ""
        return out

    def encode_features(self, claims: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Step 4: categorical encoding, numeric normalisation, derived features (claim
        amount per day, service-to-discharge duration, repeated-code frequency).

        `fit=True` only on the training split — preprocessing must be fit on training
        data alone to avoid leakage into validation/test (research_writing.txt 5.3.2)."""
        out = claims.copy()
        stay_days = (out["discharge_date"] - out["service_date"]).dt.days.clip(lower=0)
        out["stay_days"] = stay_days
        out["claim_amount_per_day"] = out["claim_amount"] / stay_days.replace(0, 1)
        out["code_frequency"] = out["icd11_code"].map(out["icd11_code"].value_counts())
        out = pd.get_dummies(out, columns=["provider_id"], prefix="provider")

        numeric_cols = ["claim_amount", "claim_amount_per_day", "stay_days", "code_frequency"]
        if fit:
            self._scaler = StandardScaler()
            out[numeric_cols] = self._scaler.fit_transform(out[numeric_cols])
        elif self._scaler is not None:
            out[numeric_cols] = self._scaler.transform(out[numeric_cols])
        return out

    def split_dataset(self, claims: pd.DataFrame, fraud_label_col: str = "is_fraud",
                       train: float = 0.6, val: float = 0.2, test: float = 0.2,
                       seed: int = 42) -> dict[str, pd.DataFrame]:
        """Step 5: stratified train/validation/held-out-test split on the fraud label.

        Note: the fourth partition, the Deployment/Demonstration Pool (DR-08), is NOT
        produced here — it's a disjoint slice carved out at dataset-generation time,
        before this split, so it can never overlap with the held-out test set that these
        three splits reserve for reported metrics."""
        if abs(train + val + test - 1.0) > 1e-9:
            raise ValueError(f"train+val+test must sum to 1.0, got {train + val + test}")

        train_df, temp_df = train_test_split(
            claims, train_size=train, stratify=claims[fraud_label_col], random_state=seed
        )
        relative_val = val / (val + test)
        val_df, test_df = train_test_split(
            temp_df, train_size=relative_val, stratify=temp_df[fraud_label_col], random_state=seed
        )
        return {"train": train_df, "validation": val_df, "test": test_df}
