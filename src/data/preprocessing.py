"""ClaimsPreprocessor (Section 3.6.1 Class Diagram): accepts raw claim records and
produces normalised feature matrices and cleaned clinical text strings.

Implements Section 3.2.2's five-step pipeline: schema alignment, data-quality validation,
noise/fraud/text-noise injection (src/data/noise_injection.py — called separately by
the dataset-generation script, not from here), feature encoding and normalisation, and
dataset splitting/stratification (Section 3.2.1's dataset-size justification: ~20,000
claims, 60/20/20 train/val/test).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

REQUIRED_COLUMNS = [
    "claim_id", "provider_id", "icd11_code", "claim_amount",
    "service_date", "discharge_date", "clinical_notes",
]

# Structured inputs to the anomaly detectors. Clinical notes are deliberately absent:
# text is the classifier's job, which is what keeps the two halves of the hybrid separate.
FEATURE_COLUMNS = [
    "log_claim_amount", "stay_days", "claim_amount_per_day", "is_inpatient",
    "facility_level", "patient_age", "is_female", "is_dependent",
    "code_frequency", "amount_vs_code_median", "unusual_admission",
    "member_claim_count", "member_dependent_claims", "provider_caesarean_share",
    "has_missing_field",
]
# Fields that may arrive blank and are imputed rather than rejected (Section 3.2.2).
OPTIONAL_FIELDS = ["patient_age", "patient_sex", "discharge_date"]
# ICD-11 stem code after repair: four characters, then an optional dot extension.
ICD11_CODE_PATTERN = r"[0-9A-Z][A-Z][0-9][0-9A-Z](\.[0-9A-Z]{1,4})?"
HISTORY_COLUMNS = ["claim_id", "member_id", "provider_id", "beneficiary_type", "icd11_code"]
# ICD-11: single spontaneous vertex delivery, elective and emergency caesarean section.
CAESAREAN_CODES = {"JB22.0", "JB22.1"}
DELIVERY_CODES = {"JB20.0"} | CAESAREAN_CODES


def repair_icd11_code(code):
    """Undo formatting faults with one correct reading: stray spaces, lower case, and a
    dropped dot (ICD-11 stem codes always have four characters before the dot). Anything
    else is left for validate_quality to flag."""
    if not isinstance(code, str):
        return code
    code = code.strip().upper()
    if "." not in code and len(code) > 4:
        code = f"{code[:4]}.{code[4:]}"
    return code


class ClaimsPreprocessor:
    def __init__(self):
        self._scaler: StandardScaler | None = None
        self._code_stats: pd.DataFrame | None = None
        self._history: pd.DataFrame | None = None
        self._median_age: float = 0.0

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
        out["icd11_code"] = out["icd11_code"].map(repair_icd11_code)
        return out

    def validate_quality(self, claims: pd.DataFrame) -> pd.DataFrame:
        """Step 2: logical-consistency checks (service date <= discharge date <= claim
        date, plausible claim-amount ranges). Rows failing these checks are flagged, not
        silently dropped — Section 3.2.2 says invalid rows are "flagged for review"."""
        out = claims.copy()
        out["quality_flag_date_order"] = out["service_date"] > out["discharge_date"]
        out["quality_flag_negative_amount"] = out["claim_amount"] < 0
        out["quality_flag_missing_notes"] = out["clinical_notes"].str.strip() == ""
        optional = [f for f in OPTIONAL_FIELDS if f in out.columns]
        out["quality_flag_missing_field"] = out[optional].isna().any(axis=1)
        out["quality_flag_invalid_code"] = ~out["icd11_code"].astype(str).str.fullmatch(ICD11_CODE_PATTERN)
        return out

    def encode_features(self, claims: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Step 4: numeric encoding, derived features and normalisation. Adds the
        FEATURE_COLUMNS, scaled, and leaves the original columns in place.

        Three kinds of feature:
        - per claim: amount, stay, amount per day, facility level, age, sex, dependent;
        - peer comparison against the training split's claims for the same ICD-11 code
          (code frequency, amount relative to the code's median, how unusual an admission
          is for that code);
        - history aggregates over the training claims plus the batch being scored: claims
          per member, dependent claims per member, the provider's caesarean share. A
          post-payment audit sees the claims history, and these patterns (Table 3.1 rows
          4 and 6) only exist across claims. They use no labels.

        Blank optional fields are imputed from the training split (age: median; stay: the
        code's median stay; sex: halfway between the two encodings) and counted in
        `has_missing_field`, so a defective claim is still scored.

        `fit=True` only on the training split — preprocessing must be fit on training
        data alone to avoid leakage into validation/test (research_writing.txt 5.3.2)."""
        out = claims.copy()
        if fit:
            self._history = out[HISTORY_COLUMNS].copy()
            by_code = out.groupby("icd11_code")
            known_stay = (out["discharge_date"] - out["service_date"]).dt.days.clip(lower=0)
            self._code_stats = pd.DataFrame({
                "frequency": by_code.size() / len(out),
                "median_amount": by_code["claim_amount"].median(),
                # Claims with no discharge date are left out of both, not counted as 0.
                "inpatient_rate": (known_stay > 0).where(known_stay.notna()).groupby(out["icd11_code"]).mean(),
                "median_stay": known_stay.groupby(out["icd11_code"]).median(),
            })
            self._median_age = float(out["patient_age"].median())
        elif self._code_stats is None:
            raise RuntimeError("encode_features(fit=True) must run on the training split first")

        stats = self._code_stats.reindex(out["icd11_code"])
        optional = [f for f in OPTIONAL_FIELDS if f in out.columns]
        out["has_missing_field"] = out[optional].isna().sum(axis=1).astype(float)
        stay_days = (out["discharge_date"] - out["service_date"]).dt.days.clip(lower=0)
        stay_days = stay_days.fillna(pd.Series(stats["median_stay"].to_numpy(), index=out.index)).fillna(0)
        out["stay_days"] = stay_days
        out["patient_age"] = out["patient_age"].astype(float).fillna(self._median_age)
        out["claim_amount_per_day"] = out["claim_amount"] / stay_days.replace(0, 1)
        out["log_claim_amount"] = np.log1p(out["claim_amount"])
        out["is_inpatient"] = (stay_days > 0).astype(float)
        out["is_dependent"] = (out["beneficiary_type"] == "dependent").astype(float)
        out["is_female"] = out["patient_sex"].map({"female": 1.0, "male": 0.0}).fillna(0.5)

        # Codes never seen in training count as rare, with no admission history.
        out["code_frequency"] = stats["frequency"].fillna(0.0).to_numpy()
        median = stats["median_amount"].fillna(0.0).to_numpy()
        out["amount_vs_code_median"] = np.log1p(out["claim_amount"]) - np.log1p(median)
        out["unusual_admission"] = out["is_inpatient"] * (1 - stats["inpatient_rate"].fillna(0.0).to_numpy())

        history = pd.concat([self._history, out[HISTORY_COLUMNS]]).drop_duplicates("claim_id")
        member_claims = history.groupby("member_id").size()
        member_dependents = (history["beneficiary_type"] == "dependent").groupby(history["member_id"]).sum()
        deliveries = history[history["icd11_code"].isin(DELIVERY_CODES)]
        caesarean_share = deliveries["icd11_code"].isin(CAESAREAN_CODES).groupby(deliveries["provider_id"]).mean()
        out["member_claim_count"] = out["member_id"].map(member_claims).astype(float)
        out["member_dependent_claims"] = out["member_id"].map(member_dependents).astype(float)
        # Only the caesarean claims carry the provider's share: the excess caesareans are
        # what is billed fraudulently, not the provider's normal deliveries.
        out["provider_caesarean_share"] = (out["icd11_code"].isin(CAESAREAN_CODES)
                                           * out["provider_id"].map(caesarean_share).fillna(0.0))

        if fit:
            self._scaler = StandardScaler()
            out[FEATURE_COLUMNS] = self._scaler.fit_transform(out[FEATURE_COLUMNS])
        else:
            out[FEATURE_COLUMNS] = self._scaler.transform(out[FEATURE_COLUMNS])
        return out

    def feature_matrix(self, claims: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """The detectors' input: encode_features restricted to FEATURE_COLUMNS."""
        return self.encode_features(claims, fit=fit)[FEATURE_COLUMNS].to_numpy(dtype=float)

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
