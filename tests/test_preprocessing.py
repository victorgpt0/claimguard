import pandas as pd
import pytest

from src.data.preprocessing import FEATURE_COLUMNS, ClaimsPreprocessor


def test_align_schema_raises_on_missing_column(sample_claims_df):
    preprocessor = ClaimsPreprocessor()
    incomplete = sample_claims_df.drop(columns=["clinical_notes"])
    with pytest.raises(ValueError):
        preprocessor.align_schema(incomplete)


def test_align_schema_casts_types(sample_claims_df):
    preprocessor = ClaimsPreprocessor()
    aligned = preprocessor.align_schema(sample_claims_df)
    assert str(aligned["service_date"].dtype).startswith("datetime64")
    assert aligned["claim_amount"].dtype.kind == "f"


def test_validate_quality_flags_bad_date_order(sample_claims_df):
    preprocessor = ClaimsPreprocessor()
    aligned = preprocessor.align_schema(sample_claims_df)
    aligned.loc[0, "discharge_date"] = aligned.loc[0, "service_date"] - __import__("pandas").Timedelta(days=1)
    validated = preprocessor.validate_quality(aligned)
    assert validated.loc[0, "quality_flag_date_order"]
    assert not validated.loc[1, "quality_flag_date_order"]


def test_encode_features_adds_derived_columns(sample_claims_df):
    preprocessor = ClaimsPreprocessor()
    aligned = preprocessor.align_schema(sample_claims_df)
    encoded = preprocessor.encode_features(aligned, fit=True)
    assert "stay_days" in encoded.columns
    assert "claim_amount_per_day" in encoded.columns
    assert "code_frequency" in encoded.columns


def test_feature_matrix_has_fixed_columns_across_splits(sample_claims_df):
    preprocessor = ClaimsPreprocessor()
    aligned = preprocessor.align_schema(sample_claims_df)
    train = preprocessor.feature_matrix(aligned.iloc[:3], fit=True)
    # The held-out batch has a provider and a code never seen in training.
    held_out = aligned.iloc[3:].assign(provider_id="P9", icd11_code="9Z99")
    scored = preprocessor.feature_matrix(held_out)
    assert train.shape[1] == scored.shape[1] == len(FEATURE_COLUMNS)
    assert not pd.isna(scored).any()


def test_encode_features_requires_fit_on_training_first(sample_claims_df):
    preprocessor = ClaimsPreprocessor()
    aligned = preprocessor.align_schema(sample_claims_df)
    with pytest.raises(RuntimeError):
        preprocessor.encode_features(aligned)


def test_history_features_count_member_claims_and_caesarean_share(sample_claims_df):
    preprocessor = ClaimsPreprocessor()
    claims = preprocessor.align_schema(sample_claims_df)
    claims["icd11_code"] = ["JB22.1", "JB22.1", "JB20.0", "JB22.1"]
    claims["provider_id"] = ["P1", "P1", "P2", "P2"]
    preprocessor.encode_features(claims.iloc[:2], fit=True)
    scaled = preprocessor.encode_features(claims.iloc[2:])[FEATURE_COLUMNS]
    raw = preprocessor._scaler.inverse_transform(scaled)
    row = dict(zip(FEATURE_COLUMNS, raw[1]))  # claim 4: P2 caesarean, member M3
    assert row["member_claim_count"] == 1
    assert row["provider_caesarean_share"] == 0.5  # P2: one caesarean in two deliveries


def test_split_dataset_respects_proportions_and_stratification(sample_claims_df):
    preprocessor = ClaimsPreprocessor()
    # duplicate rows so stratified split has enough samples per class
    claims = __import__("pandas").concat([sample_claims_df] * 10, ignore_index=True)
    splits = preprocessor.split_dataset(claims, train=0.6, val=0.2, test=0.2, seed=1)
    total = sum(len(df) for df in splits.values())
    assert total == len(claims)
    assert set(splits.keys()) == {"train", "validation", "test"}


def test_split_dataset_rejects_bad_proportions(sample_claims_df):
    preprocessor = ClaimsPreprocessor()
    with pytest.raises(ValueError):
        preprocessor.split_dataset(sample_claims_df, train=0.5, val=0.3, test=0.3)


def test_defective_claims_are_flagged_imputed_and_still_scored(sample_claims_df):
    preprocessor = ClaimsPreprocessor()
    claims = sample_claims_df.copy()
    claims["icd11_code"] = ["1A00", " 1a01 ", "1A00", "BAD CODE"]
    claims["patient_age"] = claims["patient_age"].astype(float)
    claims.loc[1, "patient_age"] = None
    claims.loc[2, "discharge_date"] = None
    aligned = preprocessor.align_schema(claims)
    assert aligned.loc[1, "icd11_code"] == "1A01"
    flags = preprocessor.validate_quality(aligned)
    assert flags["quality_flag_missing_field"].tolist() == [False, True, True, False]
    assert flags["quality_flag_invalid_code"].tolist() == [False, False, False, True]
    matrix = preprocessor.feature_matrix(aligned, fit=True)
    assert not pd.isna(matrix).any()
