import pytest

from src.data.ingestion import load_claims_batch


def test_load_claims_batch_csv(tmp_path, sample_claims_df):
    path = tmp_path / "claims.csv"
    sample_claims_df.to_csv(path, index=False)
    loaded = load_claims_batch(path)
    assert len(loaded) == len(sample_claims_df)


def test_load_claims_batch_parquet(tmp_path, sample_claims_df):
    path = tmp_path / "claims.parquet"
    sample_claims_df.to_parquet(path, index=False)
    loaded = load_claims_batch(path)
    assert len(loaded) == len(sample_claims_df)


def test_load_claims_batch_rejects_unsupported_extension(tmp_path):
    path = tmp_path / "claims.txt"
    path.write_text("not a real claims file")
    with pytest.raises(ValueError):
        load_claims_batch(path)


def test_load_claims_batch_rejects_missing_columns(tmp_path, sample_claims_df):
    path = tmp_path / "claims.csv"
    sample_claims_df.drop(columns=["clinical_notes"]).to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_claims_batch(path)
