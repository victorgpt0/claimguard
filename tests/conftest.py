import pandas as pd
import pytest


@pytest.fixture
def sample_claims_df() -> pd.DataFrame:
    return pd.DataFrame({
        "claim_id": [1, 2, 3, 4],
        "provider_id": ["P1", "P1", "P2", "P2"],
        "patient_ref": ["PT1", "PT2", "PT3", "PT4"],
        "icd11_code": ["1A00", "1A01", "1A00", "2B02"],
        "claim_amount": [1000.0, 2000.0, 1500.0, 5000.0],
        "service_date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
        "discharge_date": ["2026-01-01", "2026-01-02", "2026-01-04", "2026-01-10"],
        "clinical_notes": [
            "Patient presented with fever and pain.",
            "Doctor prescribed medicine for infection.",
            "Routine consultation, no complications.",
            "Emergency admission following injury.",
        ],
        "claim_type": ["outpatient", "outpatient", "outpatient", "inpatient"],
        "is_fraud": [0, 0, 1, 1],
    })


@pytest.fixture
def tmp_db_path(tmp_path):
    return str(tmp_path / "test.sqlite")
