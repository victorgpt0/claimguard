"""Structured-only baseline (#30): the text it trains on carries no notes and no label
leakage, its label is is_fraud, and its predictions use the shared format."""
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from src.baselines.structured_only import (export, predictions_frame, seed_of, structured_text)
from src.evaluation.predictions import read_predictions, write_predictions

CLAIM = dict(claim_id="CLM1", icd11_code="JB20.0", claim_amount=0.0, claim_type="outpatient",
             service_date="2025-06-15", discharge_date=None, facility_level=3,
             facility_type="dispensary", beneficiary_type="dependent", patient_age=27.0,
             patient_sex="female", clinical_notes="SECRET NOTE TEXT", is_fraud=1,
             fraud_typology="ghost_billing", label="consistent", has_text_noise=False,
             has_structured_noise=False, is_ghost_billed=True, original_icd11_code="JB20.0",
             member_id="MBR9", provider_id="PRV1")


def test_structured_text_uses_only_submitted_fields():
    text = structured_text(SimpleNamespace(**CLAIM))
    assert text == "code JB20.0 | KES 0 | outpatient | stay none | level 3 dispensary | dependent | age 27 | female"
    inpatient = SimpleNamespace(**{**CLAIM, "claim_type": "inpatient", "claim_amount": 67200.0,
                                   "discharge_date": "2025-06-19", "patient_age": float("nan")})
    assert "KES 67,200 | inpatient | stay 4 days" in structured_text(inpatient)
    assert "age unknown" in structured_text(inpatient)


def test_export_labels_by_is_fraud_and_drops_notes(tmp_path):
    data = tmp_path / "processed"
    data.mkdir()
    clean = {**CLAIM, "claim_id": "CLM2", "is_fraud": 0, "fraud_typology": "none", "label": "inconsistent"}
    for split in ("train", "validation", "test", "deployment_pool"):
        pd.DataFrame([CLAIM, clean]).to_parquet(data / f"{split}.parquet")
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({"JB20.0": {"code": "JB20.0", "title": "Single spontaneous delivery",
                                            "definition": "", "inclusion": [], "exclusion": []}}))
    counts = export(data, cache, tmp_path / "out")
    assert counts == {"train": 2, "validation": 2, "test": 2, "deployment_pool": 2}
    rows = [json.loads(line) for line in open(tmp_path / "out" / "validation.jsonl")]
    # The label follows is_fraud, not the text label (here deliberately the opposite).
    assert [r["label"] for r in rows] == ["inconsistent", "consistent"]
    assert all("SECRET" not in json.dumps(r) for r in rows)
    assert rows[0]["icd11_definition"] == "Single spontaneous delivery"
    assert rows[0]["fraud_typology"] == "ghost_billing"


def test_predictions_in_the_shared_format(tmp_path):
    frame = predictions_frame([{"claim_id": "A"}, {"claim_id": "B"}], [0.9, 0.2])
    write_predictions("structured_only_seed7", "validation", frame, tmp_path)
    back = read_predictions("structured_only_seed7", "validation", tmp_path)
    assert list(back["flagged"]) == [1, 0]
    assert list(back["score"]) == [0.9, 0.2]


def test_seed_from_run_folder(tmp_path):
    assert seed_of(tmp_path / "20261018T161108_seed42") == 42
    with pytest.raises(ValueError):
        seed_of(tmp_path / "active")
