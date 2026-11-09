import json

import pandas as pd
import pytest

from src.classification.export import export, reference_text
from src.classification.train import class_weights
from src.icd11.client import ICD11Definition


def _cache_file(tmp_path):
    path = tmp_path / "cache.json"
    path.write_text(json.dumps({
        "CA40.Z": {"code": "CA40.Z", "title": "Pneumonia, organism unspecified",
                   "definition": "A disease of the lungs.", "inclusion": [], "exclusion": [],
                   "definition_source": "CA40"},
        "BD1Z": {"code": "BD1Z", "title": "Heart failure, unspecified", "definition": "",
                 "inclusion": [], "exclusion": []},
    }))
    return path


def _split(codes, labels):
    return pd.DataFrame({
        "claim_id": [f"CLM{i}" for i in range(len(codes))],
        "icd11_code": codes,
        "clinical_notes": ["Pt reviewed."] * len(codes),
        "label": labels,
        "fraud_typology": ["none"] * len(codes),
        "has_text_noise": [False] * len(codes),
        "has_structured_noise": [True] * len(codes),
    })


def test_reference_text_falls_back_to_title():
    with_definition = ICD11Definition("CA40.Z", "Pneumonia, organism unspecified", "A disease of the lungs.", [], [])
    without = ICD11Definition("BD1Z", "Heart failure, unspecified", "", [], [])
    assert reference_text(with_definition) == "Pneumonia, organism unspecified. A disease of the lungs."
    assert reference_text(without) == "Heart failure, unspecified"


def test_export_repairs_codes_and_keeps_labels(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    for split in ("train", "validation", "test", "deployment_pool"):
        _split([" ca40z ", "BD1Z"], ["inconsistent", "consistent"]).to_parquet(data / f"{split}.parquet")
    counts = export(data, _cache_file(tmp_path), tmp_path / "out")
    assert counts == {"train": 2, "validation": 2, "test": 2, "deployment_pool": 2}
    rows = [json.loads(line) for line in open(tmp_path / "out" / "train.jsonl")]
    assert rows[0]["icd11_code"] == "CA40.Z"
    assert rows[0]["icd11_definition"].startswith("Pneumonia")
    assert [r["label"] for r in rows] == ["inconsistent", "consistent"]
    assert rows[0]["has_structured_noise"] is True


def test_export_fails_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        export(tmp_path, tmp_path / "missing.json", tmp_path / "out")


def test_class_weights_favour_the_rare_class():
    examples = [{"label": "consistent"}] * 95 + [{"label": "inconsistent"}] * 5
    consistent, inconsistent = class_weights(examples).tolist()
    assert inconsistent == pytest.approx(19 * consistent)
    assert (consistent + inconsistent) / 2 == pytest.approx(5.26, abs=0.01)


def test_stratify_splits_by_typology_noise_and_code_frequency():
    from src.classification.evaluate import stratify
    predictions = pd.DataFrame({
        "actual":    [1, 1, 0, 0, 0],
        "predicted": [1, 0, 1, 0, 0],
        "fraud_typology": ["upcoding", "ghost_billing", "none", "none", "inflated_caesarean"],
        "has_text_noise": [True, False, False, True, False],
        "has_structured_noise": [False] * 5,
        "icd11_code": ["JB22.1", "JB22.1", "CA40.Z", "RARE", "JB22.1"],
    })
    train_codes = ["JB22.1"] * 120 + ["CA40.Z"] * 79 + ["RARE"]  # RARE: 1 of 200 = 0.5%
    result = stratify(predictions, train_codes)
    assert result["overall"]["recall"] == 0.5
    assert result["flagged_by_typology"]["upcoding"]["rate"] == 1.0
    assert result["flagged_by_typology"]["ghost_billing"]["rate"] == 0.0
    assert result["text_noise"]["True"]["claims"] == 2
    assert result["code_frequency"]["common"]["claims"] == 4
    assert result["code_frequency"]["rare"]["claims"] == 1
