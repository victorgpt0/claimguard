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
