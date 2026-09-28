from unittest.mock import MagicMock, patch

import torch

from src.classification.classifier import LABELS, SemanticConsistencyClassifier


def _fake_tokenizer_call(*args, **kwargs):
    fake_batch = MagicMock()
    fake_batch.to.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
    return fake_batch


def test_predict_returns_verdict_from_highest_probability_class(tmp_path):
    with patch("src.classification.classifier.AutoTokenizer") as MockTokenizer, \
         patch("src.classification.classifier.AutoModelForSequenceClassification") as MockModelCls:
        MockTokenizer.from_pretrained.return_value = MagicMock(side_effect=_fake_tokenizer_call)

        fake_model = MagicMock()
        fake_model.to.return_value = fake_model
        fake_model.eval.return_value = fake_model
        fake_logits = torch.tensor([[0.1, 5.0]])  # index 1 ("inconsistent") dominates
        fake_model.return_value = MagicMock(logits=fake_logits)
        MockModelCls.from_pretrained.return_value = fake_model

        classifier = SemanticConsistencyClassifier(model_path=str(tmp_path / "no-adapter-yet"))
        verdict = classifier.predict("claim-1", "clinical notes", "icd11 definition")

    assert verdict.verdict == LABELS[1]
    assert verdict.model_version == "untrained-base"
    assert 0.0 <= verdict.confidence <= 1.0


def test_load_metrics_returns_none_when_absent(tmp_path):
    assert SemanticConsistencyClassifier.load_metrics(str(tmp_path / "nonexistent")) is None


def test_load_metrics_reads_json(tmp_path):
    model_dir = tmp_path / "checkpoint"
    model_dir.mkdir()
    (model_dir / "metrics.json").write_text('{"f1": 0.9, "recall": 0.85}')
    metrics = SemanticConsistencyClassifier.load_metrics(str(model_dir))
    assert metrics == {"f1": 0.9, "recall": 0.85}
