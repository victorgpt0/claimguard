from unittest.mock import MagicMock, patch

from src.feedback.feedback_repository import FeedbackRepository
from src.feedback.retrainer import ClassifierRetrainer


def _feedback_repo(tmp_path, n_examples=0):
    repo = FeedbackRepository(str(tmp_path / "feedback" / "examples.jsonl"))
    for i in range(n_examples):
        repo.materialize_example(
            claim={"id": i, "clinical_notes": f"notes {i}", "icd11_code": "1A00"},
            verdict={"verdict": "inconsistent"},
            auditor_decision="approved",
            icd11_definition="Cholera",
        )
    return repo


def test_maybe_trigger_retraining_false_below_threshold(tmp_path):
    repo = _feedback_repo(tmp_path, n_examples=3)
    retrainer = ClassifierRetrainer(repo, str(tmp_path / "active"), str(tmp_path / "candidates"))
    assert retrainer.maybe_trigger_retraining(trigger_count=50) is False


def test_maybe_trigger_retraining_true_at_threshold(tmp_path):
    repo = _feedback_repo(tmp_path, n_examples=50)
    retrainer = ClassifierRetrainer(repo, str(tmp_path / "active"), str(tmp_path / "candidates"))
    assert retrainer.maybe_trigger_retraining(trigger_count=50) is True


def test_evaluate_and_promote_promotes_when_no_active_model_exists(tmp_path):
    repo = _feedback_repo(tmp_path)
    active_dir = tmp_path / "active"
    candidate_dir = tmp_path / "candidate_checkpoint"
    candidate_dir.mkdir()
    (candidate_dir / "adapter_model.bin").write_text("fake weights")

    retrainer = ClassifierRetrainer(repo, str(active_dir), str(tmp_path / "candidates"))

    with patch("src.classification.classifier.SemanticConsistencyClassifier"), \
         patch.object(ClassifierRetrainer, "_evaluate_on_test_set",
                       return_value={"f1": 0.8, "recall": 0.85}):
        promoted = retrainer.evaluate_and_promote(str(candidate_dir), held_out_test_examples=[])

    assert promoted is True
    assert active_dir.exists()
    assert (active_dir / "adapter_model.bin").exists()


def test_evaluate_and_promote_rejects_worse_candidate(tmp_path):
    repo = _feedback_repo(tmp_path)
    active_dir = tmp_path / "active"
    active_dir.mkdir()
    (active_dir / "adapter_model.bin").write_text("active weights")

    candidate_dir = tmp_path / "candidate_checkpoint"
    candidate_dir.mkdir()
    (candidate_dir / "adapter_model.bin").write_text("candidate weights")

    candidates_holding_dir = tmp_path / "candidates"
    retrainer = ClassifierRetrainer(repo, str(active_dir), str(candidates_holding_dir))

    metrics_by_call = iter([{"f1": 0.5, "recall": 0.6}, {"f1": 0.9, "recall": 0.9}])
    with patch("src.classification.classifier.SemanticConsistencyClassifier"), \
         patch.object(ClassifierRetrainer, "_evaluate_on_test_set",
                       side_effect=lambda *a, **k: next(metrics_by_call)):
        promoted = retrainer.evaluate_and_promote(str(candidate_dir), held_out_test_examples=[])

    assert promoted is False
    assert (active_dir / "adapter_model.bin").read_text() == "active weights"  # unchanged
    assert (candidates_holding_dir / "candidate_checkpoint" / "adapter_model.bin").exists()
