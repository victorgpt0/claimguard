import json

import pytest

from src.feedback.feedback_repository import FeedbackRepository, feedback_label

CLAIM = {"id": 1, "clinical_notes": "notes", "icd11_code": "1A00"}


@pytest.fixture
def repo(tmp_path):
    return FeedbackRepository(str(tmp_path / "feedback" / "examples.jsonl"))


@pytest.mark.parametrize("decision, verdict, label", [
    ("approved", "inconsistent", "inconsistent"),  # auditor confirms the mismatch
    ("rejected", "inconsistent", "consistent"),    # classifier false positive
    ("rejected", "consistent", "consistent"),      # stage-1 false positive
    ("approved", "consistent", None),              # fraud, but not a text mismatch
])
def test_feedback_label(decision, verdict, label):
    assert feedback_label(decision, verdict) == label


def test_example_matches_the_classifier_training_format(repo):
    written = repo.materialize_example(CLAIM, {"verdict": "inconsistent"}, "approved",
                                       icd11_definition="Cholera. An infection.")
    assert written is True
    example = repo.read_examples(since_consumed_marker=False)[0]
    # The keys train.py and ClassifierRetrainer read.
    assert example["label"] == "inconsistent"
    assert example["icd11_definition"] == "Cholera. An infection."
    assert example["clinical_notes"] == "notes"


def test_approval_of_a_consistent_verdict_writes_nothing(repo):
    written = repo.materialize_example(CLAIM, {"verdict": "consistent"}, "approved",
                                       icd11_definition="Cholera")
    assert written is False
    assert repo.total_example_count() == 0


def test_materialize_example_escalated_raises(repo):
    with pytest.raises(ValueError):
        repo.materialize_example(CLAIM, {"verdict": "inconsistent"}, "escalated",
                                 icd11_definition="Cholera")


def test_count_unconsumed_examples_and_mark_consumed(repo):
    for _ in range(3):
        repo.materialize_example(CLAIM, {"verdict": "inconsistent"}, "approved",
                                 icd11_definition="Cholera")
    assert repo.count_unconsumed_examples() == 3

    repo.mark_consumed()
    assert repo.count_unconsumed_examples() == 0

    repo.materialize_example({"id": 2, "clinical_notes": "notes2", "icd11_code": "1A01"},
                             {"verdict": "consistent"}, "rejected", icd11_definition="Typhoid")
    assert repo.count_unconsumed_examples() == 1
    assert json.loads(repo.feedback_dataset_path.read_text().splitlines()[-1])["label"] == "consistent"
