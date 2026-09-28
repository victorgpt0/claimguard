import pytest

from src.feedback.feedback_repository import FeedbackRepository


@pytest.fixture
def repo(tmp_path):
    return FeedbackRepository(str(tmp_path / "feedback" / "examples.jsonl"))


def test_materialize_example_approved(repo):
    repo.materialize_example(
        claim={"id": 1, "clinical_notes": "notes", "icd11_code": "1A00"},
        verdict={"verdict": "inconsistent"},
        auditor_decision="approved",
    )
    assert repo.total_example_count() == 1
    examples = repo.read_examples(since_consumed_marker=False)
    assert examples[0]["label"] == "confirmed_fraud"


def test_materialize_example_escalated_raises(repo):
    with pytest.raises(ValueError):
        repo.materialize_example(
            claim={"id": 1, "clinical_notes": "notes", "icd11_code": "1A00"},
            verdict={"verdict": "inconsistent"},
            auditor_decision="escalated",
        )


def test_count_unconsumed_examples_and_mark_consumed(repo):
    for _ in range(3):
        repo.materialize_example(
            claim={"id": 1, "clinical_notes": "notes", "icd11_code": "1A00"},
            verdict={"verdict": "inconsistent"},
            auditor_decision="approved",
        )
    assert repo.count_unconsumed_examples() == 3

    repo.mark_consumed()
    assert repo.count_unconsumed_examples() == 0

    repo.materialize_example(
        claim={"id": 2, "clinical_notes": "notes2", "icd11_code": "1A01"},
        verdict={"verdict": "consistent"},
        auditor_decision="rejected",
    )
    assert repo.count_unconsumed_examples() == 1
