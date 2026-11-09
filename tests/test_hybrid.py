"""Configuration C (#33): text lane OR any rule."""
import pandas as pd
import pytest

from src.evaluation.hybrid import combine


def test_reports_a_claim_flagged_by_either_part():
    ids = ["A", "B", "C", "D"]
    text = pd.DataFrame({"claim_id": ids, "flagged": [1, 0, 1, 0], "score": [0.9, 0.1, 0.8, 0.2]})
    # Listed in reverse: only B is flagged by a rule.
    rules = pd.DataFrame({"claim_id": list(reversed(ids)), "flagged": [0, 0, 1, 0], "score": [0, 0, 1, 0]})
    assert list(combine(text, rules)["flagged"]) == [1, 1, 1, 0]  # joined by claim id, not position


def test_refuses_predictions_for_different_claims():
    text = pd.DataFrame({"claim_id": ["A", "B"], "flagged": [1, 0]})
    with pytest.raises(ValueError):
        combine(text, pd.DataFrame({"claim_id": ["A"], "flagged": [1]}))
