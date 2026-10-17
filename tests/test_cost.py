import pytest

from src.evaluation.cost import CAPITATION_NOTE, cost_summary


def test_cost_summary_amount_weighted():
    amount = [1000, 3000, 0, 500, 200]
    is_fraud = [1, 1, 1, 0, 0]
    flagged = [1, 0, 1, 1, 0]
    s = cost_summary(amount, is_fraud, flagged, minutes=(20, 40))
    assert s["fraud_kes_total"] == 4000
    assert s["fraud_kes_flagged"] == 1000
    assert s["fraud_kes_missed"] == 3000
    assert s["amount_weighted_recall"] == 0.25
    assert s["flagged_claims"] == 3
    assert s["auditor_hours"] == {"20": 1.0, "40": 2.0}


def test_cost_summary_float_minutes_and_no_fraud_kes():
    s = cost_summary([0, 0], [1, 0], [1, 1], minutes=[30.0, 7.5])
    assert s["amount_weighted_recall"] is None          # capitation-only fraud: KES 0
    assert s["auditor_hours"] == {"30": 1.0, "7.5": pytest.approx(0.2)}


def test_capitation_note():
    assert "KES 0" in CAPITATION_NOTE
