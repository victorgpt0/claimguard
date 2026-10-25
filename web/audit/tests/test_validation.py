"""Upload validation (FR-15): one positive and one negative case per rule."""
import pandas as pd
import pytest

from audit.validation import MAX_ROWS, UPLOAD_COLUMNS, validate

GOOD = {"claim_id": "CLM1", "provider_id": "PRV1", "member_id": "M1", "beneficiary_type": "principal",
        "patient_age": "34", "patient_sex": "female", "facility_level": "4", "icd11_code": "DB10.02",
        "claim_amount": "67200", "service_date": "2026-01-10", "discharge_date": "2026-01-13",
        "clinical_notes": "Right lower quadrant pain and fever; appendicectomy done."}
CODES = {"DB10.02", "JB20.0"}


def check(*rows, existing=(), heldout=()):
    frame = pd.DataFrame([{**GOOD, **row} for row in rows], columns=UPLOAD_COLUMNS, dtype=str)
    return [(p.row, p.column, p.message) for p in validate(frame, CODES, set(existing), set(heldout))]


def test_a_good_claim_passes():
    assert check({}) == []
    # Blank age, sex and discharge date are allowed; KES 0 (capitation) is valid; codes are repaired.
    assert check({"patient_age": "", "patient_sex": "", "discharge_date": "", "claim_amount": "0",
                  "icd11_code": " db1002 "}) == []


@pytest.mark.parametrize("change, column, message", [
    ({"claim_id": ""}, "claim_id", "empty"),
    ({"clinical_notes": ""}, "clinical_notes", "empty"),
    ({"beneficiary_type": "spouse"}, "beneficiary_type", "principal or dependent"),
    ({"patient_sex": "f"}, "patient_sex", "female, male"),
    ({"patient_age": "140"}, "patient_age", "0 to 120"),
    ({"facility_level": "3.5"}, "facility_level", "whole number"),
    ({"facility_level": "9"}, "facility_level", "1 to 6"),
    ({"claim_amount": "-5"}, "claim_amount", "at least 0"),
    ({"claim_amount": "abc"}, "claim_amount", "at least 0"),
    ({"service_date": "10/31/2026x"}, "service_date", "not a date"),
    ({"discharge_date": "2026-01-01"}, "discharge_date", "before the service date"),
    ({"icd11_code": "ZZ99"}, "icd11_code", "reference definition"),
    ({"clinical_notes": "Fever, cough."}, "clinical_notes", "fewer than 5 words"),
    ({"clinical_notes": "患者 presented with fever and cough today"}, "clinical_notes", "not in English"),
])
def test_each_rule_reports_its_row_and_column(change, column, message):
    problems = check({}, {"claim_id": "CLM2", **change})
    assert len(problems) == 1
    row, col, text = next(p for p in problems if p[1] == column)
    assert row == 3 and message in text  # line 3: the header is line 1, the good claim line 2


def test_duplicates_stored_ids_and_the_test_split():
    problems = check({}, {})
    assert (3, "claim_id", "duplicate of line 2") in problems
    assert any("earlier batch" in p[2] for p in check({}, existing={"CLM1"}))
    assert any("held-out test split" in p[2] for p in check({}, heldout={"CLM1"}))


def test_whole_file_problems():
    assert check()[0][2] == "the file has no claims"
    frame = pd.DataFrame([GOOD]).drop(columns=["member_id", "facility_level"])
    assert validate(frame, CODES, set(), set())[0].column == "member_id, facility_level"
    big = pd.DataFrame([{**GOOD, "claim_id": f"C{i}"} for i in range(MAX_ROWS + 1)])
    assert "at most" in validate(big, CODES, set(), set())[0].message
