import json

import pandas as pd
import pytest

from src.baselines import rule_based
from src.baselines.rule_based import RULES, apply_rules, outpatient_only_codes, rule_counts

OUTPATIENT_ONLY = {"1F40.Z", "CA07.0"}


def _claim(claim_id, **overrides):
    claim = {
        "claim_id": claim_id, "provider_id": "PRV0001", "member_id": f"MBR{claim_id}",
        "beneficiary_type": "principal", "patient_age": 30.0, "patient_sex": "female",
        "icd11_code": "CA07.0", "claim_type": "outpatient", "service_date": "2025-05-01",
        "discharge_date": "2025-05-01", "claim_amount": 0.0,
    }
    claim.update(overrides)
    return claim


def _fired(results, claim_id):
    row = results.set_index("claim_id").loc[claim_id]
    return {rule for rule in RULES if row[rule]}


def test_clean_claim_fires_no_rule():
    results = apply_rules(pd.DataFrame([_claim("C1")]), OUTPATIENT_ONLY)
    assert _fired(results, "C1") == set()
    assert not results["flagged"].iloc[0]
    assert results["score"].iloc[0] == 0


def test_upcoding_flags_amount_above_provider_code_median():
    claims = [_claim(f"U{i}", icd11_code="BD10", claim_amount=10_000.0, provider_id="PRV0002")
              for i in range(3)]
    claims.append(_claim("U_HIGH", icd11_code="BD10", claim_amount=16_000.0, provider_id="PRV0002"))
    results = apply_rules(pd.DataFrame(claims), OUTPATIENT_ONLY)
    assert _fired(results, "U_HIGH") == {"rule_upcoding"}
    assert _fired(results, "U0") == set()


def test_upcoding_skips_zero_amount_claims():
    claims = [_claim(f"Z{i}") for i in range(3)]
    assert not apply_rules(pd.DataFrame(claims), OUTPATIENT_ONLY)["rule_upcoding"].any()


def test_conversion_flags_short_inpatient_stay_for_outpatient_only_code():
    claims = pd.DataFrame([
        _claim("CONV", claim_type="inpatient", discharge_date="2025-05-02", claim_amount=2240.0),
        _claim("LONG", claim_type="inpatient", discharge_date="2025-05-04", claim_amount=6720.0,
               member_id="MBR_LONG", provider_id="PRV0009"),
    ])
    results = apply_rules(claims, OUTPATIENT_ONLY)
    assert "rule_conversion" in _fired(results, "CONV")
    assert "rule_conversion" not in _fired(results, "LONG")  # stay above the fixed limit


def test_duplicate_flags_both_claims_with_same_member_code_and_date():
    claims = pd.DataFrame([_claim("D1", member_id="MBR9"), _claim("D2", member_id="MBR9"),
                           _claim("D3", member_id="MBR9", service_date="2025-06-01",
                                  discharge_date="2025-06-01")])
    results = apply_rules(claims, OUTPATIENT_ONLY)
    assert _fired(results, "D1") == _fired(results, "D2") == {"rule_duplicate"}
    assert _fired(results, "D3") == set()


def test_caesarean_rate_flags_only_the_providers_caesarean_claims():
    # 2 caesareans of 4 deliveries = 50% > 25%.
    claims = [_claim("CS1", icd11_code="JB22.1", claim_amount=30_000.0),
              _claim("CS2", icd11_code="JB22.1", claim_amount=30_000.0),
              _claim("N1", icd11_code="JB20.0", claim_amount=10_000.0),
              _claim("N2", icd11_code="JB20.0", claim_amount=10_000.0),
              # Another provider at 1 of 5 = 20%, under the threshold.
              _claim("OK_CS", icd11_code="JB22.1", claim_amount=30_000.0, provider_id="PRV0003"),
              *[_claim(f"OK_N{i}", icd11_code="JB20.0", claim_amount=10_000.0, provider_id="PRV0003")
                for i in range(4)]]
    results = apply_rules(pd.DataFrame(claims), OUTPATIENT_ONLY)
    assert _fired(results, "CS1") == {"rule_caesarean_rate"}
    assert _fired(results, "N1") == set()
    assert _fired(results, "OK_CS") == set()


def test_dependents_flags_household_above_fixed_size():
    many = [_claim(f"DEP{i}", member_id="MBR_H", beneficiary_type="dependent",
                   patient_age=float(i + 1), service_date=f"2025-05-{i + 1:02d}",
                   discharge_date=f"2025-05-{i + 1:02d}")
            for i in range(rule_based.MAX_DEPENDENTS_PER_PRINCIPAL + 1)]
    few = [_claim(f"REAL{i}", member_id="MBR_R", beneficiary_type="dependent",
                  patient_age=float(i + 1), service_date=f"2025-05-{i + 1:02d}",
                  discharge_date=f"2025-05-{i + 1:02d}")
           for i in range(rule_based.MAX_DEPENDENTS_PER_PRINCIPAL)]
    principal = _claim("PRINC", member_id="MBR_H", service_date="2025-06-01",
                       discharge_date="2025-06-01")
    results = apply_rules(pd.DataFrame([*many, *few, principal]), OUTPATIENT_ONLY)
    assert _fired(results, "DEP0") == {"rule_dependents"}
    assert _fired(results, "REAL0") == set()
    assert _fired(results, "PRINC") == set()  # only the dependents' claims are flagged


def test_score_counts_rules_and_malformed_codes_are_repaired():
    # A duplicated, converted claim billed with a malformed code fires two rules.
    claims = pd.DataFrame([
        _claim(f"X{i}", member_id="MBR_X", icd11_code=" ca07.0", claim_type="inpatient",
               discharge_date="2025-05-02", claim_amount=2240.0) for i in range(2)])
    results = apply_rules(claims, OUTPATIENT_ONLY)
    assert _fired(results, "X0") == {"rule_conversion", "rule_duplicate"}
    assert results["score"].iloc[0] == 2


def test_outpatient_only_codes_are_the_capitation_codes():
    codes = outpatient_only_codes()
    assert {"1F40.Z", "CA07.0"} <= codes
    assert "JB22.1" not in codes


def test_rule_counts_breaks_down_by_typology():
    claims = pd.DataFrame([_claim("D1", member_id="M"), _claim("D2", member_id="M")])
    claims["is_fraud"] = [1, 0]
    claims["fraud_typology"] = ["ghost_billing", "none"]
    counts = rule_counts(apply_rules(claims, OUTPATIENT_ONLY), claims)
    assert counts["rules"]["rule_duplicate"] == {"fired": 2, "fraud": 1,
                                                 "by_typology": {"ghost_billing": 1, "none": 1}}
    json.dumps(counts)


def test_cli_writes_predictions_and_counts(tmp_path, monkeypatch):
    data = tmp_path / "v0"
    data.mkdir()
    claims = pd.DataFrame([_claim("D1", member_id="M"), _claim("D2", member_id="M"), _claim("C")])
    claims["is_fraud"] = [1, 1, 0]
    claims["fraud_typology"] = ["ghost_billing", "ghost_billing", "none"]
    claims.to_parquet(data / "validation.parquet", index=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["rule_based", "--data", str(data), "--split", "validation"])
    rule_based.main()

    predictions = pd.read_csv(tmp_path / "experiments/predictions/rule_based/validation.csv")
    assert list(predictions.columns) == ["claim_id", "flagged", "score"]
    assert predictions["flagged"].tolist() == [1, 1, 0]
    counts = json.loads((tmp_path / "experiments/baselines/rule_based/v0_validation.json").read_text())
    assert counts["rules"]["rule_duplicate"]["fired"] == 2


def test_cli_refuses_test_split_without_final(monkeypatch):
    monkeypatch.setattr("sys.argv", ["rule_based", "--data", "x", "--split", "test"])
    with pytest.raises(SystemExit):
        rule_based.main()
