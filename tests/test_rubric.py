import pandas as pd
import pytest

from src.evaluation.rubric import CRITERIA, SHEET_COLUMNS, sample_sheet, summarise

REPORTS = [{"report_id": i, "icd11_code": "1F40.Z", "clinical_notes": f"notes {i}",
            "llm_narrative": f"narrative {i}", "shap_attribution": "{}"} for i in range(80)]


def test_sample_is_fixed_size_seeded_and_blind():
    sheet = sample_sheet(REPORTS, {"1F40.Z": "Malaria"}, n=50, seed=0)
    assert len(sheet) == 50 and sheet["report_id"].is_unique
    assert list(sheet.columns) == SHEET_COLUMNS
    assert not {"label", "is_fraud", "fraud_typology", "verdict"} & set(sheet.columns)
    assert (sheet[CRITERIA] == "").all().all()
    assert (sheet["title"] == "Malaria").all() and (sheet["definition"] == "").all()
    with_definition = sample_sheet(REPORTS, {"1F40.Z": "Malaria"}, n=5, definitions={"1F40.Z": "A disease"})
    assert (with_definition["definition"] == "A disease").all()
    assert sheet.equals(sample_sheet(REPORTS, {"1F40.Z": "Malaria"}, n=50, seed=0))
    assert len(sample_sheet(REPORTS[:10], {}, n=50)) == 10


def test_summarise_mean_std_share():
    sheet = pd.DataFrame({c: [5, 4, 3, None] for c in CRITERIA})
    s = summarise(sheet)["clarity"]
    assert s == {"scored": 3, "mean": 4.0, "std": 1.0, "share_ge4": round(2 / 3, 3)}


def test_summarise_rejects_out_of_range():
    sheet = pd.DataFrame({c: [6] for c in CRITERIA})
    with pytest.raises(ValueError):
        summarise(sheet)


# SHAP coherence uses the same sample/summarise sheet pattern (task 7.6).
from src.evaluation.shap_coherence import allocate, sample_sheet as coherence_sheet, summarise as coherence_summary


def test_allocate_spreads_leftovers():
    assert allocate({"a": 50, "b": 50}, 10) == {"a": 5, "b": 5}
    assert allocate({"a": 2, "b": 50, "c": 50}, 20) == {"a": 2, "b": 9, "c": 9}
    assert allocate({"a": 3, "b": 4}, 100) == {"a": 3, "b": 4}


def test_coherence_sample_stratified_with_top3():
    typologies = ["none"] * 40 + ["upcoding"] * 5 + ["ghost_billing"] * 30
    claims = pd.DataFrame({"claim_id": [f"C{i}" for i in range(75)], "fraud_typology": typologies,
                           "icd11_code": ["1A00"] * 75})
    shap = {"stay_days": 0.1, "patient_age": -0.5, "is_female": 0.3, "facility_level": 0.01}
    log = {"claims": [{"claim_ref": f"C{i}", "flagged": i % 3 != 2, "report_id": i, "shap": shap}
                      if i % 3 != 2 else {"claim_ref": f"C{i}", "flagged": False} for i in range(75)]}
    sheet = coherence_sheet(log, claims, n=20, seed=0)
    assert len(sheet) == 20
    counts = sheet["fraud_typology"].value_counts().to_dict()
    assert counts["upcoding"] == 3                   # every flagged upcoding claim (3 of 5)
    assert counts["ghost_billing"] == 9 and counts["none"] == 8   # leftover by name order
    assert sheet[["feature_1", "feature_2", "feature_3"]].iloc[0].tolist() == ["patient_age", "is_female", "stay_days"]
    assert set(sheet["claim_ref"]) <= {f"C{i}" for i in range(75) if i % 3 != 2}
    assert (sheet["coherent"] == "").all()


def test_coherence_summary_per_typology():
    sheet = pd.DataFrame({"fraud_typology": ["upcoding", "upcoding", "none", "none"],
                          "coherent": ["y", "N", "y", None]})
    s = coherence_summary(sheet)
    assert s["overall"] == {"marked": 3, "coherent": 2, "share": 0.667}
    assert s["by_typology"]["upcoding"]["share"] == 0.5
    with pytest.raises(ValueError):
        coherence_summary(pd.DataFrame({"fraud_typology": ["x"], "coherent": ["maybe"]}))


def test_coherence_sample_uses_reported_when_logged():
    claims = pd.DataFrame({"claim_id": ["A", "B"], "fraud_typology": ["none", "upcoding"], "icd11_code": ["X", "X"]})
    log = {"claims": [{"claim_ref": "A", "flagged": True, "reported": False, "shap": {"f": 1}},
                      {"claim_ref": "B", "flagged": True, "reported": True, "lane": "text_inconsistent",
                       "shap": {"f": 1}}]}
    assert coherence_sheet(log, claims, n=10)["claim_ref"].tolist() == ["B"]
    assert len(coherence_sheet(log, claims, n=10, reported_only=False)) == 2
