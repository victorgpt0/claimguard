import json
import sys

from src.evaluation import faithfulness
from src.evaluation.faithfulness import check, cites_notes, cites_shap, names_code, title_main_words
from src.reporting.repository import AuditReportRepository

TITLE = "Malaria due to Plasmodium falciparum, unspecified"
NOTES = ("28-year-old female presented with dysuria and suprapubic pain for three days. "
         "Urinalysis positive for nitrites. Started oral nitrofurantoin.")
SHAP = {"claim_amount_per_day": 0.4, "stay_days": 0.3, "is_female": 0.01, "patient_age": -0.2}

GOOD = ("The claim is billed as malaria due to Plasmodium falciparum, unspecified, but the notes "
        "describe dysuria, suprapubic pain and positive nitrites treated with nitrofurantoin, "
        "which point to a urinary tract infection. The amount per day is also unusually high.")
GENERIC = ("The clinical notes do not support the billed diagnosis code. The claim is "
           "inconsistent and should be referred for further investigation by an auditor.")


def test_faithful_narrative_passes():
    result = check(GOOD, TITLE, NOTES, SHAP)
    assert result == {"names_code": True, "cites_notes": True, "cites_shap": True, "passed": True}


def test_generic_narrative_fails():
    result = check(GENERIC, TITLE, NOTES, SHAP)
    assert result["passed"] is False
    assert result["names_code"] is False and result["cites_notes"] is False


def test_names_code_punctuation_and_main_words_fallback():
    assert names_code("billed as MALARIA due to plasmodium-falciparum unspecified", TITLE)
    # Fallback: the head's main words, without the "unspecified" qualifier.
    assert title_main_words(TITLE) == {"malaria", "plasmodium", "falciparum"}
    assert names_code("a Plasmodium falciparum malaria diagnosis", TITLE)
    assert not names_code("a malaria diagnosis", TITLE)
    assert title_main_words("Stroke not known if ischaemic or haemorrhagic") == {"stroke"}
    assert not names_code("anything", "")


def test_cites_notes_ignores_title_words_and_generic_vocabulary():
    notes = "Patient presented with malaria symptoms and fever"
    assert not cites_notes("The patient presented with malaria.", notes, TITLE)
    assert cites_notes("Fever is recorded.", notes, TITLE)
    # "dry" and "eye" are under 4 characters, so only the 2-gram "dry eye" counts.
    assert not cites_notes("The eye was dry.", "Dry eye", TITLE)
    assert cites_notes("A dry eye.", "Dry eye", TITLE)


def test_cites_shap_humanised_and_top3_only():
    assert cites_shap("the claim_amount_per_day is high", SHAP)
    assert cites_shap("the length of stay is long", SHAP)
    assert not cites_shap("the patient is female", SHAP)      # is_female is 4th by |value|
    assert cites_shap("the patient is female", ["is_female"])


def test_cli_over_pipeline_db(tmp_path, monkeypatch):
    db = tmp_path / "run.db"
    repo = AuditReportRepository(str(db))
    for narrative in [GOOD, GENERIC, None]:
        claim_id = repo.save_claim({"provider_id": "P1", "patient_ref": "M1", "icd11_code": "1F40.Z",
                                    "claim_amount": 100.0, "service_date": "2025-01-01",
                                    "clinical_notes": NOTES})
        repo.save_report(claim_id, None, SHAP, narrative)
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({"1F40.Z": {"code": "1F40.Z", "title": TITLE, "definition": "",
                                            "inclusion": [], "exclusion": [], "definition_source": "x"}}))
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["faithfulness", "--db", str(db), "--cache", str(cache), "--out", str(out)])
    faithfulness.main()
    run = next(out.iterdir())
    summary = json.loads((run / "summary.json").read_text())
    assert summary["narratives"] == 2
    assert summary["pass_rates"]["passed"] == 0.5
