"""End-to-end dataset construction (stages A -> B -> C) at small scale. Stage B's LLM is
replaced by a fake that echoes the requested condition, so this checks the plumbing and
labelling, not note quality."""
import json
import re

import pandas as pd
import pytest

from src.data.build_dataset import UNSEEN_ABBREVIATIONS, build
from src.data.generator import GeneratorConfig, generate_base_claims, load_catalogue, note_requests
from src.data.note_synthesis import FINDINGS_ONLY_STYLE, build_prompt, run, validate_boilerplate, validate_note

N_CLAIMS = 1200


_GENERIC_WORDS = ["accordingly", "as advised", "per protocol", "as indicated", "and settled",
                  "without incident", "satisfactorily", "as planned"]
_GENERIC_ENDINGS = ["Stable.", "Review if need arises.", "Continue care.", "Discharged home."]


def fake_llm(prompts, seeds):
    notes = []
    for messages, seed in zip(prompts, seeds):
        user = messages[-1]["content"]
        condition = re.search(r"Condition: (.+?)\.\n", user)
        if condition:
            notes.append(f"Patient reviewed with history of {condition.group(1).lower()}, "
                         f"examination and treatment given, patient admitted then discharged.")
        else:
            notes.append(f"Patient seen and reviewed, managed {_GENERIC_WORDS[seed % 8]}. "
                         f"{_GENERIC_ENDINGS[(seed // 8) % 4]}")
    return notes


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw")
    out = tmp_path_factory.mktemp("processed")
    config = GeneratorConfig(n_claims=N_CLAIMS, n_providers=40, seed=7)
    catalogue = load_catalogue()
    claims, _ = generate_base_claims(catalogue, config)
    claims.to_parquet(raw / "base_claims.parquet", index=False)
    with open(raw / "note_requests.jsonl", "w") as f:
        for request in note_requests(claims, catalogue):
            f.write(json.dumps(request) + "\n")
    (raw / "generator_config.json").write_text(json.dumps(
        {"n_claims": N_CLAIMS, "fraud_prevalence": 0.10, "n_typologies": 6}))
    run(raw / "note_requests.jsonl", raw / "notes.jsonl", raw / "boilerplate_notes.jsonl",
        fake_llm, seed=7, n_boilerplate=64)
    manifest = build(raw, out, seed=7)
    splits = {name: pd.read_parquet(out / f"{name}.parquet")
              for name in ("train", "validation", "test", "deployment_pool")}
    return claims, manifest, splits


def test_catalogue_is_consistent():
    catalogue = load_catalogue()
    assert len(catalogue["conditions"]) >= 30
    for condition in catalogue["conditions"]:
        assert condition["payment"] in {"capitation", "per_diem", "case_rate"}
        assert (condition["payment"] == "case_rate") == ("tariff_kes" in condition)


def test_generator_produces_exact_size_and_both_provider_typologies(built):
    claims, _, _ = built
    assert len(claims) == N_CLAIMS
    assert claims["claim_id"].is_unique
    assert claims["is_inflated_csection"].sum() > 0
    assert claims["is_anomalous_dependent"].sum() > 0


def test_generator_is_reproducible():
    catalogue = load_catalogue()
    config = GeneratorConfig(n_claims=300, n_providers=30, seed=3)
    a, _ = generate_base_claims(catalogue, config)
    b, _ = generate_base_claims(catalogue, config)
    pd.testing.assert_frame_equal(a, b)


def test_claims_respect_facility_levels_and_tariffs(built):
    claims, _, _ = built
    conditions = {c["code"]: c for c in load_catalogue()["conditions"]}
    for row in claims.itertuples():
        condition = conditions[row.icd11_code]
        assert row.facility_level in condition["levels"]
        if condition["payment"] == "capitation":
            assert row.claim_amount == 0
        elif condition["payment"] == "case_rate":
            assert row.claim_amount == condition["tariff_kes"]


def test_splits_are_disjoint_and_cover_every_claim(built):
    claims, _, splits = built
    ids = [set(df["claim_id"]) for df in splits.values()]
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            assert not ids[i] & ids[j]
    assert sum(len(s) for s in ids) == len(claims)


def test_fraud_prevalence_and_one_typology_per_claim(built):
    _, manifest, splits = built
    everything = pd.concat(splits.values())
    assert 0.09 <= everything["is_fraud"].mean() <= 0.11
    assert set(everything["fraud_typology"]) == {
        "none", "upcoding", "outpatient_to_inpatient", "ghost_billing",
        "inflated_caesarean", "record_fabrication", "anomalous_dependents"}
    flags = everything[["is_upcoded", "is_converted_admission", "is_ghost_billed",
                        "is_inflated_csection", "is_fabricated_record", "is_anomalous_dependent"]]
    assert (flags.sum(axis=1) <= 1).all()
    assert (everything["is_fraud"] == flags.any(axis=1).astype(int)).all()


def test_classifier_label_follows_typology(built):
    _, _, splits = built
    everything = pd.concat(splits.values())
    inconsistent = everything["label"] == "inconsistent"
    assert set(everything.loc[inconsistent, "fraud_typology"]) == {
        "upcoding", "ghost_billing", "record_fabrication"}


def test_upcoded_claims_keep_original_code_and_bill_more(built):
    _, _, splits = built
    upcoded = pd.concat(splits.values()).query("is_upcoded")
    assert len(upcoded) > 0
    assert (upcoded["original_icd11_code"] != upcoded["icd11_code"]).all()
    tariffs = {c["code"]: c.get("tariff_kes") for c in load_catalogue()["conditions"]}
    assert (upcoded["claim_amount"] > upcoded["original_icd11_code"].map(tariffs)).all()


def test_converted_admissions_are_billed_per_diem(built):
    _, _, splits = built
    converted = pd.concat(splits.values()).query("is_converted_admission")
    assert len(converted) > 0
    assert (converted["claim_type"] == "inpatient").all()
    assert (converted["claim_amount"] > 0).all()
    assert (converted["facility_level"] >= 3).all()


def test_unseen_noise_glossary_only_in_test_and_pool(built):
    _, _, splits = built
    unseen_forms = [v for v in UNSEEN_ABBREVIATIONS.values() if len(v) > 2]

    def uses_unseen(df):
        return df["clinical_notes"].apply(lambda t: any(f in t for f in unseen_forms)).any()

    assert not uses_unseen(splits["train"])
    assert uses_unseen(splits["test"])


def test_manifest_records_inputs_and_counts(built):
    _, manifest, splits = built
    assert set(manifest["inputs"]) >= {"base_claims.parquet", "notes.jsonl", "boilerplate_notes.jsonl"}
    assert manifest["splits"]["test"]["claims"] == len(splits["test"])


def test_prompt_is_deterministic_and_never_contains_the_code():
    request = {"claim_id": "CLM000001", "icd11_code": "CA40.Z", "diagnosis": "Pneumonia, organism unspecified",
               "procedure": "", "package": "medical_inpatient", "patient_age": 40,
               "patient_sex": "male", "facility_type": "county_referral_hospital", "length_of_stay": 4}
    assert build_prompt(request, 1) == build_prompt(request, 1)
    assert "CA40" not in json.dumps(build_prompt(request, 1))


def test_some_prompts_ask_for_findings_without_the_diagnosis():
    # #51: a share of notes must not name the diagnosis, and those prompts must not also
    # invite the model to name it.
    request = {"claim_id": "CLM000001", "icd11_code": "JA00", "diagnosis": "Single spontaneous delivery",
               "procedure": "", "package": "maternity", "patient_age": 25,
               "patient_sex": "female", "facility_type": "health_centre", "length_of_stay": 1}
    users = [build_prompt({**request, "claim_id": f"CLM{i:06d}"}, 1)[1]["content"] for i in range(60)]
    findings_only = [u for u in users if FINDINGS_ONLY_STYLE in u]
    assert findings_only and len(findings_only) < len(users)
    assert not any("name the working diagnosis" in u for u in findings_only)


def test_validate_note_rejects_code_leaks():
    request = {"icd11_code": "CA40.Z"}
    assert validate_note("Pt with pneumonia, coded CA40.Z, given IV antibiotics.", request)
    assert validate_note("Admitted 2024 with productive cough and fever, crackles at bases.", request) is None


def test_validate_boilerplate_rejects_clinical_specifics():
    assert validate_boilerplate("Patient reviewed and managed as per protocol. Stable for discharge.") is None
    assert validate_boilerplate("Patient reviewed for fever and cough, given antibiotics.")
    assert validate_boilerplate("**Clinical Note** Patient reviewed and managed as per protocol.")
    assert validate_boilerplate("Seen on 12/03 and managed as per protocol, stable now.")


def test_test_split_fabrication_notes_are_unseen_in_training(built):
    _, manifest, splits = built
    assert manifest["boilerplate_notes"]["test_pool"] > 0

    def fabrication_notes(df):
        # Compare before text noise: noisy notes differ from their source anyway.
        return set(df.loc[df["is_fabricated_record"] & ~df["has_text_noise"], "clinical_notes"])

    assert fabrication_notes(splits["test"])
    assert not fabrication_notes(splits["train"]) & fabrication_notes(splits["test"])


def test_structured_noise_is_applied_and_preprocessing_copes(built):
    from src.data.preprocessing import ClaimsPreprocessor
    _, manifest, splits = built
    for name, df in splits.items():
        assert 0.09 <= df["has_structured_noise"].mean() <= 0.11, name
        assert manifest["splits"][name]["structured_noise"] == df["has_structured_noise"].sum()
    preprocessor = ClaimsPreprocessor()
    train = preprocessor.align_schema(splits["train"])
    preprocessor.feature_matrix(train, fit=True)
    test = preprocessor.align_schema(splits["test"])
    assert not pd.isna(preprocessor.feature_matrix(test)).any()
    assert not preprocessor.validate_quality(test)["quality_flag_invalid_code"].any()


def test_validate_note_rejects_notes_that_switch_language():
    """v0.6 had 156 notes (0.8%) where the note model switched to Chinese mid-note."""
    assert validate_note("Presented with severe headache and 急剧增加的心率，血压显著升高。") == "not in English"
    assert validate_note("Temp 37.8°C, pt c/o dysuria; naïve to antibiotics, café-au-lait spots noted.") is None
