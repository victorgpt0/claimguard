import pandas as pd

from src.data.build_dataset import UNSEEN_ABBREVIATIONS
from src.data.noise_injection import (
    CLINICAL_ABBREVIATIONS,
    apply_structured_noise,
    apply_text_noise,
    inject_ghost_billing,
    inject_outpatient_to_inpatient_conversion,
    inject_record_fabrication,
    inject_text_noise,
    inject_upcoding,
)


def test_inject_upcoding_replaces_code_and_amount(sample_claims_df):
    # sample_claims_df has two rows with icd11_code "1A00" (claim_id 1 and 3), so
    # rate=1.0 must flag both.
    result = inject_upcoding(
        sample_claims_df, rate=1.0,
        tariff_lookup={"9Z99": 9999.0},
        higher_tariff_codes={"1A00": ["9Z99"]},
        seed=1,
    )
    upcoded_rows = result[result["is_upcoded"]]
    assert len(upcoded_rows) == 2
    assert (upcoded_rows["icd11_code"] == "9Z99").all()
    assert (upcoded_rows["claim_amount"] == 9999.0).all()
    # The notes still describe the original code; that code must be kept for evaluation.
    assert (upcoded_rows["original_icd11_code"] == "1A00").all()
    untouched = result[~result["is_upcoded"]]
    assert (untouched["original_icd11_code"] == untouched["icd11_code"]).all()


def test_inject_outpatient_to_inpatient_conversion(sample_claims_df):
    # Both outpatient rows (claim_id 1 and 3) carry icd11_code "1A00", so rate=1.0
    # must flag both.
    result = inject_outpatient_to_inpatient_conversion(
        sample_claims_df, rate=1.0, outpatient_only_codes={"1A00"}, seed=1
    )
    converted = result[result["is_converted_admission"]]
    assert len(converted) == 2
    assert (converted["claim_type"] == "inpatient").all()


def test_conversion_bills_per_diem_only_where_admission_is_possible(sample_claims_df):
    claims = sample_claims_df.assign(facility_level=[2, 2, 4, 4])
    result = inject_outpatient_to_inpatient_conversion(
        claims, rate=1.0, outpatient_only_codes={"1A00"}, seed=1,
        per_diem_rates={3: 2240.0, 4: 3360.0},
    )
    converted = result[result["is_converted_admission"]]
    # Claim 1 is at a Level 2 dispensary, which cannot admit; only claim 3 converts.
    assert converted["claim_id"].tolist() == [3]
    stay = (pd.to_datetime(converted["discharge_date"]) - pd.to_datetime(converted["service_date"])).dt.days
    assert (converted["claim_amount"] == 3360.0 * stay).all()


def test_inject_ghost_billing_uses_notes_from_another_chapter(sample_claims_df):
    donors = pd.DataFrame({
        "icd11_code": ["1A40.Z", "JB20.0"],
        "clinical_notes": ["Same-chapter note.", "Delivered a live infant, no complications."],
    })
    claims = sample_claims_df.assign(icd11_code="1F40.Z")
    result = inject_ghost_billing(claims, rate=0.5, donor_notes=donors, seed=1)
    flagged = result[result["is_ghost_billed"]]
    assert len(flagged) == 2
    # Chapter 1 claims may only receive the chapter J donor note.
    assert (flagged["clinical_notes"] == "Delivered a live infant, no complications.").all()


def test_inject_record_fabrication(sample_claims_df):
    result = inject_record_fabrication(
        sample_claims_df, rate=1.0, boilerplate_notes=["Generic template text."], seed=1
    )
    assert result["is_fabricated_record"].all()
    assert (result["clinical_notes"] == "Generic template text.").all()


def test_inject_text_noise_abbreviates_known_terms():
    text = "Patient complains of shortness of breath with fever."
    result = inject_text_noise(text, rate=1.0, seed=1,
                               abbreviations={"patient": "pt", "complains of": "c/o",
                                              "shortness of breath": "SOB", "with": "w/"})
    lowered = result.lower()
    assert "c/o" in lowered
    assert "sob" in lowered
    assert "shortness" not in lowered


def test_caesarean_section_is_abbreviated_in_training_noise():
    # #51: genuine caesarean notes must also appear abbreviated, or "CS" marks upcoding.
    for seed in range(5):
        result = inject_text_noise("caesarean section", rate=1.0, seed=seed)
        assert result.strip().upper() == "CS"


def test_lower_segment_caesarean_takes_precedence():
    # At rate 1.0 the four-letter "LSCS" also gets a typo, so check it stayed one token
    # rather than becoming "lower segment CS".
    for seed in range(5):
        result = inject_text_noise("lower segment caesarean section", rate=1.0, seed=seed)
        assert len(result.split()) == 1 and result.strip().upper().startswith("L")


def test_unseen_glossary_uses_a_different_caesarean_form():
    assert UNSEEN_ABBREVIATIONS["caesarean section"] == "C/S"
    assert "C/S" not in CLINICAL_ABBREVIATIONS.values()
    assert "CS" not in UNSEEN_ABBREVIATIONS.values()


def test_inject_text_noise_zero_rate_is_noop():
    text = "Patient reported severe pain and fever."
    assert inject_text_noise(text, rate=0.0, seed=1) == text


def test_inject_text_noise_is_reproducible():
    text = "Patient admitted with abdominal pain and high temperature."
    assert inject_text_noise(text, rate=0.5, seed=7) == inject_text_noise(text, rate=0.5, seed=7)


def test_inject_text_noise_typos_without_abbreviations():
    text = "Severe headache persisting overnight"
    result = inject_text_noise(text, rate=1.0, seed=3, abbreviations={})
    assert result != text
    assert result.lower() != text.lower()


def test_apply_text_noise_flags_exact_fraction(sample_claims_df):
    result = apply_text_noise(sample_claims_df, fraction=0.5, rate=1.0, seed=1)
    assert result["has_text_noise"].sum() == len(sample_claims_df) // 2
    clean = result[~result["has_text_noise"]]
    original = sample_claims_df.set_index("claim_id")["clinical_notes"]
    for _, row in clean.iterrows():
        assert row["clinical_notes"] == original[row["claim_id"]]


def _claims_for_structured_noise(n=200):
    return pd.DataFrame({
        "claim_id": [f"C{i}" for i in range(n)],
        "icd11_code": ["CA40.Z", "BA03"] * (n // 2),
        "claim_amount": [3360.0, 2240.0] * (n // 2),
        "patient_age": [40] * n,
        "patient_sex": ["female"] * n,
        "discharge_date": ["2025-05-03"] * n,
        "is_fraud": [0, 1] * (n // 2),
    })


def test_structured_noise_is_reproducible_and_flags_the_fraction():
    claims = _claims_for_structured_noise()
    a = apply_structured_noise(claims, fraction=0.10, seed=5)
    b = apply_structured_noise(claims, fraction=0.10, seed=5)
    pd.testing.assert_frame_equal(a, b)
    assert a["has_structured_noise"].sum() == 20


def test_structured_noise_never_changes_amounts_or_labels():
    claims = _claims_for_structured_noise()
    noisy = apply_structured_noise(claims, fraction=1.0, seed=1)
    pd.testing.assert_series_equal(noisy["claim_amount"], claims["claim_amount"])
    pd.testing.assert_series_equal(noisy["is_fraud"], claims["is_fraud"])
    changed = noisy[["patient_age", "patient_sex", "discharge_date", "icd11_code"]].ne(
        claims[["patient_age", "patient_sex", "discharge_date", "icd11_code"]]).any(axis=1)
    assert changed.all()


def test_malformed_codes_are_repaired_by_preprocessing():
    from src.data.preprocessing import repair_icd11_code
    claims = _claims_for_structured_noise()
    noisy = apply_structured_noise(claims, fraction=1.0, seed=2)
    malformed = noisy["icd11_code"] != claims["icd11_code"]
    assert malformed.any()
    repaired = noisy.loc[malformed, "icd11_code"].map(repair_icd11_code)
    assert (repaired == claims.loc[malformed, "icd11_code"]).all()
