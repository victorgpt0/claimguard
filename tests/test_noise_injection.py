import pandas as pd

from src.data.noise_injection import (
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
