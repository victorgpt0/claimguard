import pandas as pd

from src.data.noise_injection import (
    apply_text_noise,
    inject_anomalous_dependent_counts,
    inject_ghost_billing,
    inject_inflated_caesarean_rate,
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


def test_inject_outpatient_to_inpatient_conversion(sample_claims_df):
    # Both outpatient rows (claim_id 1 and 3) carry icd11_code "1A00", so rate=1.0
    # must flag both.
    result = inject_outpatient_to_inpatient_conversion(
        sample_claims_df, rate=1.0, outpatient_only_codes={"1A00"}, seed=1
    )
    converted = result[result["is_converted_admission"]]
    assert len(converted) == 2
    assert (converted["claim_type"] == "inpatient").all()


def test_inject_ghost_billing_replaces_notes(sample_claims_df):
    result = inject_ghost_billing(
        sample_claims_df, rate=0.5, templated_notes=["Boilerplate note."], seed=1
    )
    flagged = result[result["is_ghost_billed"]]
    assert len(flagged) == 2
    assert (flagged["clinical_notes"] == "Boilerplate note.").all()


def test_inject_record_fabrication(sample_claims_df):
    result = inject_record_fabrication(
        sample_claims_df, rate=1.0, boilerplate_notes=["Generic template text."], seed=1
    )
    assert result["is_fabricated_record"].all()
    assert (result["clinical_notes"] == "Generic template text.").all()


def test_inject_inflated_caesarean_rate():
    providers = pd.DataFrame({"provider_id": ["P1", "P2", "P3", "P4"],
                               "total_deliveries": [100, 100, 100, 100]})
    result = inject_inflated_caesarean_rate(providers, rate=0.5, seed=1)
    flagged = result[result["is_inflated_csection_provider"]]
    assert len(flagged) == 2
    assert (flagged["csection_deliveries"] / flagged["total_deliveries"] > 0.15).all()


def test_inject_anomalous_dependent_counts():
    providers = pd.DataFrame({
        "provider_id": ["P1", "P2"],
        "facility_type": ["dispensary", "hospital"],
        "dependent_count": [5, 10],
    })
    realistic_max = {"dispensary": 10, "hospital": 50}
    result = inject_anomalous_dependent_counts(
        providers, rate=1.0, realistic_max_by_facility_type=realistic_max, seed=1
    )
    assert result["is_anomalous_dependent_count"].all()
    for _, row in result.iterrows():
        assert row["dependent_count"] > realistic_max[row["facility_type"]]


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
