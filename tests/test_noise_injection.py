import pandas as pd

from src.data.noise_injection import (
    inject_anomalous_dependent_counts,
    inject_code_switching,
    inject_ghost_billing,
    inject_inflated_caesarean_rate,
    inject_outpatient_to_inpatient_conversion,
    inject_record_fabrication,
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


def test_inject_code_switching_replaces_known_terms():
    text = "Patient reported severe pain and fever."
    result = inject_code_switching(text, substitution_rate=1.0, seed=1)
    assert "mgonjwa" in result.lower()
    assert "maumivu" in result.lower()
    assert "homa" in result.lower()


def test_inject_code_switching_zero_rate_is_noop():
    text = "Patient reported severe pain and fever."
    result = inject_code_switching(text, substitution_rate=0.0, seed=1)
    assert result == text


def test_inject_code_switching_preserves_unknown_words():
    text = "The xylophone remained untouched."
    result = inject_code_switching(text, substitution_rate=1.0, seed=1)
    assert result == text
