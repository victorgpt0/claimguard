"""Synthetic fraud and text-noise injection (Section 3.2.2, Table 3.1).

Each function below implements one row of Table 3.1's Fraud Typology Traceability:
documented SHA typology -> synthetic operationalisation. Keep the mapping 1:1 so the
traceability the thesis claims actually holds in the generated data.

Claims are pandas DataFrames with at minimum: claim_id, provider_id, icd11_code,
claim_amount, service_date, discharge_date, clinical_notes, claim_type
("outpatient"/"inpatient"), tariff.
"""
from __future__ import annotations

import random
import re

import numpy as np
import pandas as pd

# Clinical shorthand substituted at a controlled rate to simulate the abbreviated,
# non-standard justification text found in operational claims (Section 3.2.2). Matching
# is whole-word/phrase and case-insensitive; multi-word phrases are listed first so they
# take precedence over their component words.
CLINICAL_ABBREVIATIONS: dict[str, str] = {
    "shortness of breath": "SOB",
    "blood pressure": "BP",
    "complains of": "c/o",
    "history of": "h/o",
    "without": "w/o",
    "with": "w/",
    "patient": "pt",
    "history": "hx",
    "diagnosis": "dx",
    "treatment": "tx",
    "prescription": "rx",
    "symptoms": "sx",
    "temperature": "temp",
    "examination": "exam",
    "admitted": "adm",
    "discharged": "d/c",
    "abdominal": "abd",
    "bilateral": "b/l",
    "intravenous": "IV",
    "twice daily": "BD",
    "three times daily": "TDS",
}

def _rng(seed: int | None) -> random.Random:
    return random.Random(seed)


def inject_upcoding(claims: pd.DataFrame, rate: float, tariff_lookup: dict[str, float],
                     higher_tariff_codes: dict[str, list[str]], seed: int | None = None) -> pd.DataFrame:
    """Replace a configurable proportion of billed procedure codes with a higher-tariff
    code while leaving clinical justification text describing the original, lower-acuity
    procedure. Table 3.1 row 1 (Kilei, 2025; Azad & William, 2024).

    `higher_tariff_codes` maps an original code to a list of plausible higher-tariff
    replacement codes for the same clinical area (e.g. a coding-adjacent upcode target).
    Claims whose code has no listed replacement are left unmodified.
    """
    rng = _rng(seed)
    out = claims.copy()
    out["is_upcoded"] = False
    eligible = out.index[out["icd11_code"].isin(higher_tariff_codes)].tolist()
    n_to_flag = int(len(eligible) * rate)
    for idx in rng.sample(eligible, min(n_to_flag, len(eligible))):
        original_code = out.at[idx, "icd11_code"]
        replacement = rng.choice(higher_tariff_codes[original_code])
        out.at[idx, "icd11_code"] = replacement
        out.at[idx, "claim_amount"] = tariff_lookup.get(replacement, out.at[idx, "claim_amount"])
        out.at[idx, "is_upcoded"] = True
    return out


def inject_outpatient_to_inpatient_conversion(claims: pd.DataFrame, rate: float,
                                               outpatient_only_codes: set[str],
                                               seed: int | None = None) -> pd.DataFrame:
    """Re-label a configurable proportion of outpatient-eligible claims as inpatient
    admissions with a synthetically extended length of stay, while clinical notes retain
    outpatient-consistent language. Table 3.1 row 2 (Kilei, 2025)."""
    rng = _rng(seed)
    out = claims.copy()
    out["is_converted_admission"] = False
    eligible = out.index[
        (out["claim_type"] == "outpatient") & (out["icd11_code"].isin(outpatient_only_codes))
    ].tolist()
    n_to_flag = int(len(eligible) * rate)
    for idx in rng.sample(eligible, min(n_to_flag, len(eligible))):
        service_date = pd.to_datetime(out.at[idx, "service_date"])
        extended_stay_days = rng.randint(1, 4)
        out.at[idx, "claim_type"] = "inpatient"
        out.at[idx, "discharge_date"] = (service_date + pd.Timedelta(days=extended_stay_days)).date().isoformat()
        out.at[idx, "is_converted_admission"] = True
    return out


def inject_ghost_billing(claims: pd.DataFrame, rate: float,
                          templated_notes: list[str], seed: int | None = None) -> pd.DataFrame:
    """Generate claims with no corresponding realistic care pathway: a billed procedure
    and tariff with templated, generic clinical justification inconsistent with the
    billed diagnosis. Table 3.1 row 3 (Kilei, 2025)."""
    rng = _rng(seed)
    out = claims.copy()
    out["is_ghost_billed"] = False
    n_to_flag = int(len(out) * rate)
    for idx in rng.sample(list(out.index), min(n_to_flag, len(out))):
        out.at[idx, "clinical_notes"] = rng.choice(templated_notes)
        out.at[idx, "is_ghost_billed"] = True
    return out


def inject_inflated_caesarean_rate(providers: pd.DataFrame, rate: float,
                                    who_recommended_max: float = 0.15,
                                    inflation_margin: float = 0.20,
                                    seed: int | None = None) -> pd.DataFrame:
    """Construct provider-level synthetic caesarean-delivery proportions so a subset of
    providers exceed the WHO-recommended threshold. Table 3.1 row 4 (Kilei, 2025;
    Section 1.1 WHO guidance). `providers` needs columns: provider_id, total_deliveries."""
    rng = np.random.default_rng(seed)
    out = providers.copy()
    out["is_inflated_csection_provider"] = False
    n_to_flag = int(len(out) * rate)
    flagged_idx = rng.choice(out.index, size=min(n_to_flag, len(out)), replace=False)
    csection_rate = np.where(
        out.index.isin(flagged_idx),
        who_recommended_max + inflation_margin + rng.uniform(0, 0.10, size=len(out)),
        rng.uniform(0.05, who_recommended_max, size=len(out)),
    )
    out["csection_deliveries"] = (out["total_deliveries"] * csection_rate).round().astype(int)
    out.loc[out.index.isin(flagged_idx), "is_inflated_csection_provider"] = True
    return out


def inject_record_fabrication(claims: pd.DataFrame, rate: float,
                               boilerplate_notes: list[str], seed: int | None = None) -> pd.DataFrame:
    """Pair a valid ICD-11 procedure code with entirely templated boilerplate clinical
    justification, unrelated to any plausible patient presentation for that code. Table
    3.1 row 5 (Kilei, 2025)."""
    rng = _rng(seed)
    out = claims.copy()
    out["is_fabricated_record"] = False
    n_to_flag = int(len(out) * rate)
    for idx in rng.sample(list(out.index), min(n_to_flag, len(out))):
        out.at[idx, "clinical_notes"] = rng.choice(boilerplate_notes)
        out.at[idx, "is_fabricated_record"] = True
    return out


def inject_anomalous_dependent_counts(providers: pd.DataFrame, rate: float,
                                       realistic_max_by_facility_type: dict[str, int],
                                       seed: int | None = None) -> pd.DataFrame:
    """Assign synthetic providers beneficiary-dependent counts exceeding what's realistic
    for their registered facility type/size. Table 3.1 row 6 (Kizito, 2026 — the Kwale
    County case)."""
    rng = np.random.default_rng(seed)
    out = providers.copy()
    out["is_anomalous_dependent_count"] = False
    n_to_flag = int(len(out) * rate)
    flagged_idx = rng.choice(out.index, size=min(n_to_flag, len(out)), replace=False)
    for idx in flagged_idx:
        facility_type = out.at[idx, "facility_type"]
        realistic_max = realistic_max_by_facility_type.get(facility_type, 20)
        out.at[idx, "dependent_count"] = realistic_max * rng.integers(3, 10)
        out.at[idx, "is_anomalous_dependent_count"] = True
    return out


def inject_text_noise(text: str, rate: float, seed: int | None = None,
                      abbreviations: dict[str, str] | None = None) -> str:
    """Make clinical justification text resemble operational claim documentation
    (Section 3.2.2): each eligible term or word is altered with probability `rate`.

    Three kinds of noise are applied, in order: clinical shorthand from `abbreviations`
    (default CLINICAL_ABBREVIATIONS), single-character typos (swap, drop or doubled
    letter) in words of four or more letters, and inconsistent casing and spacing.
    Pass an alternative `abbreviations` mapping to generate the unseen noise variants
    the held-out test set requires (DR-03). Distinct from the fraud injections above —
    applied to a stratified subset of both consistent and inconsistent examples."""
    rng = _rng(seed)
    glossary = CLINICAL_ABBREVIATIONS if abbreviations is None else abbreviations

    if glossary:
        def _maybe_abbreviate(match: re.Match) -> str:
            return glossary[match.group(0).lower()] if rng.random() < rate else match.group(0)

        phrases = sorted(glossary, key=len, reverse=True)
        pattern = r"\b(" + "|".join(re.escape(p) for p in phrases) + r")\b"
        text = re.sub(pattern, _maybe_abbreviate, text, flags=re.IGNORECASE)

    def _maybe_typo(match: re.Match) -> str:
        word = match.group(0)
        if len(word) < 4 or rng.random() >= rate:
            return word
        pos = rng.randrange(1, len(word) - 1)
        operation = rng.choice(["swap", "drop", "double"])
        if operation == "swap":
            return word[:pos] + word[pos + 1] + word[pos] + word[pos + 2:]
        if operation == "drop":
            return word[:pos] + word[pos + 1:]
        return word[:pos] + word[pos] + word[pos:]

    text = re.sub(r"[A-Za-z]{4,}", _maybe_typo, text)

    if rng.random() < rate:
        text = text.lower() if rng.random() < 0.5 else text.upper()
    if rng.random() < rate:
        text = re.sub(r" ", lambda _: "  " if rng.random() < rate else " ", text)
    return text


def apply_text_noise(claims: pd.DataFrame, fraction: float, rate: float,
                     seed: int | None = None,
                     abbreviations: dict[str, str] | None = None) -> pd.DataFrame:
    """Apply `inject_text_noise` to the clinical notes of a random `fraction` of claims
    and record which were altered in `has_text_noise`, so evaluation can report clean
    and noisy claims separately (Section 3.2.4, NFR-05)."""
    rng = _rng(seed)
    out = claims.copy()
    out["has_text_noise"] = False
    n_to_flag = int(len(out) * fraction)
    for idx in rng.sample(list(out.index), min(n_to_flag, len(out))):
        out.at[idx, "clinical_notes"] = inject_text_noise(
            out.at[idx, "clinical_notes"], rate, seed=rng.randrange(2**32),
            abbreviations=abbreviations,
        )
        out.at[idx, "has_text_noise"] = True
    return out
