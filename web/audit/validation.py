"""Upload validation (FR-01, FR-15): every problem in a claims CSV is reported by row and
column before anything is stored, and any problem rejects the whole file, so a batch is
never half-stored.

The columns are those the pipeline's detectors need (src/data/preprocessing.py), not only
the ones the review pages show. Note rules: at least five words and English only, as for the
synthetic notes (src/data/note_synthesis.validate_note); its other rules (no code in the
note, at most 150 words) keep generated training data clean and do not apply to submissions.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.data.preprocessing import repair_icd11_code

UPLOAD_COLUMNS = [
    "claim_id", "provider_id", "member_id", "beneficiary_type", "patient_age", "patient_sex",
    "facility_level", "icd11_code", "claim_amount", "service_date", "discharge_date", "clinical_notes",
]
REQUIRED_VALUES = ["claim_id", "provider_id", "member_id", "beneficiary_type", "facility_level",
                   "icd11_code", "claim_amount", "service_date", "clinical_notes"]
BENEFICIARY_TYPES = {"principal", "dependent"}
SEXES = {"female", "male"}
MAX_ROWS = 2000
MIN_NOTE_WORDS = 5


@dataclass(frozen=True)
class Problem:
    row: int | None      # the CSV line number (header = 1), or None for the whole file
    column: str
    message: str


def _blank(value) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value)) or str(value).strip() == ""


def _date(value):
    try:
        return pd.Timestamp(str(value).strip()).normalize()
    except (ValueError, TypeError):
        return None


def validate(frame: pd.DataFrame, known_codes: set[str], existing_ids: set[str],
             heldout_ids: set[str]) -> list[Problem]:
    """`frame` is the CSV read with every column as text. `known_codes` are the ICD-11 codes
    in the cache, `existing_ids` the claim ids already stored, `heldout_ids` the held-out test
    split's claim ids (DR-08: they may never reach the review queue)."""
    missing = [c for c in UPLOAD_COLUMNS if c not in frame.columns]
    if missing:
        return [Problem(None, ", ".join(missing), "required column missing")]
    if frame.empty:
        return [Problem(None, "", "the file has no claims")]
    if len(frame) > MAX_ROWS:
        return [Problem(None, "", f"{len(frame)} claims; a batch may hold at most {MAX_ROWS}")]

    problems: list[Problem] = []
    seen: dict[str, int] = {}
    for index, row in enumerate(frame.to_dict(orient="records")):
        line = index + 2

        def add(column, message):
            problems.append(Problem(line, column, message))

        for column in REQUIRED_VALUES:
            if _blank(row[column]):
                add(column, "empty")
        claim_id = str(row["claim_id"]).strip()
        if claim_id and not _blank(row["claim_id"]):
            if claim_id in seen:
                add("claim_id", f"duplicate of line {seen[claim_id]}")
            seen.setdefault(claim_id, line)
            if claim_id in existing_ids:
                add("claim_id", "already stored in an earlier batch")
            if claim_id in heldout_ids:
                add("claim_id", "belongs to the held-out test split, which never enters the review queue")
        if not _blank(row["beneficiary_type"]) and str(row["beneficiary_type"]).strip() not in BENEFICIARY_TYPES:
            add("beneficiary_type", "must be principal or dependent")
        if not _blank(row["patient_sex"]) and str(row["patient_sex"]).strip() not in SEXES:
            add("patient_sex", "must be female, male or blank")
        if not _blank(row["patient_age"]):
            age = pd.to_numeric(row["patient_age"], errors="coerce")
            if pd.isna(age) or not 0 <= age <= 120:
                add("patient_age", "must be a number from 0 to 120, or blank")
        if not _blank(row["facility_level"]):
            level = pd.to_numeric(row["facility_level"], errors="coerce")
            if pd.isna(level) or level != int(level) or not 1 <= level <= 6:
                add("facility_level", "must be a whole number from 1 to 6")
        if not _blank(row["claim_amount"]):
            amount = pd.to_numeric(row["claim_amount"], errors="coerce")
            if pd.isna(amount) or amount < 0:
                add("claim_amount", "must be a number of at least 0 (KES 0 is valid for capitation claims)")
        service = None
        if not _blank(row["service_date"]):
            service = _date(row["service_date"])
            if service is None:
                add("service_date", "not a date (use YYYY-MM-DD)")
        if not _blank(row["discharge_date"]):
            discharge = _date(row["discharge_date"])
            if discharge is None:
                add("discharge_date", "not a date (use YYYY-MM-DD)")
            elif service is not None and discharge < service:
                add("discharge_date", "before the service date")
        if not _blank(row["icd11_code"]) and repair_icd11_code(str(row["icd11_code"])) not in known_codes:
            add("icd11_code", "not an ICD-11 code this system has a reference definition for")
        notes = "" if _blank(row["clinical_notes"]) else str(row["clinical_notes"])
        if notes:
            if len(notes.split()) < MIN_NOTE_WORDS:
                add("clinical_notes", f"fewer than {MIN_NOTE_WORDS} words")
            if any(ch.isalpha() and ord(ch) > 0x024F for ch in notes):
                add("clinical_notes", "not in English (the study covers English notes only)")
    return problems
