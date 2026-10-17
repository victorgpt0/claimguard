"""AuditReportRepository (Section 4.4.2 Class Diagram interpretation): the single point of
write access to SQLite — no other class writes to the database directly. Backs FR-08
(pending-report queue + detail view) and FR-09 (recording review decisions).

`record_decision` is also where FR-12's feedback capture hooks in (see
src/feedback/feedback_repository.py) and where IR-06's retraining-trigger check is
appended, per chapter4_system_analysis_design.txt's IR-06 interpretation: a lightweight
poll-on-event check, not a standing scheduler process.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

VALID_DECISIONS = {"approved", "rejected", "escalated"}
SCHEMA_PATH = Path(__file__).resolve().parent.parent / "db" / "schema.sql"


class AuditReportRepository:
    def __init__(self, db_path: str, ensure_schema: bool = True):
        self.db_path = db_path
        if ensure_schema:
            with self._connect() as conn:
                conn.executescript(SCHEMA_PATH.read_text())

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def save_claim(self, claim: dict) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """INSERT INTO Claim (provider_id, patient_ref, icd11_code, claim_amount,
                   service_date, discharge_date, clinical_notes, anomaly_score)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (claim["provider_id"], claim["patient_ref"], claim["icd11_code"],
                 claim["claim_amount"], claim["service_date"], claim.get("discharge_date"),
                 claim["clinical_notes"], claim.get("anomaly_score")),
            )
            return cursor.lastrowid

    def save_verdict(self, claim_id: int, verdict: str, confidence: float,
                      model_version: str) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """INSERT INTO ClassifierVerdict (claim_id, verdict, confidence, model_version)
                   VALUES (?, ?, ?, ?)""",
                (claim_id, verdict, confidence, model_version),
            )
            return cursor.lastrowid

    def save_report(self, claim_id: int, verdict_id: int | None, shap_attribution: dict,
                     llm_narrative: str | None) -> int:
        """FR-07: persist a newly assembled AuditReport. Returns the new report_id."""
        with self._connect() as conn:
            cursor = conn.execute(
                """INSERT INTO AuditReport (claim_id, verdict_id, shap_attribution,
                   llm_narrative, status) VALUES (?, ?, ?, ?, 'pending')""",
                (claim_id, verdict_id, json.dumps(shap_attribution), llm_narrative),
            )
            return cursor.lastrowid

    def get_pending_reports(self) -> list[dict]:
        """FR-08: pending-report queue for the Fraud Auditor dashboard."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT r.*, c.provider_id, c.icd11_code, c.claim_amount, c.clinical_notes
                   FROM AuditReport r JOIN Claim c ON c.id = r.claim_id
                   WHERE r.status = 'pending' ORDER BY r.created_at, r.id"""
            ).fetchall()
            return [dict(row) for row in rows]

    def get_all_reports(self) -> list[dict]:
        """Every report with its claim and verdict, for the dashboard's statistics."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT r.id, r.status, r.shap_attribution, r.llm_narrative IS NOT NULL AS has_narrative,
                          r.created_at, r.reviewed_at, c.provider_id, c.icd11_code, c.claim_amount,
                          c.anomaly_score, v.verdict, v.confidence
                   FROM AuditReport r
                   JOIN Claim c ON c.id = r.claim_id
                   LEFT JOIN ClassifierVerdict v ON v.id = r.verdict_id
                   ORDER BY r.id"""
            ).fetchall()
            return [dict(row) for row in rows]

    def get_report(self, report_id: int) -> dict | None:
        """FR-08: full detail view for a single report."""
        with self._connect() as conn:
            row = conn.execute(
                """SELECT r.*, c.provider_id, c.patient_ref, c.icd11_code, c.claim_amount,
                          c.service_date, c.discharge_date, c.clinical_notes, c.anomaly_score,
                          v.verdict, v.confidence, v.model_version
                   FROM AuditReport r
                   JOIN Claim c ON c.id = r.claim_id
                   LEFT JOIN ClassifierVerdict v ON v.id = r.verdict_id
                   WHERE r.id = ?""",
                (report_id,),
            ).fetchone()
            return dict(row) if row else None

    def record_decision(self, report_id: int, decision: str, reviewer_id: int | None = None) -> None:
        """FR-09: persist approve/reject/escalate + timestamp + reviewer identity.
        Where decision is approve or reject, the caller (dashboard/app.py) is responsible
        for also materialising a Feedback Dataset example (FR-12) via FeedbackRepository
        and checking the retraining trigger (FR-13, IR-06) — kept out of this method to
        preserve the single-write-path guarantee without coupling this repository to the
        feedback module's file-based storage concerns."""
        if decision not in VALID_DECISIONS:
            raise ValueError(f"Invalid decision {decision!r}; must be one of {VALID_DECISIONS}")
        with self._connect() as conn:
            conn.execute(
                """UPDATE AuditReport SET status = ?, reviewed_by = ?, reviewed_at = ?
                   WHERE id = ?""",
                (decision, reviewer_id, datetime.now(timezone.utc).isoformat(), report_id),
            )
