"""AuditReportRepository (Section 4.4.2 Class Diagram interpretation): the single point of
write access to SQLite — no other class writes to the database directly. Backs FR-08
(pending-report queue + detail view) and FR-09 (recording review decisions).

`recordDecision` is also where FR-12's feedback capture hooks in (see
src/feedback/feedback_repository.py) and where IR-06's retraining-trigger check is
appended, per chapter4_system_analysis_design.txt's IR-06 interpretation: a lightweight
poll-on-event check, not a standing scheduler process.
"""
import sqlite3


class AuditReportRepository:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def get_pending_reports(self) -> list:
        """FR-08: pending-report queue for the Fraud Auditor dashboard."""
        raise NotImplementedError

    def get_report(self, report_id: int):
        """FR-08: full detail view for a single report."""
        raise NotImplementedError

    def record_decision(self, report_id: int, decision: str, reviewer_id: int):
        """FR-09: persist approve/reject/escalate + timestamp + reviewer identity.
        Where decision is approve or reject, also materialises a Feedback Dataset example
        (FR-12) via FeedbackRepository, and checks the retraining trigger (FR-13, IR-06).
        Escalate produces no feedback example (per FR-12 scope)."""
        raise NotImplementedError

    def save_report(self, report) -> int:
        """FR-07: persist a newly assembled AuditReport. Returns the new report_id."""
        raise NotImplementedError
