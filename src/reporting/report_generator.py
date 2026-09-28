"""AuditReportGenerator (FR-07; Section 3.6.1 Class Diagram). Aggregates the SHAP
attribution, the classifier's verdict, the LLM's narrative, and claim metadata into a
structured AuditReport, then persists it via AuditReportRepository (NFR-04: every report
includes both a feature-level and a plain-language rationale where applicable)."""
from __future__ import annotations

from src.reporting.repository import AuditReportRepository


class AuditReportGenerator:
    def __init__(self, repository: AuditReportRepository):
        self.repository = repository

    def assemble_and_save(self, claim_id: int, verdict_id: int | None,
                           shap_attribution: dict, llm_narrative: str | None) -> int:
        """`llm_narrative` is None for 'consistent' verdicts (false positive, logged) —
        the LLM is never invoked for those (Section 2.5's classifier-gates-the-LLM
        guarantee)."""
        return self.repository.save_report(claim_id, verdict_id, shap_attribution, llm_narrative)
