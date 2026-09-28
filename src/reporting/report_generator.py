"""AuditReportGenerator (FR-07; Section 3.6.1 Class Diagram). Aggregates the SHAP
attribution, the classifier's verdict, the LLM's narrative, and claim metadata into a
structured AuditReport object for the Fraud Auditor's review (NFR-04: every report
includes both a feature-level and a plain-language rationale)."""
from dataclasses import dataclass


@dataclass
class AuditReport:
    claim_id: str
    verdict_id: str
    shap_attribution: dict
    llm_narrative: str | None  # None when verdict is "consistent" (false positive, logged)
    status: str  # pending | approved | rejected | escalated


class AuditReportGenerator:
    def assemble(self, claim, shap_attribution, verdict, narrative) -> AuditReport:
        raise NotImplementedError
