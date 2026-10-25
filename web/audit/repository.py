"""The pipeline's repository interface (save_claim, save_verdict, save_report; see
src/reporting/repository.py) backed by this application's database, so the unchanged
AuditPipeline writes a job's results straight into Django. The claims already exist (stored
at upload), so save_claim finds the batch's claim and records its anomaly score."""
from __future__ import annotations

from .models import AuditReport, Batch, Claim, ClassifierVerdict


class DjangoAuditRepository:
    def __init__(self, batch: Batch):
        self.batch = batch

    def save_claim(self, claim: dict) -> int:
        stored = Claim.objects.get(batch=self.batch, claim_ref=claim["claim_ref"])
        stored.anomaly_score = claim.get("anomaly_score")
        stored.save(update_fields=["anomaly_score"])
        return stored.pk

    def save_verdict(self, claim_id: int, verdict: str, confidence: float, model_version: str) -> int:
        return ClassifierVerdict.objects.create(claim_id=claim_id, verdict=verdict, confidence=confidence,
                                                model_version=model_version).pk

    def save_report(self, claim_id: int, verdict_id: int | None, shap_attribution: dict,
                    llm_narrative: str | None, flag_sources: list[str] | None = None) -> int:
        return AuditReport.objects.create(claim_id=claim_id, verdict_id=verdict_id,
                                          shap_attribution=shap_attribution, llm_narrative=llm_narrative,
                                          flag_sources=flag_sources or []).pk
