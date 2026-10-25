"""The operational data of the review workflow (Figure 4.7; FR-07 to FR-09, DR-05): submitted
claims, the classifier's verdicts, the audit reports auditors review, and the batches claims
arrive in. Training data, model checkpoints and the Feedback Dataset stay file-based (DR-05,
DR-08) and are not stored here."""
from __future__ import annotations

from django.conf import settings
from django.db import models


class Batch(models.Model):
    """A set of claims submitted together (FR-01): an auditor's upload, or a split imported
    from an earlier pipeline run."""

    class Status(models.TextChoices):
        UPLOADED = "uploaded"
        PROCESSING = "processing"
        PROCESSED = "processed"
        FAILED = "failed"
        IMPORTED = "imported"     # reports carried over from the Streamlit-era database

    name = models.CharField(max_length=200)
    uploaded_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.PROTECT,
                                    related_name="batches")
    uploaded_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.UPLOADED)
    claims_count = models.PositiveIntegerField(default=0)
    processed_at = models.DateTimeField(null=True, blank=True)
    error = models.TextField(blank=True)
    run_log = models.JSONField(null=True, blank=True)   # the pipeline's RunLog summary and settings

    class Meta:
        ordering = ["-uploaded_at"]
        verbose_name_plural = "batches"

    def __str__(self):
        return self.name


class Claim(models.Model):
    batch = models.ForeignKey(Batch, null=True, blank=True, on_delete=models.PROTECT, related_name="claims")
    claim_ref = models.CharField("claim id", max_length=64)  # the submitter's own id
    provider_id = models.CharField(max_length=64, db_index=True)
    patient_ref = models.CharField(max_length=64)
    icd11_code = models.CharField("ICD-11 code", max_length=32)
    claim_amount = models.DecimalField(max_digits=14, decimal_places=2)
    service_date = models.DateField()
    discharge_date = models.DateField(null=True, blank=True)
    clinical_notes = models.TextField()
    anomaly_score = models.FloatField(null=True, blank=True)

    class Meta:
        constraints = [models.UniqueConstraint(fields=["batch", "claim_ref"], name="unique_claim_per_batch")]

    def __str__(self):
        return self.claim_ref


class ClassifierVerdict(models.Model):
    class Verdict(models.TextChoices):
        CONSISTENT = "consistent"
        INCONSISTENT = "inconsistent"

    claim = models.ForeignKey(Claim, on_delete=models.PROTECT, related_name="verdicts")
    verdict = models.CharField(max_length=20, choices=Verdict.choices)
    confidence = models.FloatField()
    model_version = models.CharField(max_length=100)

    def __str__(self):
        return f"{self.claim} {self.verdict} ({self.confidence:.2f})"


class AuditReport(models.Model):
    class Status(models.TextChoices):
        PENDING = "pending"
        APPROVED = "approved"
        REJECTED = "rejected"
        ESCALATED = "escalated"

    class Lane(models.TextChoices):
        TEXT_INCONSISTENT = "text_inconsistent", "Text inconsistent"
        STRUCTURAL_ANOMALY = "structural_anomaly", "Structural anomaly"

    claim = models.ForeignKey(Claim, on_delete=models.PROTECT, related_name="reports")
    verdict = models.ForeignKey(ClassifierVerdict, null=True, blank=True, on_delete=models.PROTECT,
                                related_name="reports")
    shap_attribution = models.JSONField(default=dict, blank=True)
    llm_narrative = models.TextField(null=True, blank=True)   # only for inconsistent verdicts (IR-02)
    flag_sources = models.JSONField(default=list, blank=True)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.PENDING, db_index=True)
    reviewed_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.PROTECT,
                                    related_name="reviews")
    reviewed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"Report #{self.pk} ({self.claim})"

    @property
    def lane(self) -> str | None:
        """Inconsistent verdicts go to the text lane; consistent ones that stage 1 flagged
        structurally go to the structural-anomaly lane (src/pipeline.py, #46)."""
        if self.verdict is None:
            return None
        if self.verdict.verdict == ClassifierVerdict.Verdict.INCONSISTENT:
            return self.Lane.TEXT_INCONSISTENT
        return self.Lane.STRUCTURAL_ANOMALY
