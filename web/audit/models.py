"""The operational data of the review workflow (Figure 4.7; FR-07 to FR-09, DR-05): submitted
claims, the classifier's verdicts, the audit reports auditors review, and the batches claims
arrive in. Training data, model checkpoints and the Feedback Dataset stay file-based (DR-05,
DR-08) and are not stored here."""
from __future__ import annotations

from django.conf import settings
from django.db import models


class Batch(models.Model):
    """A set of claims audited together (FR-01): an auditor's upload, one audit period of the
    claims feed (#66), or a split imported from an earlier pipeline run."""

    class Status(models.TextChoices):
        OPEN = "open"             # a feed period still collecting claims; closes when the period ends
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
    # The uploaded CSV as received (FR-01), kept in the database rather than as a file: the job
    # processes it on whichever worker takes it, including the Colab worker, which shares the
    # database but not this machine's disk. Every column the detectors need is kept, not only
    # the ones the review pages show. Uploads are at most 10 MB.
    source_csv = models.TextField(blank=True)
    source_sha256 = models.CharField(max_length=64, blank=True)

    class Source(models.TextChoices):
        UPLOAD = "upload", "Manual upload"
        FEED = "feed", "Claims feed"
    # Where the claims came from: an auditor's ad hoc upload, or the scheduled pull from the
    # claims platform (#66), which collects one batch per audit period (period_start..period_end,
    # by the date SHA paid the claim).
    source = models.CharField(max_length=10, choices=Source.choices, default=Source.UPLOAD, db_index=True)
    period_start = models.DateField(null=True, blank=True)
    period_end = models.DateField(null=True, blank=True)
    closed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-uploaded_at"]
        verbose_name_plural = "batches"

    def __str__(self):
        return self.name


class Claim(models.Model):
    batch = models.ForeignKey(Batch, null=True, blank=True, on_delete=models.PROTECT, related_name="claims")
    claim_ref = models.CharField("claim id", max_length=64)  # the submitter's own id
    provider_id = models.CharField(max_length=64, db_index=True)
    patient_ref = models.CharField("member id", max_length=64)
    beneficiary_type = models.CharField(max_length=20, blank=True)
    patient_age = models.FloatField(null=True, blank=True)
    patient_sex = models.CharField(max_length=10, blank=True)
    facility_level = models.PositiveSmallIntegerField(null=True, blank=True)
    icd11_code = models.CharField("ICD-11 code", max_length=32)
    claim_amount = models.DecimalField(max_digits=14, decimal_places=2)
    service_date = models.DateField()
    discharge_date = models.DateField(null=True, blank=True)
    clinical_notes = models.TextField()
    paid_at = models.DateTimeField(null=True, blank=True)   # when SHA paid it (feed claims only)
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
    # The classifier's softmax output for the predicted class (0.5-1.0). Not calibrated, so
    # the interface and exports call it "model score", never confidence or a probability.
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


class Job(models.Model):
    """One background run (Celery) over a batch, with its progress for the batch page. The
    queue says where it runs: "cpu" on a local worker (stage 1, SHAP and the classifier, no
    narratives), "gpu" on the Colab worker notebook (the same, with LLM narratives)."""

    class Status(models.TextChoices):
        QUEUED = "queued"
        RUNNING = "running"
        SUCCEEDED = "succeeded"
        FAILED = "failed"

    class Queue(models.TextChoices):
        CPU = "cpu", "Local CPU (no narratives)"
        GPU = "gpu", "Colab GPU (with narratives)"

    batch = models.ForeignKey(Batch, on_delete=models.PROTECT, related_name="jobs")
    queue = models.CharField(max_length=10, choices=Queue.choices)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.QUEUED)
    stage = models.CharField(max_length=40, blank=True)
    done = models.PositiveIntegerField(default=0)
    total = models.PositiveIntegerField(default=0)
    message = models.TextField(blank=True)
    worker = models.CharField(max_length=200, blank=True)
    task_id = models.CharField(max_length=64, blank=True)
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.PROTECT,
                                   related_name="jobs")
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"Job #{self.pk} ({self.batch}, {self.queue})"

    @property
    def active(self) -> bool:
        return self.status in (self.Status.QUEUED, self.Status.RUNNING)

    @property
    def percent(self) -> int:
        return round(100 * self.done / self.total) if self.total else 0
