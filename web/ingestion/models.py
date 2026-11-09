"""The claims feed's own records (#66): where the last pull stopped, a log of every pull, and
the claims a pull refused. The claims themselves go into audit.Claim, in the open Batch of
their audit period (audit.Batch.source = feed)."""
from __future__ import annotations

from django.conf import settings
from django.db import models


class FeedState(models.Model):
    """A single row: the cursor (the last claim received, by paid_at and the platform's
    sequence number), whether scheduled pulls are paused, and how the last one went."""

    after_paid_at = models.DateTimeField(null=True, blank=True)
    after_seq = models.PositiveBigIntegerField(default=0)
    paused = models.BooleanField(default=False)
    platform_as_of = models.DateTimeField(null=True, blank=True)   # the platform's clock at the last pull
    last_attempt_at = models.DateTimeField(null=True, blank=True)
    last_success_at = models.DateTimeField(null=True, blank=True)
    last_error = models.TextField(blank=True)
    # Set while a pull runs, so a scheduled pull and a "Sync now" never run at once and store
    # the same page twice. A pull that died is assumed gone after STALE_AFTER.
    running_since = models.DateTimeField(null=True, blank=True)

    class Meta:
        verbose_name = "feed state"

    @classmethod
    def get(cls) -> "FeedState":
        state, _ = cls.objects.get_or_create(pk=1)
        return state


class IngestionRun(models.Model):
    """One pull from the claims platform."""

    class Status(models.TextChoices):
        SUCCEEDED = "succeeded"
        PARTIAL = "partial"   # some pages stored, then the platform failed; the next pull resumes
        FAILED = "failed"

    started_at = models.DateTimeField(auto_now_add=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.SUCCEEDED)
    triggered_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.PROTECT,
                                     related_name="+")   # null for the schedule
    fetched = models.PositiveIntegerField(default=0)
    accepted = models.PositiveIntegerField(default=0)
    rejected = models.PositiveIntegerField(default=0)
    duplicates = models.PositiveIntegerField(default=0)
    platform_as_of = models.DateTimeField(null=True, blank=True)
    cursor_from = models.CharField(max_length=60, blank=True)
    cursor_to = models.CharField(max_length=60, blank=True)
    batches_closed = models.JSONField(default=list, blank=True)   # batch ids closed by this pull
    error = models.TextField(blank=True)

    class Meta:
        ordering = ["-started_at", "-id"]


class RejectedClaim(models.Model):
    """A claim the feed refused (FR-15's checks, per claim): kept with its problems for an
    auditor to follow up with the platform, never processed."""

    run = models.ForeignKey(IngestionRun, on_delete=models.CASCADE, related_name="rejections")
    claim_ref = models.CharField(max_length=64)
    paid_at = models.DateTimeField(null=True, blank=True)
    problems = models.JSONField(default=list)     # [{"column": ..., "message": ...}]
    record = models.JSONField(default=dict)       # the claim as received, without its clinical notes
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at", "-id"]
