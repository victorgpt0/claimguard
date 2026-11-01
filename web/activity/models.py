"""ActivityLog (#56, NFR-03): one row per security-relevant action, with who and when.
Details name the object acted on (a report id, a username, a file name), never a password
or clinical-note text."""
from __future__ import annotations

from django.conf import settings
from django.db import models


class ActivityLog(models.Model):
    class Action(models.TextChoices):
        SIGN_IN = "sign_in"
        SIGN_IN_FAILED = "sign_in_failed"
        SIGN_IN_REFUSED = "google_sign_in_refused"
        SIGN_OUT = "sign_out"
        DECISION = "decision"
        USER_CREATED = "user_created"
        USER_DEACTIVATED = "user_deactivated"
        USER_REACTIVATED = "user_reactivated"
        ROLE_CHANGED = "role_changed"
        PASSWORD_RESET = "password_reset"
        PASSWORD_CHANGED = "password_changed"
        EMAIL_LINKED = "email_linked"
        UPLOAD = "upload"
        JOB_STARTED = "job_started"
        DOWNLOAD = "download"

    user = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.PROTECT,
                             related_name="activity")  # null for a failed or refused sign-in
    action = models.CharField(max_length=40, choices=Action.choices)
    detail = models.CharField(max_length=255, blank=True)
    at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ["-at", "-id"]

    def __str__(self):
        return f"{self.at:%Y-%m-%d %H:%M} {self.user or '-'} {self.action}"

    @classmethod
    def record(cls, action: str, user=None, detail: str = "") -> "ActivityLog":
        if user is not None and not getattr(user, "is_authenticated", False):
            user = None
        return cls.objects.create(action=action, user=user, detail=(detail or "")[:255])
