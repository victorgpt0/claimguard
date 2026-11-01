"""User (Figure 4.7, NFR-03): one account per person with one of the two roles of the use case
diagram. Accounts are deactivated, never deleted, so every decision stays attributable to its
reviewer (FR-09); the last active Administrator can never be deactivated or demoted, so the
system cannot lock itself out."""
from __future__ import annotations

from django.contrib.auth.models import AbstractUser
from django.core.exceptions import ValidationError
from django.db import models
from django.db.models.functions import Lower


class User(AbstractUser):
    class Role(models.TextChoices):
        AUDITOR = "Auditor", "Fraud Auditor"
        ADMINISTRATOR = "Administrator", "System Administrator"

    role = models.CharField(max_length=20, choices=Role.choices, default=Role.AUDITOR)
    # The Google account linked to this user (#61): a verified Google email signs in only as
    # the user whose email it is. Blank means no Google sign-in.
    email = models.EmailField("Google account email", blank=True)

    class Meta:
        constraints = [models.UniqueConstraint(Lower("email"), condition=~models.Q(email=""),
                                               name="unique_linked_email")]

    @property
    def is_administrator(self) -> bool:
        return self.role == self.Role.ADMINISTRATOR

    @property
    def is_auditor(self) -> bool:
        return self.role == self.Role.AUDITOR

    def clean(self):
        super().clean()
        self.email = self.email.strip().lower()
        if self.pk and self._removes_last_administrator():
            raise ValidationError("The last active Administrator cannot be deactivated or made an Auditor.")

    def _removes_last_administrator(self) -> bool:
        before = type(self).objects.filter(pk=self.pk).first()
        if before is None or not (before.is_active and before.is_administrator):
            return False
        if self.is_active and self.is_administrator:
            return False
        others = type(self).objects.filter(role=self.Role.ADMINISTRATOR, is_active=True).exclude(pk=self.pk)
        return not others.exists()

    def save(self, *args, **kwargs):
        if self.pk and self._removes_last_administrator():
            raise ValidationError("The last active Administrator cannot be deactivated or made an Auditor.")
        self.email = (self.email or "").strip().lower()
        # The Administrator manages accounts in Django's admin site; Auditors never reach it.
        self.is_staff = self.is_superuser = self.is_administrator
        super().save(*args, **kwargs)

    def delete(self, *args, **kwargs):
        raise ValidationError("Accounts are deactivated, never deleted, so decisions stay attributable.")
