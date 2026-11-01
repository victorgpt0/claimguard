"""Writes sign-ins, failed sign-ins, sign-outs and password changes to the activity log.
A failed sign-in records the attempted username only, never the password."""
from allauth.account.signals import password_changed, password_reset
from allauth.socialaccount.models import SocialLogin
from django.contrib.auth.signals import user_logged_in, user_logged_out, user_login_failed
from django.dispatch import receiver

from .models import ActivityLog


@receiver(user_logged_in)
def signed_in(sender, request, user, **kwargs):
    via = "google" if isinstance(getattr(request, "_socialaccount_login", None), SocialLogin) or \
        request.path.startswith("/accounts/google/") else "password"
    ActivityLog.record(ActivityLog.Action.SIGN_IN, user, via)


@receiver(user_login_failed)
def sign_in_failed(sender, credentials, request=None, **kwargs):
    username = str(credentials.get("username") or credentials.get("login") or "")[:64]
    ActivityLog.record(ActivityLog.Action.SIGN_IN_FAILED, detail=f"username: {username}")


@receiver(user_logged_out)
def signed_out(sender, request, user, **kwargs):
    if user is not None:
        ActivityLog.record(ActivityLog.Action.SIGN_OUT, user)


@receiver(password_changed)
def changed_own_password(sender, request, user, **kwargs):
    ActivityLog.record(ActivityLog.Action.PASSWORD_CHANGED, user)


@receiver(password_reset)
def reset_password(sender, request, user, **kwargs):
    ActivityLog.record(ActivityLog.Action.PASSWORD_RESET, user, user.username)
