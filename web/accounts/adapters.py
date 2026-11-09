"""Sign-in rules (NFR-03, #61). Nobody signs themselves up: the Administrator creates accounts.
Google proves identity only. A Google account gets in as an existing, active user in one of
two ways: the user connected it themselves while signed in with their password (Google
sign-in page, allauth's "connect" process), or its verified email equals the email the
Administrator linked to the user (SOCIALACCOUNT_EMAIL_AUTHENTICATION connects it on first
use). Any other Google account is refused."""
from __future__ import annotations

from allauth.account.adapter import DefaultAccountAdapter
from allauth.socialaccount.adapter import DefaultSocialAccountAdapter

from activity.models import ActivityLog


class NoSignupAccountAdapter(DefaultAccountAdapter):
    def is_open_for_signup(self, request):
        return False


class LinkedAccountsOnlyAdapter(DefaultSocialAccountAdapter):
    def is_open_for_signup(self, request, sociallogin):
        # Reached only when no existing user matched the verified email: refuse, and log it.
        email = (sociallogin.user.email or "").lower() or "a Google account"
        ActivityLog.record(ActivityLog.Action.SIGN_IN_REFUSED, detail=email)
        return False
