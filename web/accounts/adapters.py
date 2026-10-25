"""Sign-in rules (NFR-03, #61). Nobody signs themselves up: the Administrator creates accounts.
Google proves identity only, and a Google account gets in only as the existing, active user
whose linked email equals its verified email (SOCIALACCOUNT_EMAIL_AUTHENTICATION connects it);
any other Google account is refused."""
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
