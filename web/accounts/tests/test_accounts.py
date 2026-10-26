"""Accounts (#55, #56, #61 on Django): roles, the last-Administrator rule, no deletion, sign-in
pages, the activity log, legacy password import and Google sign-in restricted to linked users."""
import hashlib
import os

import pytest
from django.core.exceptions import ValidationError
from django.test import Client
from django.urls import reverse

from accounts.hashers import from_legacy
from accounts.models import User
from activity.models import ActivityLog


def test_roles_set_admin_site_access(make_user):
    auditor = make_user()
    admin = make_user("admin", role="Administrator", password="admin-pass-12")
    assert not (auditor.is_staff or auditor.is_superuser) and auditor.is_auditor
    assert admin.is_staff and admin.is_superuser and admin.is_administrator


def test_last_active_administrator_is_protected(make_user):
    admin = make_user("admin", role="Administrator")
    admin.is_active = False
    with pytest.raises(ValidationError):
        admin.save()
    admin.refresh_from_db()
    admin.role = User.Role.AUDITOR
    with pytest.raises(ValidationError):
        admin.save()
    make_user("admin2", role="Administrator")
    admin.refresh_from_db()
    admin.is_active = False
    admin.save()  # another active Administrator remains


def test_accounts_are_never_deleted(make_user):
    with pytest.raises(ValidationError):
        make_user().delete()


def test_linked_emails_are_unique_and_lowercase(make_user):
    user = make_user(email=" Auditor@Example.com ")
    assert user.email == "auditor@example.com"
    make_user("other")  # blank emails do not clash
    with pytest.raises(Exception):
        make_user("third", email="AUDITOR@example.com")


def test_pages_need_sign_in_and_sign_in_is_logged(make_user):
    make_user()
    client = Client()
    response = client.get(reverse("audit:overview"))
    assert response.status_code == 302 and reverse("account_login") in response["Location"]
    client.post(reverse("account_login"), {"login": "auditor", "password": "wrong-password"})
    response = client.post(reverse("account_login"), {"login": "auditor", "password": "auditor-pass-1"})
    assert response.status_code == 302
    assert "Sign out" in client.get(reverse("audit:overview")).content.decode()
    client.post(reverse("account_logout"))
    actions = list(ActivityLog.objects.order_by("id").values_list("action", "detail"))
    assert actions == [("sign_in_failed", "username: auditor"), ("sign_in", "password"), ("sign_out", "")]
    assert all("pass" not in detail.replace("password", "") for _, detail in actions)


def test_no_self_sign_up(make_user):
    response = Client().post(reverse("account_signup"), {"username": "intruder", "password1": "x" * 12,
                                                         "password2": "x" * 12})
    assert not User.objects.filter(username="intruder").exists()
    assert response.status_code in (200, 302, 403)


def test_auditors_cannot_open_the_admin_site(make_user):
    make_user()
    client = Client()
    client.login(username="auditor", password="auditor-pass-1")
    assert client.get("/admin/").status_code == 302  # redirected to the admin login


def test_imported_streamlit_password_verifies_and_is_upgraded(db):
    salt = os.urandom(16)
    old = f"{salt.hex()}${hashlib.pbkdf2_hmac('sha256', b'old-password-1', salt, 600_000).hex()}"
    user = User(username="legacy", role="Auditor", password=from_legacy(old))
    user.save()
    assert not user.check_password("wrong-password")
    assert user.check_password("old-password-1")
    user.refresh_from_db()
    assert user.password.startswith("pbkdf2_sha256$")  # re-hashed with Django's hasher


def test_google_sign_in_only_for_a_linked_user(rf, make_user, settings):
    from allauth.socialaccount.models import SocialAccount, SocialLogin
    from accounts.adapters import LinkedAccountsOnlyAdapter
    stranger = SocialLogin(user=User(email="stranger@gmail.com"), account=SocialAccount(provider="google"))
    assert LinkedAccountsOnlyAdapter().is_open_for_signup(rf.get("/"), stranger) is False
    entry = ActivityLog.objects.get()
    assert (entry.action, entry.detail, entry.user) == ("google_sign_in_refused", "stranger@gmail.com", None)
    assert settings.SOCIALACCOUNT_EMAIL_AUTHENTICATION and settings.SOCIALACCOUNT_EMAIL_AUTHENTICATION_AUTO_CONNECT


def test_an_idle_session_expires_after_30_minutes(make_user):
    make_user()
    client = Client()
    client.post(reverse("account_login"), {"login": "auditor", "password": "auditor-pass-1"})
    assert client.session.get_expiry_age() == 30 * 60
    session = client.session
    session.set_expiry(-1)   # as if 30 idle minutes had passed
    session.save()
    response = client.get(reverse("audit:overview"))
    assert response.status_code == 302 and reverse("account_login") in response["Location"]


def test_administrator_account_changes_are_logged(make_user):
    admin = make_user("admin", role="Administrator", password="admin-pass-12")
    client = Client()
    client.force_login(admin)
    client.post(reverse("admin:accounts_user_add"), {
        "username": "newauditor", "role": "Auditor", "email": "New@Example.com",
        "password1": "a-long-pass-phrase-1", "password2": "a-long-pass-phrase-1"})
    user = User.objects.get(username="newauditor")
    assert user.email == "new@example.com" and user.check_password("a-long-pass-phrase-1")
    client.post(reverse("admin:accounts_user_change", args=[user.pk]), {
        "username": "newauditor", "role": "Auditor", "email": "new@example.com", "is_active": ""})
    user.refresh_from_db()
    assert not user.is_active
    actions = list(ActivityLog.objects.exclude(action="sign_in").values_list("action", "detail"))
    assert ("user_created", "newauditor (Auditor)") in actions and ("user_deactivated", "newauditor") in actions


def test_activity_log_is_read_only_and_exportable(make_user):
    admin = make_user("admin", role="Administrator", password="admin-pass-12")
    ActivityLog.record(ActivityLog.Action.DECISION, admin, "report #1: approved")
    entry = ActivityLog.objects.get()
    client = Client()
    client.force_login(admin)
    assert client.get(reverse("admin:activity_activitylog_add")).status_code == 403
    assert client.get(reverse("admin:activity_activitylog_delete", args=[entry.pk])).status_code == 403
    assert client.post(reverse("admin:activity_activitylog_change", args=[entry.pk]),
                       {"action": "sign_in", "detail": "edited"}).status_code == 403
    entry.refresh_from_db()
    assert entry.detail == "report #1: approved"
    response = client.post(reverse("admin:activity_activitylog_changelist"),
                           {"action": "download_csv", "_selected_action": [entry.pk]})
    assert response["Content-Type"] == "text/csv" and "report #1: approved" in response.content.decode()
    assert ActivityLog.objects.filter(action="download").exists()
