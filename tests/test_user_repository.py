"""UserRepository and ActivityLog (#56): account rules the Administrator works under."""
import pytest

from src.auth.activity_log import ActivityLog
from src.auth.user import authenticate, user_for_google_identity
from src.auth.user_repository import UserRepository


@pytest.fixture
def users(tmp_db_path):
    repo = UserRepository(tmp_db_path)
    repo.create_user("admin", "admin-pass-12", "Administrator")
    repo.create_user("auditor", "auditor-pass-1", "Auditor", email="Auditor@Example.com")
    return repo


def test_create_user_rules(users):
    with pytest.raises(ValueError, match="already in use"):
        users.create_user("auditor", "another-pass-1", "Auditor")
    with pytest.raises(ValueError, match="already in use"):
        users.create_user("other", "another-pass-1", "Auditor", email="auditor@example.com")
    with pytest.raises(ValueError, match="at least 10"):
        users.create_user("other", "short", "Auditor")
    with pytest.raises(ValueError, match="role"):
        users.create_user("other", "another-pass-1", "Superuser")
    with pytest.raises(ValueError, match="username"):
        users.create_user("  ", "another-pass-1", "Auditor")
    assert users.find("auditor")["email"] == "auditor@example.com"
    assert users.find("auditor")["created_at"]


def test_deactivated_users_cannot_sign_in_by_password_or_google(users, tmp_db_path):
    auditor = users.find("auditor")
    users.set_active(auditor["id"], False, acting_user_id=users.find("admin")["id"])
    assert authenticate(tmp_db_path, "auditor", "auditor-pass-1") is None
    assert user_for_google_identity(tmp_db_path, {"email": "auditor@example.com", "email_verified": True}) is None
    users.set_active(auditor["id"], True)
    assert authenticate(tmp_db_path, "auditor", "auditor-pass-1").username == "auditor"


def test_the_last_active_administrator_and_oneself_cannot_be_deactivated(users):
    admin = users.find("admin")
    with pytest.raises(ValueError, match="your own account"):
        users.set_active(admin["id"], False, acting_user_id=admin["id"])
    with pytest.raises(ValueError, match="last active Administrator"):
        users.set_active(admin["id"], False, acting_user_id=users.find("auditor")["id"])
    second = users.create_user("admin2", "admin2-pass-1", "Administrator")
    users.set_active(admin["id"], False, acting_user_id=second)  # another admin remains
    assert not users.is_active(admin["id"])


def test_reset_and_change_password(users, tmp_db_path):
    auditor = users.find("auditor")["id"]
    users.reset_password(auditor, "reset-pass-123")
    assert authenticate(tmp_db_path, "auditor", "reset-pass-123")
    with pytest.raises(ValueError, match="incorrect"):
        users.change_password(auditor, "auditor-pass-1", "changed-pass-1")
    with pytest.raises(ValueError, match="differ"):
        users.change_password(auditor, "reset-pass-123", "reset-pass-123")
    users.change_password(auditor, "reset-pass-123", "changed-pass-1")
    assert authenticate(tmp_db_path, "auditor", "changed-pass-1")
    with pytest.raises(ValueError, match="at least 10"):
        users.reset_password(auditor, "short")


def test_link_and_unlink_email(users):
    admin = users.find("admin")["id"]
    with pytest.raises(ValueError, match="already linked"):
        users.link_email(admin, "AUDITOR@example.com")
    users.link_email(admin, " Admin@Example.com ")
    assert users.get(admin)["email"] == "admin@example.com"
    users.link_email(admin, "")
    assert users.get(admin)["email"] is None


def test_activity_log_records_and_filters(users, tmp_db_path):
    log = ActivityLog(tmp_db_path)
    log.record("sign_in", users.find("auditor")["id"], "password")
    log.record("sign_in_failed", None, "username: ghost")
    log.record("user_created", users.find("admin")["id"], "x (Auditor)")
    entries = log.entries()
    assert [e["action"] for e in entries] == ["user_created", "sign_in_failed", "sign_in"]  # newest first
    assert entries[1]["username"] is None
    assert [e["action"] for e in log.entries(username="auditor")] == ["sign_in"]
    assert [e["detail"] for e in log.entries(action="sign_in_failed")] == ["username: ghost"]
