import sqlite3
from pathlib import Path

from src.auth.user import authenticate, hash_password, verify_password

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "src" / "db" / "schema.sql"


def test_hash_password_produces_different_hashes_for_same_password():
    h1 = hash_password("hunter2")
    h2 = hash_password("hunter2")
    assert h1 != h2  # different random salts


def test_verify_password_round_trip():
    hashed = hash_password("correct horse battery staple")
    assert verify_password("correct horse battery staple", hashed)
    assert not verify_password("wrong password", hashed)


def _seeded_db(tmp_db_path: str) -> str:
    conn = sqlite3.connect(tmp_db_path)
    conn.executescript(SCHEMA_PATH.read_text())
    conn.execute(
        "INSERT INTO User (username, password_hash, role) VALUES (?, ?, ?)",
        ("auditor", hash_password("auditpass"), "Auditor"),
    )
    conn.commit()
    conn.close()
    return tmp_db_path


def test_authenticate_success(tmp_db_path):
    db_path = _seeded_db(tmp_db_path)
    user = authenticate(db_path, "auditor", "auditpass")
    assert user is not None
    assert user.username == "auditor"
    assert user.role == "Auditor"


def test_authenticate_wrong_password(tmp_db_path):
    db_path = _seeded_db(tmp_db_path)
    assert authenticate(db_path, "auditor", "wrongpass") is None


def test_authenticate_unknown_user(tmp_db_path):
    db_path = _seeded_db(tmp_db_path)
    assert authenticate(db_path, "nobody", "whatever") is None


def test_seed_user_is_idempotent_and_never_resets_a_password(tmp_db_path):
    import pytest
    from src.db.seed import seed_user, user_exists
    assert seed_user(tmp_db_path, "admin", "first-password", "Administrator")
    assert not seed_user(tmp_db_path, "admin", "second-password", "Administrator")
    assert user_exists(tmp_db_path, "admin")
    assert authenticate(tmp_db_path, "admin", "first-password").role == "Administrator"
    assert authenticate(tmp_db_path, "admin", "second-password") is None
    with pytest.raises(ValueError):
        seed_user(tmp_db_path, "auditor", "short", "Auditor")


def test_google_identity_signs_in_only_a_linked_verified_email(tmp_db_path):
    import pytest
    from src.auth.user import link_email, user_for_google_identity
    from src.db.seed import seed_user
    seed_user(tmp_db_path, "auditor", "auditor-pass-1", "Auditor")
    seed_user(tmp_db_path, "admin", "admin-pass-12", "Administrator")
    link_email(tmp_db_path, "auditor", " Auditor@Example.com ")
    user = user_for_google_identity(tmp_db_path, {"email": "auditor@example.com", "email_verified": True})
    assert (user.username, user.role) == ("auditor", "Auditor")
    assert user_for_google_identity(tmp_db_path, {"email": "AUDITOR@example.com", "email_verified": True})
    # Unverified, unlinked or missing emails never sign in.
    assert user_for_google_identity(tmp_db_path, {"email": "auditor@example.com", "email_verified": False}) is None
    assert user_for_google_identity(tmp_db_path, {"email": "auditor@example.com"}) is None
    assert user_for_google_identity(tmp_db_path, {"email": "stranger@gmail.com", "email_verified": True}) is None
    assert user_for_google_identity(tmp_db_path, {}) is None
    with pytest.raises(ValueError, match="already linked"):
        link_email(tmp_db_path, "admin", "auditor@example.com")
    with pytest.raises(ValueError, match="no user"):
        link_email(tmp_db_path, "nobody", "x@example.com")


def test_old_database_gains_the_email_column(tmp_db_path):
    """A database from before #61 (no User.email) is migrated in place, keeping its users."""
    conn = sqlite3.connect(tmp_db_path)
    conn.execute("CREATE TABLE User (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT NOT NULL UNIQUE, "
                 "password_hash TEXT NOT NULL, role TEXT NOT NULL)")
    conn.execute("INSERT INTO User (username, password_hash, role) VALUES (?, ?, ?)",
                 ("auditor", hash_password("auditpass"), "Auditor"))
    conn.commit()
    conn.close()
    from src.reporting.repository import AuditReportRepository
    AuditReportRepository(tmp_db_path)
    with sqlite3.connect(tmp_db_path) as conn:
        assert "email" in {row[1] for row in conn.execute("PRAGMA table_info(User)")}
    assert authenticate(tmp_db_path, "auditor", "auditpass").role == "Auditor"
