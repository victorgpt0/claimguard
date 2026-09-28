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
