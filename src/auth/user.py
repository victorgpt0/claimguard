"""User entity + authenticate() (NFR-03). Retained for authentication and role-checking
even though in-app account management is out of scope (FR-11 scope note, per the scope
audit in feedback_resolution_matrix.md rows 6-7) — the two accounts (Auditor,
Administrator) are seeded directly at setup, not created through the system.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import sqlite3
from dataclasses import dataclass

_PBKDF2_ITERATIONS = 600_000  # OWASP 2023+ guidance for PBKDF2-HMAC-SHA256


@dataclass
class User:
    id: int
    username: str
    role: str  # "Auditor" | "Administrator"


def hash_password(password: str, salt: bytes | None = None) -> str:
    """PBKDF2-HMAC-SHA256 with a random per-user salt. Stored as 'salt_hex$hash_hex' so
    verification doesn't need a separate salt column. Not bcrypt/argon2 (those aren't in
    the Python standard library and this project deliberately keeps dependencies
    minimal for a proof-of-concept at this scale) — but a fixed, unsalted sha256 (the
    original placeholder here) would have been genuinely insecure; this isn't."""
    salt = salt or os.urandom(16)
    derived = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _PBKDF2_ITERATIONS)
    return f"{salt.hex()}${derived.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    salt_hex, _, expected_hex = stored_hash.partition("$")
    salt = bytes.fromhex(salt_hex)
    candidate = hash_password(password, salt=salt)
    return hmac.compare_digest(candidate, stored_hash)


def authenticate(db_path: str, username: str, password: str) -> User | None:
    """NFR-03: only authenticated users access any audit report. Returns None on any
    failure (unknown username or wrong password) — deliberately not distinguishing the
    two in the return value, so a caller can't use response shape to enumerate usernames."""
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT id, username, password_hash, role FROM User WHERE username = ?",
            (username,),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return None
    user_id, db_username, password_hash, role = row
    if not verify_password(password, password_hash):
        return None
    return User(id=user_id, username=db_username, role=role)
