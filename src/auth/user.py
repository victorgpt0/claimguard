"""User entity, password sign-in (`authenticate`) and Google sign-in lookup (NFR-03, #55,
#61). The first Administrator is seeded at setup (src.db.seed); a Google account signs in
only as the user it has been linked to.
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


def normalise_email(email: str) -> str:
    return email.strip().lower()


def link_email(db_path: str, username: str, email: str) -> None:
    """#61: links a Google account to an existing user. Raises ValueError for an unknown
    user or an email already linked to someone else."""
    from src.db.migrations import ensure_schema

    conn = sqlite3.connect(db_path)
    try:
        ensure_schema(conn)
        try:
            cursor = conn.execute("UPDATE User SET email = ? WHERE username = ?",
                                  (normalise_email(email), username))
        except sqlite3.IntegrityError:
            raise ValueError(f"{email} is already linked to another user") from None
        if cursor.rowcount == 0:
            raise ValueError(f"no user named {username!r}")
        conn.commit()
    finally:
        conn.close()


def user_for_google_identity(db_path: str, identity: dict) -> User | None:
    """#61: Google proves who someone is; the User table decides whether they get in. Returns
    the user whose linked email matches a *verified* Google email, or None, so a Google
    account nobody has linked never reaches a report (NFR-03)."""
    email = identity.get("email")
    if not email or identity.get("email_verified") is not True:
        return None
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute("SELECT id, username, role FROM User WHERE email = ?",
                           (normalise_email(email),)).fetchone()
    except sqlite3.OperationalError:  # a database from before the email column
        return None
    finally:
        conn.close()
    return User(id=row[0], username=row[1], role=row[2]) if row else None
