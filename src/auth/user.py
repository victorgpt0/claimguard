"""User entity + authenticate() (NFR-03). Retained for authentication and role-checking
even though in-app account management is out of scope (FR-11 scope note, per the scope
audit in feedback_resolution_matrix.md rows 6-7) — the two accounts (Auditor,
Administrator) are seeded directly at setup, not created through the system.
"""
import hashlib
from dataclasses import dataclass


@dataclass
class User:
    id: int
    username: str
    password_hash: str
    role: str  # "Auditor" | "Administrator"


def hash_password(password: str) -> str:
    # Placeholder only — replace with a real KDF (bcrypt/argon2) before this touches
    # anything beyond local dev; sha256 alone is not adequate password storage.
    return hashlib.sha256(password.encode()).hexdigest()


def authenticate(db_path: str, username: str, password: str) -> User | None:
    """NFR-03: only authenticated users access any audit report."""
    raise NotImplementedError
