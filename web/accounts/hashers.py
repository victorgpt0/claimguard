"""Verifies passwords imported from the Streamlit-era SQLite database (src/auth/user.py:
PBKDF2-HMAC-SHA256, 600,000 iterations, stored as "salt_hex$hash_hex"). Imported hashes are
stored as "claimguard_legacy$salt_hex$hash_hex"; Django re-hashes a password with its default
hasher at the user's next successful sign-in, so the legacy format disappears over time."""
from __future__ import annotations

import hashlib
import hmac

from django.contrib.auth.hashers import BasePasswordHasher, mask_hash

ITERATIONS = 600_000


class LegacyClaimGuardHasher(BasePasswordHasher):
    algorithm = "claimguard_legacy"

    def salt(self):
        return ""

    def encode(self, password, salt):
        raise NotImplementedError("legacy hashes are only verified, never created")

    def decode(self, encoded):
        algorithm, salt_hex, hash_hex = encoded.split("$", 2)
        assert algorithm == self.algorithm
        return {"algorithm": algorithm, "salt": salt_hex, "hash": hash_hex, "iterations": ITERATIONS}

    def verify(self, password, encoded):
        decoded = self.decode(encoded)
        derived = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(decoded["salt"]), ITERATIONS)
        return hmac.compare_digest(derived.hex(), decoded["hash"])

    def safe_summary(self, encoded):
        decoded = self.decode(encoded)
        return {"algorithm": decoded["algorithm"], "iterations": ITERATIONS,
                "salt": mask_hash(decoded["salt"]), "hash": mask_hash(decoded["hash"])}

    def must_update(self, encoded):
        return True

    def harden_runtime(self, password, encoded):
        pass


def from_legacy(stored: str) -> str:
    """'salt_hex$hash_hex' from the old User table -> a value for User.password."""
    salt_hex, _, hash_hex = stored.partition("$")
    if not (salt_hex and hash_hex):
        raise ValueError("not a legacy ClaimGuard password hash")
    return f"{LegacyClaimGuardHasher.algorithm}${salt_hex}${hash_hex}"
