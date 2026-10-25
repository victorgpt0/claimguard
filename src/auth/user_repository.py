"""UserRepository (#56): the Administrator's account management. Accounts are deactivated,
never deleted, so every past decision stays attributable to its reviewer (FR-09). The last
active Administrator can never be deactivated, so the system cannot lock itself out.
Sign-in itself stays in src/auth/user.py.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from src.auth.user import (MIN_PASSWORD_LENGTH, ROLES, hash_password, normalise_email,
                           verify_password)
from src.db.migrations import ensure_schema


def check_password_rule(password: str) -> None:
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"The password must be at least {MIN_PASSWORD_LENGTH} characters.")


class UserRepository:
    def __init__(self, db_path: str):
        self.db_path = db_path
        with self._connect() as conn:
            ensure_schema(conn)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def list_users(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT id, username, role, email, active, created_at FROM User ORDER BY username")
            return [dict(row) for row in rows]

    def get(self, user_id: int) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT id, username, role, email, active, created_at FROM User WHERE id = ?",
                               (user_id,)).fetchone()
            return dict(row) if row else None

    def find(self, username: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT id FROM User WHERE username = ?", (username,)).fetchone()
        return self.get(row["id"]) if row else None

    def is_active(self, user_id: int) -> bool:
        user = self.get(user_id)
        return bool(user and user["active"])

    def create_user(self, username: str, password: str, role: str, email: str | None = None) -> int:
        username = username.strip()
        if not username:
            raise ValueError("A username is required.")
        if role not in ROLES:
            raise ValueError(f"The role must be one of {', '.join(ROLES)}.")
        check_password_rule(password)
        try:
            with self._connect() as conn:
                cursor = conn.execute(
                    "INSERT INTO User (username, password_hash, role, email, created_at) VALUES (?, ?, ?, ?, ?)",
                    (username, hash_password(password), role, normalise_email(email) if email else None,
                     datetime.now(timezone.utc).isoformat(timespec="seconds")))
                return cursor.lastrowid
        except sqlite3.IntegrityError:
            raise ValueError(f"The username '{username}' or that email is already in use.") from None

    def set_active(self, user_id: int, active: bool, acting_user_id: int | None = None) -> None:
        user = self.get(user_id)
        if user is None:
            raise ValueError("No such user.")
        if not active:
            if user_id == acting_user_id:
                raise ValueError("You cannot deactivate your own account.")
            if user["role"] == "Administrator" and user["active"] and self._active_admins() <= 1:
                raise ValueError("The last active Administrator cannot be deactivated.")
        with self._connect() as conn:
            conn.execute("UPDATE User SET active = ? WHERE id = ?", (int(active), user_id))

    def _active_admins(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM User WHERE role = 'Administrator' AND active = 1").fetchone()[0]

    def reset_password(self, user_id: int, new_password: str) -> None:
        check_password_rule(new_password)
        with self._connect() as conn:
            if conn.execute("UPDATE User SET password_hash = ? WHERE id = ?",
                            (hash_password(new_password), user_id)).rowcount == 0:
                raise ValueError("No such user.")

    def change_password(self, user_id: int, current_password: str, new_password: str) -> None:
        """A user's own change: the current password is required."""
        with self._connect() as conn:
            row = conn.execute("SELECT password_hash FROM User WHERE id = ?", (user_id,)).fetchone()
        if row is None or not verify_password(current_password, row["password_hash"]):
            raise ValueError("The current password is incorrect.")
        if new_password == current_password:
            raise ValueError("The new password must differ from the current one.")
        self.reset_password(user_id, new_password)

    def link_email(self, user_id: int, email: str | None) -> None:
        """#61: links a Google account to the user (None or blank unlinks it)."""
        value = normalise_email(email) if email and email.strip() else None
        try:
            with self._connect() as conn:
                if conn.execute("UPDATE User SET email = ? WHERE id = ?", (value, user_id)).rowcount == 0:
                    raise ValueError("No such user.")
        except sqlite3.IntegrityError:
            raise ValueError(f"{value} is already linked to another user.") from None
