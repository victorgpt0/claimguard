"""ActivityLog (#56, NFR-03): one row per security-relevant action, with who and when.
Details name the object acted on (a report id, a username, a file name) but never a
password or clinical-note text."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from src.db.migrations import ensure_schema

SIGN_IN = "sign_in"
SIGN_IN_FAILED = "sign_in_failed"
SIGN_IN_REFUSED = "google_sign_in_refused"
SIGN_OUT = "sign_out"
SESSION_EXPIRED = "session_expired"
DECISION = "decision"
USER_CREATED = "user_created"
USER_DEACTIVATED = "user_deactivated"
USER_REACTIVATED = "user_reactivated"
PASSWORD_RESET = "password_reset"
PASSWORD_CHANGED = "password_changed"
EMAIL_LINKED = "email_linked"
LOG_DOWNLOADED = "activity_log_downloaded"
ACTIONS = {SIGN_IN, SIGN_IN_FAILED, SIGN_IN_REFUSED, SIGN_OUT, SESSION_EXPIRED, DECISION, USER_CREATED,
           USER_DEACTIVATED, USER_REACTIVATED, PASSWORD_RESET, PASSWORD_CHANGED, EMAIL_LINKED,
           LOG_DOWNLOADED}


class ActivityLog:
    def __init__(self, db_path: str):
        self.db_path = db_path
        conn = sqlite3.connect(db_path)
        try:
            ensure_schema(conn)
            conn.commit()
        finally:
            conn.close()

    def record(self, action: str, user_id: int | None = None, detail: str = "") -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("INSERT INTO ActivityLog (user_id, action, detail, at) VALUES (?, ?, ?, ?)",
                         (user_id, action, detail or None, datetime.now(timezone.utc).isoformat(timespec="seconds")))
            conn.commit()
        finally:
            conn.close()

    def entries(self, username: str | None = None, action: str | None = None) -> list[dict]:
        """Newest first, optionally for one user and one action."""
        query = ("SELECT a.id, a.at, u.username, a.action, a.detail FROM ActivityLog a "
                 "LEFT JOIN User u ON u.id = a.user_id WHERE 1 = 1")
        params: list = []
        if username:
            query += " AND u.username = ?"
            params.append(username)
        if action:
            query += " AND a.action = ?"
            params.append(action)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            return [dict(row) for row in conn.execute(query + " ORDER BY a.id DESC", params)]
        finally:
            conn.close()
