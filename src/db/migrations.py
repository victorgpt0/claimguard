"""Creates the schema (schema.sql) and brings a database from an earlier version up to date.
Every caller that opens the database for writing goes through `ensure_schema`, so an old
database on Drive (e.g. the TC-028 or TC-034 run) gains new columns in place."""
from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"

# Columns added after the first databases were written: (table, column, type).
ADDED_COLUMNS = (
    ("AuditReport", "flag_sources", "TEXT"),
    ("Claim", "claim_ref", "TEXT"),
    ("User", "email", "TEXT"),  # #61: the Google account linked to this user
)


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_PATH.read_text())
    for table, column, kind in ADDED_COLUMNS:
        columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {kind}")
    # ALTER TABLE cannot add a UNIQUE column, so uniqueness is an index. Several NULLs are allowed.
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_user_email ON User(email)")
