"""Seeds the two User accounts (Auditor, Administrator) directly into the database.

Per FR-11's scope note (feedback_resolution_matrix.md rows 6-7): in-app account
management is out of scope for this proof-of-concept — these two accounts are the only
ones that will ever exist, created here, not through the running application.

Usage: python -m src.db.seed --db-path /path/to/sha_fraud.sqlite
"""
import argparse
import getpass
import sqlite3
from pathlib import Path

from src.auth.user import hash_password

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def seed(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA_PATH.read_text())

    auditor_password = getpass.getpass("Set password for 'auditor' account: ")
    admin_password = getpass.getpass("Set password for 'admin' account: ")

    conn.execute(
        "INSERT OR IGNORE INTO User (username, password_hash, role) VALUES (?, ?, ?)",
        ("auditor", hash_password(auditor_password), "Auditor"),
    )
    conn.execute(
        "INSERT OR IGNORE INTO User (username, password_hash, role) VALUES (?, ?, ?)",
        ("admin", hash_password(admin_password), "Administrator"),
    )
    conn.commit()
    conn.close()
    print(f"Seeded schema + 2 user accounts at {db_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", required=True)
    args = parser.parse_args()
    seed(args.db_path)
