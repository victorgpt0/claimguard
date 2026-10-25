"""Seeds the first Administrator account (NFR-03) into the database, creating the schema
if needed. Auditor accounts are created by the Administrator in the dashboard (#56);
`--auditor` also seeds one Auditor, for a demonstration before that exists.

Idempotent: an existing username is left unchanged, so re-running never resets a password.

`--link-email USERNAME EMAIL` links a Google account to an existing user (#61), so that user
can also use "Sign in with Google". It never changes the password.

Usage: python -m src.db.seed --db-path /path/to/sha_fraud.sqlite [--admin admin] [--auditor auditor]
       python -m src.db.seed --db-path /path/to/sha_fraud.sqlite --link-email admin someone@gmail.com
"""
import argparse
import getpass
import sqlite3
from src.auth.user import hash_password, link_email
from src.db.migrations import ensure_schema

MIN_PASSWORD_LENGTH = 10


def seed_user(db_path: str, username: str, password: str, role: str) -> bool:
    """Creates the account unless the username exists; returns whether it was created."""
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"password must be at least {MIN_PASSWORD_LENGTH} characters")
    conn = sqlite3.connect(db_path)
    try:
        ensure_schema(conn)
        cursor = conn.execute(
            "INSERT OR IGNORE INTO User (username, password_hash, role) VALUES (?, ?, ?)",
            (username, hash_password(password), role),
        )
        conn.commit()
        return cursor.rowcount == 1
    finally:
        conn.close()


def user_exists(db_path: str, username: str) -> bool:
    conn = sqlite3.connect(db_path)
    try:
        ensure_schema(conn)
        return conn.execute("SELECT 1 FROM User WHERE username = ?", (username,)).fetchone() is not None
    finally:
        conn.close()


def _ask_password(username: str) -> str:
    while True:
        password = getpass.getpass(f"Set password for '{username}' (at least {MIN_PASSWORD_LENGTH} characters): ")
        if len(password) < MIN_PASSWORD_LENGTH:
            print("Too short.")
        elif getpass.getpass("Repeat it: ") != password:
            print("The two entries differ.")
        else:
            return password


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--admin", default="admin", help="Administrator username")
    parser.add_argument("--auditor", help="also seed an Auditor with this username")
    parser.add_argument("--link-email", nargs=2, metavar=("USERNAME", "EMAIL"),
                        help="link a Google account to an existing user, then stop")
    args = parser.parse_args()
    if args.link_email:
        username, email = args.link_email
        link_email(args.db_path, username, email)
        print(f"Linked {email.strip().lower()} to '{username}'")
        return
    accounts = [(args.admin, "Administrator")] + ([(args.auditor, "Auditor")] if args.auditor else [])
    for username, role in accounts:
        if user_exists(args.db_path, username):
            print(f"'{username}' already exists in {args.db_path}; left unchanged")
        elif seed_user(args.db_path, username, _ask_password(username), role):
            print(f"Created {role} '{username}' in {args.db_path}")


if __name__ == "__main__":
    main()
