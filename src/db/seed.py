"""Seeds the first Administrator account (NFR-03) into the database, creating the schema
if needed. Other accounts are created by the Administrator in the dashboard's Users tab
(#56); `--auditor` also seeds one Auditor, for convenience.

Idempotent: an existing username is left unchanged, so re-running never resets a password.
`--link-email USERNAME EMAIL` links a Google account to an existing user (#61), so that user
can also use "Sign in with Google". It never changes the password.

Usage: python -m src.db.seed --db-path /path/to/sha_fraud.sqlite [--admin admin] [--auditor auditor]
       python -m src.db.seed --db-path /path/to/sha_fraud.sqlite --link-email admin someone@gmail.com
"""
import argparse
import getpass

from src.auth.user import MIN_PASSWORD_LENGTH
from src.auth.user_repository import UserRepository


def seed_user(db_path: str, username: str, password: str, role: str) -> bool:
    """Creates the account unless the username exists; returns whether it was created."""
    users = UserRepository(db_path)
    if users.find(username):
        return False
    users.create_user(username, password, role)
    return True


def user_exists(db_path: str, username: str) -> bool:
    return UserRepository(db_path).find(username) is not None


def link_email(db_path: str, username: str, email: str) -> None:
    users = UserRepository(db_path)
    user = users.find(username)
    if user is None:
        raise ValueError(f"no user named {username!r}")
    users.link_email(user["id"], email)


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
