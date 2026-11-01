#!/usr/bin/env python
"""Django's command-line utility for the ClaimGuard web application."""
import os
import sys
from pathlib import Path


def main():
    # The ML library (src/) lives at the repository root, one level up.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "claimguard_web.settings")
    from django.core.management import execute_from_command_line
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
