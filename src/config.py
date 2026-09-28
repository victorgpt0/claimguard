"""Loads config/config.yaml (paths, thresholds, model names) and .env (secrets) into a
single place other modules import from, rather than each module re-implementing its own
YAML/dotenv loading.
"""
from __future__ import annotations

from pathlib import Path

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: str | Path | None = None) -> dict:
    """Loads config.yaml (defaults to config/config.yaml at the repo root) and .env
    (defaults to repo-root .env, silently skipped if absent — e.g. on Colab, where
    google.colab.userdata is used instead per implementation_roadmap.md)."""
    load_dotenv(REPO_ROOT / ".env")
    config_path = Path(config_path) if config_path else REPO_ROOT / "config" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"{config_path} not found — copy config/config.example.yaml to config/config.yaml first."
        )
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))
