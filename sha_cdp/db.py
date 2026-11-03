"""The claims platform's own database, separate from the audit system's (#65). The URL comes
from SHA_CDP_DATABASE_URL: a second PostgreSQL database in deployment, a SQLite file locally.
SQLAlchemy Core, so both work with the same code."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from sqlalchemy import (Column, Date, DateTime, Float, Integer, MetaData, String, Table, Text, create_engine)
from sqlalchemy.engine import Engine

DEFAULT_URL = f"sqlite:///{Path(__file__).resolve().parent.parent / 'data' / 'sha_cdp.sqlite'}"

metadata = MetaData()

# One row per lodged claim. The columns are the claim as the platform holds it (LN 49/2024
# reg. 58(4): member, patient, clinical details, amount) and its processing times. There are
# no fraud labels: the platform does not know which claims are fraudulent.
claims = Table(
    "claims", metadata,
    Column("seq", Integer, primary_key=True, autoincrement=True),   # the platform's own sequence
    Column("claim_id", String(64), nullable=False, unique=True),
    Column("provider_id", String(64), nullable=False),
    Column("facility_level", Integer, nullable=False),
    Column("member_id", String(64), nullable=False),
    Column("beneficiary_type", String(20), nullable=False),
    Column("patient_age", Float),
    Column("patient_sex", String(10)),
    Column("icd11_code", String(32), nullable=False),
    Column("claim_amount", Float, nullable=False),
    Column("service_date", Date, nullable=False),
    Column("discharge_date", Date),
    Column("clinical_notes", Text, nullable=False),
    Column("lodged_at", DateTime, nullable=False),
    Column("approved_at", DateTime, nullable=False),
    Column("paid_at", DateTime, nullable=False, index=True),
)

# A single row: the simulation clock (clock.py).
sim_clock = Table(
    "sim_clock", metadata,
    Column("id", Integer, primary_key=True),
    Column("anchor_sim", DateTime, nullable=False),    # simulated time at anchor_real
    Column("anchor_real", DateTime, nullable=False),   # wall-clock time (UTC) of the last change
    Column("speed", Float, nullable=False, default=0.0),   # simulated seconds per real second; 0 = paused
)


def database_url() -> str:
    return os.environ.get("SHA_CDP_DATABASE_URL") or DEFAULT_URL


@lru_cache(maxsize=4)
def engine(url: str | None = None) -> Engine:
    url = url or database_url()
    if url.startswith("postgres://"):   # the form Neon and Heroku print; SQLAlchemy wants postgresql+psycopg
        url = "postgresql+psycopg://" + url[len("postgres://"):]
    elif url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://"):]
    if url.startswith("sqlite:///"):
        Path(url[len("sqlite:///"):]).parent.mkdir(parents=True, exist_ok=True)
    eng = create_engine(url, pool_pre_ping=True)
    metadata.create_all(eng)
    return eng
