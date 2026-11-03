"""The simulation clock. The platform holds six months of claims at once; the clock decides
which of them have been paid "so far", so the demonstration can show claims arriving over
time and audit periods closing, at a pace the presenter controls (advance, play, pause).

Simulated now = anchor_sim + speed x (real now - anchor_real). Every change re-anchors, so
pausing and resuming never jumps."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import func, insert, select, update
from sqlalchemy.engine import Engine

from .db import claims, sim_clock

DAY = 86400.0


def real_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _row(conn) -> dict:
    row = conn.execute(select(sim_clock).where(sim_clock.c.id == 1)).mappings().first()
    if row is None:
        # Start just before the first payment, paused: nothing has been paid yet.
        first = conn.execute(select(func.min(claims.c.paid_at))).scalar()
        start = (first or real_now()).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        row = {"id": 1, "anchor_sim": start, "anchor_real": real_now(), "speed": 0.0}
        conn.execute(insert(sim_clock).values(**row))
    return dict(row)


def now(eng: Engine) -> datetime:
    with eng.begin() as conn:
        row = _row(conn)
    return row["anchor_sim"] + timedelta(seconds=row["speed"] * (real_now() - row["anchor_real"]).total_seconds())


def state(eng: Engine) -> dict:
    with eng.begin() as conn:
        row = _row(conn)
    current = row["anchor_sim"] + timedelta(seconds=row["speed"] * (real_now() - row["anchor_real"]).total_seconds())
    return {"now": current.isoformat(), "playing": row["speed"] > 0, "days_per_minute": round(row["speed"] * 60 / DAY, 3)}


def _set(eng: Engine, sim: datetime, speed: float) -> dict:
    with eng.begin() as conn:
        _row(conn)
        conn.execute(update(sim_clock).where(sim_clock.c.id == 1).values(anchor_sim=sim, anchor_real=real_now(),
                                                                          speed=speed))
    return state(eng)


def _speed(eng: Engine) -> float:
    with eng.begin() as conn:
        return _row(conn)["speed"]


def set_to(eng: Engine, at: datetime) -> dict:
    return _set(eng, at, _speed(eng))


def advance(eng: Engine, days: float) -> dict:
    if days <= 0:
        raise ValueError("the clock only moves forward")
    return _set(eng, now(eng) + timedelta(days=days), _speed(eng))


def play(eng: Engine, days_per_minute: float) -> dict:
    if days_per_minute <= 0:
        raise ValueError("days_per_minute must be positive")
    return _set(eng, now(eng), days_per_minute * DAY / 60)


def pause(eng: Engine) -> dict:
    return _set(eng, now(eng), 0.0)
