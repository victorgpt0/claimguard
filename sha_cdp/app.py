"""The platform's API (#65; IR-09 on the audit side). Two credentials, both from the
environment:

    SHA_CDP_API_KEY      read-only: the audit system's key, for the paid-claims export and stats
    SHA_CDP_ADMIN_TOKEN  the simulation's operator: moves the clock

Only paid claims are exported, and only those paid by the clock's current time: a
post-payment audit sees a claim once SHA has paid it, never before. Paging uses a cursor of
(paid_at, seq), the last claim the caller has, so a repeated pull returns nothing twice and
a pull that fails part-way is simply repeated.

    uvicorn sha_cdp.app:app --port 8100
"""
from __future__ import annotations

import hmac
import os
from datetime import datetime

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import and_, func, or_, select

from . import clock, db

app = FastAPI(title="SHA Centralized Digital Platform (simulated)", version="1.0",
              description="A stand-in for SHA's claims platform: paid claims for post-payment audit.")
MAX_PAGE = 500


def _engine():
    return db.engine()


def _check(expected_env: str, authorization: str | None) -> None:
    expected = os.environ.get(expected_env, "")
    supplied = (authorization or "").removeprefix("Bearer ").strip()
    if not expected or not hmac.compare_digest(supplied.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="invalid or missing token",
                            headers={"WWW-Authenticate": "Bearer"})


def reader(authorization: str | None = Header(default=None)) -> None:
    _check("SHA_CDP_API_KEY", authorization)


def operator(authorization: str | None = Header(default=None)) -> None:
    _check("SHA_CDP_ADMIN_TOKEN", authorization)


def _claim(row) -> dict:
    out = dict(row)
    for key in ("service_date", "discharge_date", "lodged_at", "approved_at", "paid_at"):
        out[key] = out[key].isoformat() if out[key] is not None else None
    return out


@app.get("/health")
def health():
    eng = _engine()
    return {"status": "ok", "clock": clock.state(eng)}


@app.get("/api/v1/claims/paid", dependencies=[Depends(reader)])
def paid_claims(after_paid_at: datetime | None = None, after_seq: int = 0,
                limit: int = Query(default=200, ge=1, le=MAX_PAGE)):
    eng = _engine()
    as_of = clock.now(eng)
    c = db.claims.c
    query = select(db.claims).where(c.paid_at <= as_of)
    if after_paid_at is not None:
        after_paid_at = after_paid_at.replace(tzinfo=None)
        query = query.where(or_(c.paid_at > after_paid_at, and_(c.paid_at == after_paid_at, c.seq > after_seq)))
    with eng.connect() as conn:
        rows = conn.execute(query.order_by(c.paid_at, c.seq).limit(limit + 1)).mappings().all()
    page, has_more = rows[:limit], len(rows) > limit
    last = page[-1] if page else None
    return {"as_of": as_of.isoformat(), "claims": [_claim(r) for r in page], "has_more": has_more,
            "next": {"after_paid_at": last["paid_at"].isoformat(), "after_seq": last["seq"]} if last else None}


@app.get("/api/v1/stats", dependencies=[Depends(reader)])
def stats():
    """What the platform holds as of the clock: for the audit side's pipeline page."""
    eng = _engine()
    as_of = clock.now(eng)
    c = db.claims.c
    with eng.connect() as conn:
        total, lodged, approved, paid, paid_amount, first, last = conn.execute(select(
            func.count(), func.count().filter(c.lodged_at <= as_of), func.count().filter(c.approved_at <= as_of),
            func.count().filter(c.paid_at <= as_of), func.coalesce(func.sum(c.claim_amount).filter(c.paid_at <= as_of), 0),
            func.min(c.paid_at), func.max(c.paid_at))).one()
        upcoming = conn.execute(select(func.min(c.paid_at)).where(c.paid_at > as_of)).scalar()
    return {"as_of": as_of.isoformat(), "clock": clock.state(eng), "claims_held": total, "lodged": lodged,
            "approved": approved, "paid": paid, "paid_amount": float(paid_amount),
            "awaiting_payment": lodged - paid, "next_payment_at": upcoming.isoformat() if upcoming else None,
            "payments_from": first.isoformat() if first else None, "payments_to": last.isoformat() if last else None}


class ClockChange(BaseModel):
    action: str                          # advance | play | pause | set
    days: float | None = None            # advance
    days_per_minute: float | None = None  # play
    at: datetime | None = None           # set


@app.get("/api/v1/clock", dependencies=[Depends(reader)])
def get_clock():
    return clock.state(_engine())


@app.post("/api/v1/clock", dependencies=[Depends(operator)])
def change_clock(change: ClockChange):
    eng = _engine()
    try:
        if change.action == "advance" and change.days:
            return clock.advance(eng, change.days)
        if change.action == "play" and change.days_per_minute:
            return clock.play(eng, change.days_per_minute)
        if change.action == "pause":
            return clock.pause(eng)
        if change.action == "set" and change.at:
            return clock.set_to(eng, change.at.replace(tzinfo=None))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    raise HTTPException(status_code=422, detail="give action advance+days, play+days_per_minute, pause, or set+at")
