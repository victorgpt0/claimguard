"""Audit periods (#66). Detection runs per batch, not per claim: the provider- and member-level
features (a provider's caesarean share, a member's dependants) are computed over a batch and
its history, so a claim scored alone would hide the provider-level typologies. The feed
therefore collects claims into one batch per audit period, by the date SHA paid them, and
processes the batch when the period ends.

SHA publishes no audit cycle (LN 49/2024 sets only the lodging and payment deadlines), so the
period is a setting, config.yaml `ingestion.audit_period`: monthly (the default), fortnightly
or weekly."""
from __future__ import annotations

import calendar
from datetime import date, timedelta

PERIODS = ("monthly", "fortnightly", "weekly")
# Fortnights count from a fixed Monday, so every run splits time the same way.
_FORTNIGHT_EPOCH = date(2025, 1, 6)


def period_for(day: date, kind: str) -> tuple[date, date]:
    """The audit period (first day, last day, both inclusive) that `day` falls in."""
    if kind == "monthly":
        return day.replace(day=1), day.replace(day=calendar.monthrange(day.year, day.month)[1])
    if kind == "weekly":
        start = day - timedelta(days=day.weekday())
        return start, start + timedelta(days=6)
    if kind == "fortnightly":
        start = _FORTNIGHT_EPOCH + timedelta(days=((day - _FORTNIGHT_EPOCH).days // 14) * 14)
        return start, start + timedelta(days=13)
    raise ValueError(f"audit_period must be one of {', '.join(PERIODS)}, not {kind!r}")


def label(start: date, end: date, kind: str) -> str:
    if kind == "monthly":
        return start.strftime("%B %Y")
    return f"{start:%d %b} – {end:%d %b %Y}"
