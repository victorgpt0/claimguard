"""The ingestion pipeline (#66): pull newly paid claims from the claims platform, check them,
collect them into the open batch of their audit period, and hand each batch to the detection
pipeline once its period has ended.

    extract   GET /api/v1/claims/paid after the cursor, page by page (incremental, idempotent)
    check     the upload checks (FR-15, audit/validation.py), applied per claim: a bad claim is
              quarantined with its problems, the rest of the page is stored
    dedupe    a claim id already stored is counted and skipped, so a repeated pull is harmless
    load      audit.Claim rows in the OPEN feed batch of the claim's audit period
    close     when the platform's clock has passed a period's last day and every claim paid by
              then has been pulled, the batch is frozen (its CSV written, as an upload's is) and,
              if ingestion.auto_process is on, a job is queued for it

Each page is stored in one transaction together with the cursor that follows it, so a pull
that fails part-way keeps what it stored and the next pull resumes from there."""
from __future__ import annotations

import hashlib
import io
import logging
from datetime import date, datetime, timedelta, timezone as dt_timezone

import pandas as pd
from django.db import transaction
from django.db.models import Q
from django.utils import timezone

from audit import services
from audit.models import Batch, Claim, Job
from audit.validation import UPLOAD_COLUMNS, validate

from . import periods
from .client import PlatformClient, PlatformError
from .models import FeedState, IngestionRun, RejectedClaim

logger = logging.getLogger(__name__)
PAGE_SIZE = 500
MAX_PAGES = 40          # 20,000 claims per pull at most; the rest come with the next pull
STALE_AFTER = timedelta(minutes=10)
STATE_FIELDS = ["after_paid_at", "after_seq", "platform_as_of", "last_attempt_at", "last_success_at", "last_error"]
DEFAULTS = {"audit_period": "monthly", "auto_process": True, "queue": "cpu"}


def feed_settings() -> dict:
    """config.yaml `ingestion:` over DEFAULTS."""
    try:
        configured = services.config().get("ingestion") or {}
    except services.ConfigMissing:
        configured = {}
    out = {**DEFAULTS, **configured}
    if out["audit_period"] not in periods.PERIODS:
        raise ValueError(f"ingestion.audit_period must be one of {', '.join(periods.PERIODS)}")
    if out["queue"] not in Job.Queue.values:
        raise ValueError(f"ingestion.queue must be one of {', '.join(Job.Queue.values)}")
    return out


def _aware(value: str) -> datetime:
    """The platform's times are naive UTC."""
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt_timezone.utc)


def _text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _number(value):
    return None if value in (None, "") else float(value)


def _open_batch(start: date, end: date, kind: str) -> Batch:
    batch = (Batch.objects.select_for_update()
             .filter(source=Batch.Source.FEED, period_start=start, period_end=end, status=Batch.Status.OPEN).first())
    if batch is None:
        batch = Batch.objects.create(name=f"Claims feed · {periods.label(start, end, kind)}", source=Batch.Source.FEED,
                                     period_start=start, period_end=end, status=Batch.Status.OPEN)
    return batch


def _store_page(claims: list[dict], run: IngestionRun, kind: str, known_codes: set[str], heldout: set[str]) -> None:
    refs = [str(c.get("claim_id") or "").strip() for c in claims]
    existing = set(Claim.objects.filter(claim_ref__in=[r for r in refs if r]).values_list("claim_ref", flat=True))
    fresh = [c for c, ref in zip(claims, refs) if ref not in existing]
    run.fetched += len(claims)
    run.duplicates += len(claims) - len(fresh)
    if not fresh:
        return
    frame = pd.DataFrame([{col: _text(c.get(col)) for col in UPLOAD_COLUMNS} for c in fresh], columns=UPLOAD_COLUMNS)
    problems: dict[int, list[dict]] = {}
    for problem in validate(frame, known_codes, set(), heldout):
        if problem.row is None:   # a whole-page problem: refuse every claim on the page
            for index in range(len(fresh)):
                problems.setdefault(index, []).append({"column": problem.column, "message": problem.message})
        else:
            problems.setdefault(problem.row - 2, []).append({"column": problem.column, "message": problem.message})

    rows = []
    for index, (claim, row) in enumerate(zip(fresh, frame.to_dict(orient="records"))):
        paid_at = _aware(claim["paid_at"]) if claim.get("paid_at") else None
        if index in problems or paid_at is None:
            RejectedClaim.objects.create(
                run=run, claim_ref=row["claim_id"][:64], paid_at=paid_at,
                problems=problems.get(index) or [{"column": "paid_at", "message": "missing"}],
                record={k: v for k, v in claim.items() if k != "clinical_notes"}
                | {"clinical_notes_words": len(str(claim.get("clinical_notes") or "").split())})
            run.rejected += 1
            continue
        start, end = periods.period_for(paid_at.date(), kind)
        batch = _open_batch(start, end, kind)
        rows.append(Claim(
            batch=batch, claim_ref=row["claim_id"].strip(), provider_id=row["provider_id"].strip(),
            patient_ref=row["member_id"].strip(), beneficiary_type=row["beneficiary_type"].strip(),
            patient_age=_number(row["patient_age"]), patient_sex=row["patient_sex"].strip(),
            facility_level=int(float(row["facility_level"])), icd11_code=row["icd11_code"].strip(),
            claim_amount=row["claim_amount"].strip(), service_date=pd.Timestamp(row["service_date"]).date(),
            discharge_date=pd.Timestamp(row["discharge_date"]).date() if row["discharge_date"].strip() else None,
            clinical_notes=row["clinical_notes"], paid_at=paid_at))
    Claim.objects.bulk_create(rows)
    run.accepted += len(rows)


def batch_csv(batch: Batch) -> str:
    """The batch's claims in the upload format, which the processing job reads (audit/tasks.py)."""
    records = [{
        "claim_id": c.claim_ref, "provider_id": c.provider_id, "member_id": c.patient_ref,
        "beneficiary_type": c.beneficiary_type, "patient_age": "" if c.patient_age is None else c.patient_age,
        "patient_sex": c.patient_sex, "facility_level": c.facility_level, "icd11_code": c.icd11_code,
        "claim_amount": c.claim_amount, "service_date": c.service_date.isoformat(),
        "discharge_date": c.discharge_date.isoformat() if c.discharge_date else "", "clinical_notes": c.clinical_notes,
    } for c in batch.claims.order_by("paid_at", "claim_ref")]
    buffer = io.StringIO()
    pd.DataFrame(records, columns=UPLOAD_COLUMNS).to_csv(buffer, index=False)
    return buffer.getvalue()


def close_batch(batch: Batch, queue: str | None, user=None) -> Job | None:
    """Freeze an open feed batch and, given a queue, queue its processing job."""
    from audit.tasks import process_batch
    csv = batch_csv(batch)
    batch.source_csv, batch.source_sha256 = csv, hashlib.sha256(csv.encode("utf-8")).hexdigest()
    batch.claims_count, batch.status, batch.closed_at = batch.claims.count(), Batch.Status.UPLOADED, timezone.now()
    batch.save(update_fields=["source_csv", "source_sha256", "claims_count", "status", "closed_at"])
    if not queue:
        return None
    job = Job.objects.create(batch=batch, queue=queue, created_by=user)
    transaction.on_commit(lambda: process_batch.apply_async(args=[job.pk], queue=queue))
    return job


def close_due(as_of: datetime, settings_: dict) -> list[int]:
    """Close every open feed batch whose period ended before the platform's clock."""
    closed = []
    queue = settings_["queue"] if settings_["auto_process"] else None
    for batch in Batch.objects.filter(source=Batch.Source.FEED, status=Batch.Status.OPEN,
                                      period_end__lt=as_of.date()).order_by("period_start"):
        with transaction.atomic():
            batch = Batch.objects.select_for_update().get(pk=batch.pk)
            if batch.status == Batch.Status.OPEN:
                close_batch(batch, queue)
                closed.append(batch.pk)
    return closed


class Busy(Exception):
    """Another pull is running."""


def sync(user=None, client: PlatformClient | None = None, scheduled: bool = False) -> IngestionRun | None:
    """One pull. Returns None when a scheduled pull is skipped because the feed is paused;
    raises Busy when another pull is running."""
    state = FeedState.get()
    if scheduled and state.paused:
        return None
    now = timezone.now()
    claimed = FeedState.objects.filter(pk=state.pk).filter(
        Q(running_since__isnull=True) | Q(running_since__lt=now - STALE_AFTER)).update(running_since=now)
    if not claimed:
        raise Busy("a pull from the claims platform is already running")
    try:
        return _sync(state, user, client or PlatformClient())
    finally:
        FeedState.objects.filter(pk=state.pk).update(running_since=None)


def _sync(state: FeedState, user, client: PlatformClient) -> IngestionRun:
    run = IngestionRun.objects.create(triggered_by=user, cursor_from=_cursor(state))
    state.last_attempt_at = timezone.now()
    as_of = None
    complete = False
    try:
        settings_ = feed_settings()
        kind = settings_["audit_period"]
        known_codes, heldout = services.known_codes(), services.heldout_ids()
        for _ in range(MAX_PAGES):
            page = client.paid_claims(state.after_paid_at.astimezone(dt_timezone.utc).replace(tzinfo=None).isoformat()
                                      if state.after_paid_at else None, state.after_seq, PAGE_SIZE)
            as_of = _aware(page.as_of)
            with transaction.atomic():
                _store_page(page.claims, run, kind, known_codes, heldout)
                if page.next:
                    state.after_paid_at = _aware(page.next["after_paid_at"])
                    state.after_seq = page.next["after_seq"]
                state.platform_as_of = as_of
                state.save(update_fields=STATE_FIELDS)
                run.save()
            if not page.has_more:
                complete = True
                break
        # A period is closed only once everything paid up to the clock has been pulled, so no
        # claim of a closed period can still be on its way.
        if complete:
            run.batches_closed = close_due(as_of, settings_)
        run.status = IngestionRun.Status.SUCCEEDED
        state.last_success_at, state.last_error = timezone.now(), ""
    except (PlatformError, services.ConfigMissing, FileNotFoundError, ValueError) as exc:
        logger.warning("Claims feed pull failed: %s", exc)
        run.status = IngestionRun.Status.PARTIAL if run.accepted or run.duplicates or run.rejected \
            else IngestionRun.Status.FAILED
        run.error = state.last_error = str(exc)
    run.platform_as_of = as_of
    run.cursor_to = _cursor(state)
    run.finished_at = timezone.now()
    run.save()
    state.save(update_fields=STATE_FIELDS)
    return run


def _cursor(state: FeedState) -> str:
    if not state.after_paid_at:
        return "start"
    return f"{state.after_paid_at:%Y-%m-%d %H:%M:%S} #{state.after_seq}"
