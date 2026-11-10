"""The Data pipeline page (#66): where claims come from and where they are now. The flow from
the claims platform to the review queue with live counts, the audit periods (open batches
collecting claims, closed ones being processed), every pull, and the quarantined claims.
The page is the Administrator's alone (separation of duties: whoever decides what is audited
does not judge the results): pull now, close a period early, pause the feed, move the
platform's clock. Fraud Auditors never see it."""
from __future__ import annotations

from datetime import datetime

from django.contrib import messages
from django.db import transaction
from django.db.models import Count, Q
from django.http import HttpResponseNotAllowed
from django.shortcuts import get_object_or_404, redirect, render

from accounts.roles import administrator_required
from activity.models import ActivityLog
from audit.models import AuditReport, Batch, Claim, Job

from . import periods, sync
from .client import PlatformClient, PlatformError
from .models import FeedState, IngestionRun, RejectedClaim

CLOCK_ACTIONS = {
    "advance_week": ({"action": "advance", "days": 7}, "moved the platform clock on a week"),
    "advance_month": ({"action": "advance", "days": 30}, "moved the platform clock on 30 days"),
    "play": ({"action": "play", "days_per_minute": 15}, "started the platform clock at 15 days a minute"),
    "pause": ({"action": "pause"}, "paused the platform clock"),
}


def _when(value: str | None):
    return datetime.fromisoformat(value) if value else None


def _context() -> dict:
    state = FeedState.get()
    try:
        settings_, settings_error = sync.feed_settings(), ""
    except ValueError as exc:
        settings_, settings_error = sync.DEFAULTS, str(exc)
    try:
        platform, platform_error = PlatformClient().stats(), ""
        for key in ("as_of", "next_payment_at", "payments_from", "payments_to"):
            platform[key] = _when(platform.get(key))
    except PlatformError as exc:
        platform, platform_error = None, str(exc)

    feed = Batch.objects.filter(source=Batch.Source.FEED)
    periods_ = list(feed.annotate(n_claims=Count("claims", distinct=True),
                                  n_reports=Count("claims__reports", distinct=True),
                                  n_pending=Count("claims__reports", distinct=True,
                                                  filter=Q(claims__reports__status=AuditReport.Status.PENDING)))
                    .order_by("-period_start", "-id")[:24])
    jobs = {}
    for job in Job.objects.filter(batch__in=periods_).order_by("batch_id", "-created_at"):
        jobs.setdefault(job.batch_id, job)
    for b in periods_:
        b.job = jobs.get(b.pk)
        b.flagged = ((b.run_log or {}).get("summary") or {}).get("flagged_stage1")
        b.label = periods.label(b.period_start, b.period_end, settings_["audit_period"]) if b.period_start else b.name
    open_ = [b for b in periods_ if b.status == Batch.Status.OPEN]
    in_detection = [b for b in periods_ if b.status in (Batch.Status.UPLOADED, Batch.Status.PROCESSING)]
    ingested = Claim.objects.filter(batch__source=Batch.Source.FEED).count()
    rejected = RejectedClaim.objects.count()
    pending = AuditReport.objects.filter(claim__batch__source=Batch.Source.FEED,
                                         status=AuditReport.Status.PENDING).count()
    flow = [
        {"label": "SHA claims platform", "icon": "building-2", "tone": "bg-sky-soft text-sky",
         "value": platform["paid"] if platform else "—",
         "note": (f"paid of {platform['lodged']} lodged" if platform else "unreachable")},
        {"label": "Ingested", "icon": "download", "tone": "bg-accent-soft text-accent", "value": ingested,
         "note": f"{rejected} quarantined"},
        {"label": "Collecting", "icon": "layers", "tone": "bg-warn-soft text-warn",
         "value": sum(b.n_claims for b in open_),
         "note": (f"in {len(open_)} open period{'s' if len(open_) != 1 else ''}" if open_ else "no open period")},
        {"label": "Detection", "icon": "scan-search", "tone": "bg-warn-soft text-warn",
         "value": sum(b.n_claims for b in in_detection),
         "note": f"{len(in_detection)} batch{'es' if len(in_detection) != 1 else ''} queued or running"},
        {"label": "Review queue", "icon": "inbox", "tone": "bg-danger-soft text-danger", "value": pending,
         "note": "reports pending review"},
    ]
    return {"state": state, "feed_settings": settings_, "settings_error": settings_error, "platform": platform,
            "platform_error": platform_error, "flow": flow, "periods": periods_, "open_periods": open_,
            "runs": IngestionRun.objects.select_related("triggered_by")[:12],
            "rejections": RejectedClaim.objects.select_related("run")[:15], "rejected_total": rejected}


CLOCK_BUTTONS = [("advance_week", "Advance a week", "skip-forward"), ("advance_month", "Advance 30 days", "fast-forward"),
                 ("play", "Play (15 days a minute)", "play"), ("pause", "Pause clock", "pause")]


@administrator_required
def pipeline(request):
    return render(request, "ingestion/pipeline.html", {**_context(), "clock_buttons": CLOCK_BUTTONS})


@administrator_required
def pipeline_live(request):
    """The part of the page that HTMX refreshes every few seconds."""
    return render(request, "ingestion/live.html", _context())


def _pull(request) -> None:
    try:
        run = sync.sync(user=request.user)
    except sync.Busy:
        messages.info(request, "A pull is already running; its results will appear shortly.")
        return
    if run.status == IngestionRun.Status.SUCCEEDED:
        closed = f", closed {len(run.batches_closed)} audit period(s)" if run.batches_closed else ""
        messages.success(request, f"Pulled {run.fetched} claims: {run.accepted} stored, {run.rejected} quarantined, "
                                  f"{run.duplicates} already held{closed}.")
    else:
        messages.error(request, f"The pull {run.status}: {run.error}")


@administrator_required
def sync_now(request):
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])
    ActivityLog.record(ActivityLog.Action.FEED_SYNC, request.user, "pulled from the claims platform")
    _pull(request)
    return redirect("ingestion:pipeline")


@administrator_required
def close_now(request, pk: int):
    """Close an open audit period before it ends, e.g. to audit a month early."""
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])
    settings_ = sync.feed_settings()
    with transaction.atomic():
        batch = get_object_or_404(Batch.objects.select_for_update(), pk=pk, source=Batch.Source.FEED)
        if batch.status != Batch.Status.OPEN:
            messages.warning(request, "This audit period is already closed.")
            return redirect("ingestion:pipeline")
        if not batch.claims.exists():
            messages.warning(request, "This audit period has no claims yet.")
            return redirect("ingestion:pipeline")
        job = sync.close_batch(batch, settings_["queue"] if settings_["auto_process"] else None, request.user)
        ActivityLog.record(ActivityLog.Action.FEED_CONTROL, request.user,
                           f"closed batch #{pk} early" + (f", job #{job.pk}" if job else ""))
    messages.success(request, f"Closed {batch.name} with {batch.claims_count} claims"
                              + (f"; job #{job.pk} queued on {job.queue}." if job else "."))
    return redirect("ingestion:pipeline")


@administrator_required
def toggle_feed(request):
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])
    state = FeedState.get()
    state.paused = not state.paused
    state.save(update_fields=["paused"])
    ActivityLog.record(ActivityLog.Action.FEED_CONTROL, request.user,
                       "paused scheduled pulls" if state.paused else "resumed scheduled pulls")
    messages.success(request, "Scheduled pulls paused." if state.paused else "Scheduled pulls resumed.")
    return redirect("ingestion:pipeline")


@administrator_required
def clock(request):
    """Operate the simulated platform's clock, then pull, so the page shows the effect at once."""
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])
    action = CLOCK_ACTIONS.get(request.POST.get("action", ""))
    if action is None:
        messages.error(request, "Unknown clock action.")
        return redirect("ingestion:pipeline")
    change, detail = action
    try:
        PlatformClient().change_clock(**change)
    except PlatformError as exc:
        messages.error(request, f"The clock did not move: {exc}")
        return redirect("ingestion:pipeline")
    ActivityLog.record(ActivityLog.Action.FEED_CONTROL, request.user, detail)
    if change["action"] == "advance":
        _pull(request)
    else:
        messages.success(request, detail.capitalize() + ".")
    return redirect("ingestion:pipeline")
