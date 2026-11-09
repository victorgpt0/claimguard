"""The claims feed (#66), end to end against the real platform API (sha_cdp, run in-process):
pulls, audit periods, quarantine, de-duplication, closing and processing a period, failures,
and the Data pipeline page's permissions."""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest
from django.urls import reverse
from fastapi.testclient import TestClient

from activity.models import ActivityLog
from audit import tasks
from audit.models import AuditReport, Batch, Claim, Job
from audit.tests.conftest import admin, auditor, pipeline_files  # noqa: F401 (fixtures)
from audit.tests.test_batches import FakePipeline
from ingestion import periods, sync
from ingestion.client import PlatformClient, PlatformError
from ingestion.models import FeedState, IngestionRun, RejectedClaim
from sha_cdp import clock, db, seed
from sha_cdp.app import app

N = 60


def claims_set(n=N, bad_code_at=None):
    dates = pd.date_range("2025-10-01", periods=n, freq="2D")
    frame = pd.DataFrame({
        "claim_id": [f"LCLM{i:06d}" for i in range(1, n + 1)], "provider_id": "PRV0001", "facility_level": 4,
        "member_id": [f"LMBR{i:06d}" for i in range(1, n + 1)], "beneficiary_type": "principal",
        "patient_age": 30.0, "patient_sex": "female", "icd11_code": "JB22.1", "claim_amount": 30000.0,
        "service_date": dates.strftime("%Y-%m-%d"), "discharge_date": (dates + timedelta(days=3)).strftime("%Y-%m-%d"),
        "clinical_notes": "Emergency caesarean section for obstructed labour, mother and baby stable.",
        "is_fraud": 0, "label": "consistent",
    })
    if bad_code_at is not None:
        frame.loc[bad_code_at, "icd11_code"] = "ZZ99.9"
    return frame


@pytest.fixture
def platform(tmp_path, monkeypatch, settings):
    monkeypatch.setenv("SHA_CDP_DATABASE_URL", f"sqlite:///{tmp_path / 'cdp.sqlite'}")
    monkeypatch.setenv("SHA_CDP_API_KEY", "read-key")
    monkeypatch.setenv("SHA_CDP_ADMIN_TOKEN", "admin-token")
    settings.SHA_CDP_URL, settings.SHA_CDP_API_KEY, settings.SHA_CDP_ADMIN_TOKEN = "http://testserver", "read-key", "admin-token"
    db.engine.cache_clear()
    eng = db.engine()
    http = TestClient(app)
    # Every PlatformClient the feed or the views create talks to the in-process platform.
    monkeypatch.setattr("ingestion.client.requests.Session", lambda: http)
    monkeypatch.setattr(tasks.process_batch, "apply_async", lambda *a, **k: None)
    yield eng
    db.engine.cache_clear()


def test_periods():
    assert periods.period_for(date(2026, 2, 14), "monthly") == (date(2026, 2, 1), date(2026, 2, 28))
    assert periods.period_for(date(2026, 2, 14), "weekly") == (date(2026, 2, 9), date(2026, 2, 15))
    start, end = periods.period_for(date(2026, 2, 14), "fortnightly")
    assert (end - start).days == 13 and start <= date(2026, 2, 14) <= end and start.weekday() == 0
    with pytest.raises(ValueError):
        periods.period_for(date(2026, 2, 14), "daily")


def test_the_feed_collects_periods_and_closes_them_as_the_clock_passes(platform, pipeline_files, db):
    eng = platform
    seed.load(claims_set(), eng)
    first = sync.sync()
    assert (first.status, first.fetched) == ("succeeded", 0) and not Batch.objects.exists()   # nothing paid yet

    clock.advance(eng, 75)
    run = sync.sync()
    as_of = clock.now(eng)
    assert run.accepted > 0 and run.accepted == Claim.objects.count()
    assert all(c.paid_at.replace(tzinfo=None) <= as_of for c in Claim.objects.all())
    for batch in Batch.objects.all():
        assert batch.source == "feed" and batch.period_start.day == 1
        assert all(batch.period_start <= c.paid_at.date() <= batch.period_end for c in batch.claims.all())
    # Periods that ended before the clock are closed and queued; the current one is still collecting.
    open_ = Batch.objects.filter(status="open")
    assert open_.count() == 1 and open_.get().period_end >= as_of.date()
    assert set(run.batches_closed) == set(Batch.objects.filter(status="uploaded").values_list("pk", flat=True))
    assert Job.objects.filter(queue="cpu").count() == len(run.batches_closed)

    clock.advance(eng, 400)
    sync.sync()
    assert Claim.objects.count() == N and not Batch.objects.filter(status="open").exists()
    again = sync.sync()
    assert (again.fetched, again.accepted) == (0, 0) and Claim.objects.count() == N   # a repeat pull adds nothing
    closed = Batch.objects.order_by("period_start").first()
    csv = pd.read_csv(pd.io.common.StringIO(closed.source_csv))
    assert len(csv) == closed.claims_count == closed.claims.count() and "is_fraud" not in csv.columns


def test_a_closed_period_runs_through_the_pipeline(platform, pipeline_files, db, monkeypatch,
                                                  django_capture_on_commit_callbacks):
    monkeypatch.setattr(tasks.process_batch, "apply_async", lambda args, queue: tasks.process_batch.apply(args=args))
    monkeypatch.setattr(tasks, "build_pipeline", lambda config, narratives, repo, progress: FakePipeline(repo, False))
    seed.load(claims_set(), platform)
    clock.advance(platform, 500)
    with django_capture_on_commit_callbacks(execute=True):
        sync.sync()
    assert Job.objects.exists() and set(Job.objects.values_list("status", flat=True)) == {"succeeded"}
    assert set(Batch.objects.values_list("status", flat=True)) == {"processed"}
    report = AuditReport.objects.first()
    assert report.claim.batch.source == "feed" and report.claim.claim_ref.startswith("LCLM")


def test_bad_claims_are_quarantined_and_known_ones_skipped(platform, pipeline_files, db):
    seed.load(claims_set(bad_code_at=0), platform)
    upload = Batch.objects.create(name="earlier upload")
    Claim.objects.create(batch=upload, claim_ref="LCLM000002", provider_id="PRV0001", patient_ref="M", icd11_code="JB22.1",
                         claim_amount=1, service_date=date(2025, 10, 1), clinical_notes="Earlier copy of the claim.")
    clock.advance(platform, 500)
    run = sync.sync()
    assert (run.rejected, run.duplicates, run.accepted) == (1, 1, N - 2)
    rejected = RejectedClaim.objects.get()
    assert rejected.claim_ref == "LCLM000001" and rejected.problems[0]["column"] == "icd11_code"
    assert "clinical_notes" not in rejected.record and rejected.record["clinical_notes_words"] > 5


def test_a_failed_pull_is_recorded_and_keeps_the_cursor(platform, pipeline_files, db, monkeypatch):
    def down(*args, **kwargs):
        raise PlatformError("claims platform unreachable at http://testserver: ConnectError")
    monkeypatch.setattr(PlatformClient, "paid_claims", down)
    run = sync.sync()
    state = FeedState.get()
    assert run.status == "failed" and "unreachable" in run.error and "unreachable" in state.last_error
    assert state.after_paid_at is None and state.running_since is None


def test_scheduled_pulls_respect_pause_and_never_overlap(platform, pipeline_files, db):
    from django.utils import timezone
    FeedState.objects.update_or_create(pk=1, defaults={"paused": True})
    assert sync.sync(scheduled=True) is None and not IngestionRun.objects.exists()
    FeedState.objects.filter(pk=1).update(paused=False, running_since=timezone.now())
    with pytest.raises(sync.Busy):
        sync.sync()


def test_the_page_and_its_controls_by_role(platform, pipeline_files, auditor, admin):
    seed.load(claims_set(), platform)
    page = auditor.get(reverse("ingestion:pipeline"))
    assert page.status_code == 200 and "Data pipeline" in page.content.decode() and "connected" in page.content.decode()
    assert "Simulation controls" not in page.content.decode()
    assert auditor.post(reverse("ingestion:clock"), {"action": "advance_week"}).status_code == 403
    assert auditor.post(reverse("ingestion:toggle")).status_code == 403

    assert admin.post(reverse("ingestion:clock"), {"action": "advance_month"}).status_code == 302
    assert IngestionRun.objects.count() == 1 and Claim.objects.exists()   # the move is followed by a pull
    assert ActivityLog.objects.filter(action="feed_control").exists()
    assert "Simulation controls" in admin.get(reverse("ingestion:pipeline")).content.decode()

    clock.advance(platform, 12)   # into the middle of the next month, so a period is open
    sync.sync()
    open_batch = Batch.objects.filter(status="open").first()
    assert auditor.post(reverse("ingestion:close", args=[open_batch.pk])).status_code == 302
    open_batch.refresh_from_db()
    assert open_batch.status == "uploaded" and open_batch.jobs.exists()
    assert auditor.post(reverse("ingestion:sync")).status_code == 302
    assert ActivityLog.objects.filter(action="feed_sync").exists()
    assert auditor.get(reverse("ingestion:live")).status_code == 200


def test_the_page_says_when_the_platform_is_unreachable(auditor, pipeline_files, settings):
    settings.SHA_CDP_URL = "http://127.0.0.1:9"   # nothing listens there
    body = auditor.get(reverse("ingestion:pipeline")).content.decode()
    assert "unreachable" in body and "No claims received yet" in body
