"""Batch upload and processing (phase 3): upload stores nothing unless every line passes and
never accepts held-out claims; a job runs the pipeline through the Django repository with
progress, and a failed job leaves no partial reports."""
import io

import pandas as pd
import pytest
from django.test import Client
from django.urls import reverse

from activity.models import ActivityLog
from audit import tasks
from audit.models import AuditReport, Batch, Claim, ClassifierVerdict, Job
from audit.tests.test_validation import GOOD
from audit.validation import UPLOAD_COLUMNS


def _csv(*rows) -> io.BytesIO:
    buffer = io.BytesIO()
    pd.DataFrame([{**GOOD, **row} for row in rows], columns=UPLOAD_COLUMNS).to_csv(buffer, index=False)
    buffer.seek(0)
    buffer.name = "claims.csv"
    return buffer


@pytest.fixture
def administrator(make_user):
    client = Client()
    client.force_login(make_user("admin", role="Administrator"))
    return client


def _upload(client, *rows, synthetic=True):
    return client.post(reverse("audit:upload"), {"file": _csv(*rows), **({"synthetic": "on"} if synthetic else {})})


def test_a_valid_file_becomes_a_batch(administrator, pipeline_files):
    response = _upload(administrator, {}, {"claim_id": "CLM2", "discharge_date": "", "claim_amount": "0"})
    batch = Batch.objects.get()
    assert response.status_code == 302 and response["Location"] == reverse("audit:batch", args=[batch.pk])
    assert (batch.status, batch.claims_count, batch.uploaded_by.username) == ("uploaded", 2, "admin")
    assert batch.source_csv.startswith(",".join(UPLOAD_COLUMNS)) and len(batch.source_sha256) == 64
    outpatient = Claim.objects.get(claim_ref="CLM2")
    assert outpatient.discharge_date is None and outpatient.facility_level == 4 and outpatient.patient_age == 34
    assert ActivityLog.objects.get(action="upload").detail == f"batch #{batch.pk}: claims.csv, 2 claims"


def test_an_invalid_file_stores_nothing_and_lists_every_problem(administrator, pipeline_files):
    response = _upload(administrator, {"claim_amount": "-1"}, {"claim_id": "CLM_TEST_1"})
    page = response.content.decode()
    assert response.status_code == 200 and "2 problems: nothing was stored" in page
    assert "held-out test split" in page and "at least 0" in page
    assert not Batch.objects.exists() and not Claim.objects.exists()
    assert "rejected claims.csv: 2 problems" in ActivityLog.objects.get(action="upload").detail


def test_synthetic_confirmation_is_required(administrator, pipeline_files):
    _upload(administrator, {}, synthetic=False)
    assert not Batch.objects.exists()


def test_auditors_cannot_upload(make_user, pipeline_files):
    # Separation of duties: the Administrator decides what is audited, the Auditor reviews it.
    client = Client()
    client.force_login(make_user())
    assert client.get(reverse("audit:upload")).status_code == 403
    assert client.post(reverse("audit:upload"), {"file": _csv({}), "synthetic": "on"}).status_code == 403
    assert not Batch.objects.exists()


class FakeLog:
    def summary(self):
        return {"claims": 2, "flagged_stage1": 1, "flagged_by": {"isolation_forest": 1, "autoencoder": 0, "text_screen": 0},
                "reported": {"text_inconsistent": 1, "structural_anomaly": 0}, "screen_false_positives": 0,
                "inconsistent": 1, "narratives": 0, "narrative_failures": 0}

    started_at, settings = "2026-10-26T10:00:00", {}


class FakePipeline:
    """Stands in for AuditPipeline: flags the first claim and writes through the repository."""
    def __init__(self, repository, fail=False):
        self.repository, self.fail = repository, fail

    def run(self, claims, settings, progress):
        progress("explaining", 0, 1)
        for i, claim in enumerate(claims.itertuples()):
            if i == 0:
                claim_id = self.repository.save_claim({"claim_ref": claim.claim_id, "anomaly_score": 0.7})
                verdict = self.repository.save_verdict(claim_id, "inconsistent", 0.9, "fake")
                self.repository.save_report(claim_id, verdict, {"claim_amount": 0.3}, None, ["isolation_forest"])
            progress("auditing", i + 1, len(claims))
            if self.fail:
                raise RuntimeError("GPU out of memory")
        return FakeLog()


@pytest.mark.parametrize("fail", [False, True])
def test_a_job_runs_the_pipeline_and_reports_its_outcome(administrator, pipeline_files, monkeypatch, settings,
                                                         django_capture_on_commit_callbacks, fail):
    # Run the task in this process, as a worker would, instead of sending it to Redis.
    monkeypatch.setattr(tasks.process_batch, "apply_async", lambda args, queue: tasks.process_batch.apply(args=args))
    monkeypatch.setattr(tasks, "build_pipeline", lambda config, narratives, repo, progress: FakePipeline(repo, fail))
    monkeypatch.setattr("audit.services.workers_online", lambda: {})
    _upload(administrator, {}, {"claim_id": "CLM2"})
    batch = Batch.objects.get()
    with django_capture_on_commit_callbacks(execute=True):
        response = administrator.post(reverse("audit:start_job", args=[batch.pk]), {"queue": "cpu"})
    assert response.status_code == 302
    job, batch = Job.objects.get(), Batch.objects.get()
    assert ActivityLog.objects.filter(action="job_started").exists()
    progress = administrator.get(reverse("audit:job_progress", args=[job.pk]))
    assert progress.status_code == 286  # finished: HTMX stops polling
    if fail:
        assert (job.status, batch.status) == ("failed", "failed") and "GPU out of memory" in batch.error
        assert not AuditReport.objects.exists() and not ClassifierVerdict.objects.exists()  # no partial results
        assert "Failed: RuntimeError" in progress.content.decode()
    else:
        assert (job.status, batch.status, job.done, job.total) == ("succeeded", "processed", 2, 2)
        report = AuditReport.objects.get()
        assert report.claim.claim_ref == "CLM1" and report.claim.anomaly_score == 0.7 and report.lane == "text_inconsistent"
        assert batch.run_log["summary"]["flagged_stage1"] == 1
        assert "1 queued for review" in job.message


def test_a_batch_is_processed_once(administrator, pipeline_files, monkeypatch):
    monkeypatch.setattr("audit.services.workers_online", lambda: {})
    monkeypatch.setattr(tasks.process_batch, "apply_async", lambda *a, **k: None)
    _upload(administrator, {})
    batch = Batch.objects.get()
    administrator.post(reverse("audit:start_job", args=[batch.pk]), {"queue": "gpu"})
    administrator.post(reverse("audit:start_job", args=[batch.pk]), {"queue": "cpu"})
    assert Job.objects.count() == 1 and Job.objects.get().queue == "gpu"
    page = administrator.get(reverse("audit:batch", args=[batch.pk])).content.decode()
    assert "Waiting for a worker on the GPU queue" in page and "hx-get" in page
