"""The dashboard pages on Django (phase 2): role access, the queue, the six-panel report, the
decision with its reviewer, activity entry and feedback example, and the monitoring pages."""
import json
from datetime import date
from decimal import Decimal

import pytest
from django.test import Client
from django.urls import reverse

from activity.models import ActivityLog
from audit.models import AuditReport, Batch, Claim, ClassifierVerdict


def test_signed_out_visitors_are_sent_to_sign_in(db):
    for name in ("audit:overview", "audit:queue", "audit:model", "audit:feedback"):
        response = Client().get(reverse(name))
        assert response.status_code == 302 and "/accounts/login/" in response["Location"]


def test_each_role_reaches_only_its_pages(auditor, admin, reports, pipeline_files):
    assert auditor.get(reverse("audit:queue")).status_code == 200
    assert auditor.get(reverse("audit:model")).status_code == 403
    assert auditor.get(reverse("audit:feedback")).status_code == 403
    assert admin.get(reverse("audit:queue")).status_code == 403
    assert admin.get(reverse("audit:report", args=[reports[0].pk])).status_code == 403
    assert admin.get(reverse("audit:model")).status_code == 200
    nav = auditor.get(reverse("audit:overview")).content.decode()
    assert "Review queue" in nav and "Model performance" not in nav


def test_overview_statistics(auditor, reports, pipeline_files):
    page = auditor.get(reverse("audit:overview")).content.decode()
    assert "Flagged claims" in page and "KES 60,000" in page and "1 with narrative" in page
    assert 'id="chart-status"' in page  # chart data via json_script
    # every chart category is named beside its count, not shown by colour alone
    assert "Approved (flag confirmed)" in page and "Rejected (false positive)" in page and "Next in the queue" in page


def test_overview_empty_state_says_what_to_do(auditor, admin, db):
    page = auditor.get(reverse("audit:overview")).content.decode()
    assert "No audit reports yet" in page and "Upload a batch to start" in page
    assert "chart-status" not in page and "chart-verdicts" not in page  # no report charts without reports
    assert "A Fraud Auditor uploads batches" in admin.get(reverse("audit:overview")).content.decode()


def test_queue_orders_by_score_and_filters_by_lane(auditor, reports, pipeline_files):
    page = auditor.get(reverse("audit:queue")).content.decode()
    assert page.index("CLM2") < page.index("CLM1")  # higher anomaly score first
    page = auditor.get(reverse("audit:queue"), {"lane": "text_inconsistent"}).content.decode()
    assert "CLM1" in page and "CLM2" not in page


def test_report_shows_the_six_panels(auditor, reports, pipeline_files):
    page = auditor.get(reverse("audit:report", args=[reports[0].pk])).content.decode()
    for text in ("1 · Structured claim fields", "Forceps-assisted vaginal delivery.",
                 "Delivery by emergency caesarean section", "Flagged by: Isolation Forest, text-consistency screen",
                 "inconsistent", "Notes describe a forceps delivery.", "— (outpatient)"):
        assert text in page
    page = auditor.get(reverse("audit:report", args=[reports[1].pk])).content.decode()
    assert "No narrative: the classifier found the notes consistent" in page


def test_the_verdict_score_is_not_presented_as_confidence(auditor, reports, pipeline_files):
    """The classifier's softmax output is uncalibrated (Course Sprint III brief): the report
    and the CSV export call it a model score, never confidence or a percentage."""
    page = auditor.get(reverse("audit:report", args=[reports[0].pk])).content.decode()
    assert "model score 0.90" in page and "not a calibrated probability" in page
    assert "confidence" not in page.lower() and "90%" not in page
    header = auditor.get(reverse("audit:batch_reports", args=[reports[0].claim.batch_id])).content.decode().splitlines()[0]
    assert "classifier_score" in header.split(",") and "confidence" not in header


def test_approving_records_reviewer_activity_and_feedback(auditor, reports, pipeline_files):
    response = auditor.post(reverse("audit:decide", args=[reports[0].pk]), {"decision": "approved", "lane": ""})
    report = AuditReport.objects.get(pk=reports[0].pk)
    assert (report.status, report.reviewed_by.username) == ("approved", "auditor") and report.reviewed_at
    assert response.status_code == 302 and f"/reports/{reports[1].pk}/" in response["Location"]  # the next one
    entry = ActivityLog.objects.filter(action="decision").get()
    assert (entry.user.username, entry.detail) == ("auditor", f"report #{reports[0].pk}: approved")
    example = json.loads((pipeline_files / "artifacts" / "feedback_dataset" / "examples.jsonl").read_text())
    assert example["label"] == "inconsistent"
    assert example["icd11_definition"] == "Delivery by emergency caesarean section"


def test_a_report_is_decided_once_and_only_by_post(auditor, reports, pipeline_files):
    url = reverse("audit:decide", args=[reports[1].pk])
    assert auditor.get(url).status_code == 405
    auditor.post(url, {"decision": "escalated"})
    auditor.post(url, {"decision": "rejected"})
    assert AuditReport.objects.get(pk=reports[1].pk).status == "escalated"
    assert not (pipeline_files / "artifacts" / "feedback_dataset" / "examples.jsonl").exists()  # escalate adds none
    auditor.post(url.replace(str(reports[1].pk), str(reports[0].pk)), {"decision": "delete"})
    assert AuditReport.objects.get(pk=reports[0].pk).status == "pending"


def test_feedback_page_shows_trigger_and_history(admin, auditor, reports, pipeline_files):
    auditor.post(reverse("audit:decide", args=[reports[0].pk]), {"decision": "approved"})
    history = pipeline_files / "experiments" / "retraining_history.jsonl"
    history.parent.mkdir()
    history.write_text(json.dumps({"run_at": "2026-10-20T09:30:00+00:00", "feedback_examples": 52,
                                   "candidate_path": "models/classifier_lora/candidates/20261020T093000Z",
                                   "candidate_metrics": {"f1": 0.95, "recall": 0.97},
                                   "active_metrics": {"f1": 0.94, "recall": 0.97}, "promoted": True}) + "\n")
    page = admin.get(reverse("audit:feedback")).content.decode()
    assert "Retraining due" in page and "20261020T093000Z" in page and "0.950" in page


def test_model_page_without_a_checkpoint(admin, pipeline_files):
    assert "No classifier checkpoint" in admin.get(reverse("audit:model")).content.decode()


def test_pages_still_open_without_config(auditor, reports, settings, tmp_path):
    settings.CLAIMGUARD_CONFIG = tmp_path / "missing.yaml"
    response = auditor.get(reverse("audit:report", args=[reports[0].pk]))
    assert response.status_code == 200 and "The ICD-11 cache is not available" in response.content.decode()


def test_overview_funnel_comes_from_the_pool_run_only(auditor, reports, pipeline_files):
    runs = pipeline_files / "experiments" / "pipeline_runs"
    runs.mkdir(parents=True)
    summary = {"flagged_stage1": 1, "inconsistent": 1, "narratives": 1}
    (runs / "a_pool.json").write_text(json.dumps({"started_at": "2026-10-18T20:28:52", "settings": {"batch": "deployment_pool"},
                                                  "summary": {**summary, "claims": 2000}}))
    (runs / "b_validation.json").write_text(json.dumps({"started_at": "2026-10-25T17:53:00", "settings": {"batch": "validation"},
                                                        "summary": {**summary, "claims": 3600}}))
    page = auditor.get(reverse("audit:overview")).content.decode()
    assert "batch of 2000 claims" in page and "3600" not in page
