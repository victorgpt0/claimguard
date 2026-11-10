"""The batch summary report (FR-16, #59): counts match a seeded database, and both downloads
have the expected sections and columns."""
import csv
import io
from datetime import date
from decimal import Decimal

import pytest
from django.urls import reverse
from django.utils import timezone

from activity.models import ActivityLog
from audit.models import AuditReport, Batch, Claim, ClassifierVerdict
from audit.summary import CSV_COLUMNS, batch_summary


@pytest.fixture
def batch(reports, make_user):
    """The two seeded reports, plus a flagged claim the classifier cleared, an unflagged claim,
    a decided report, and the run log a job leaves."""
    batch = Batch.objects.get()
    common = dict(batch=batch, patient_ref="M2", icd11_code="DB10.02", service_date=date(2026, 1, 2),
                  clinical_notes="Appendicectomy for acute appendicitis.")
    cleared = Claim.objects.create(claim_ref="CLM3", provider_id="PRV002", claim_amount=Decimal("5000"), **common)
    ClassifierVerdict.objects.create(claim=cleared, verdict="consistent", confidence=0.8, model_version="seed7")
    Claim.objects.create(claim_ref="CLM4", provider_id="=HYPERLINK(1)", claim_amount=Decimal("1000"), **common)
    reports[1].status, reports[1].reviewed_by, reports[1].reviewed_at = "rejected", make_user("reviewer"), timezone.now()
    reports[1].save()
    batch.run_log = {"started_at": "2026-10-26T10:00:00", "summary": {
        "flagged_by": {"isolation_forest": 2, "autoencoder": 1, "text_screen": 1}, "narrative_failures": 0},
        "settings": {"queue": "gpu", "job": 7, "dataset": "v0.6-20k", "thresholds": {"text_screen_threshold": 0.0056}}}
    batch.save()
    return batch


def test_summary_counts_match_the_database(batch):
    s = batch_summary(batch)
    assert (s["claims"], s["flagged"], s["cleared"], s["queued"]) == (4, 3, 1, 2)
    assert s["lanes"] == {"text_inconsistent": 1, "structural_anomaly": 1}
    assert (s["narratives"], s["narrative_failures"]) == (1, 0)
    assert s["amount_claimed"] == Decimal("66000") and s["amount_under_audit"] == Decimal("60000")
    assert s["review"] == {"pending": 1, "approved": 0, "rejected": 1, "escalated": 0} and s["reviewed"] == 1
    assert s["reviewers"] == {"reviewer": {"approved": 0, "rejected": 1, "escalated": 0}}
    assert s["top_providers"][0]["claim__provider_id"] == "PRV001" and s["top_providers"][0]["claims"] == 2
    assert [c["claim__icd11_code"] for c in s["top_codes"]] == ["JB22.1"]   # only queued claims count
    assert s["model_versions"] == ["seed7"] and s["flagged_by"]["isolation_forest"] == 2


def test_an_imported_batch_without_a_run_log_is_summarised(reports):
    s = batch_summary(Batch.objects.get())
    assert (s["flagged"], s["queued"], s["flagged_by"], s["narrative_failures"], s["settings"]) == (2, 2, None, None, {})


def test_the_batch_page_shows_the_summary(admin, batch):
    page = admin.get(reverse("audit:batch", args=[batch.pk])).content.decode()
    assert "Batch summary" in page and "KES 60,000" in page and "chart-funnel" in page
    assert "Download summary (Markdown)" in page and "v0.6-20k" in page


def test_markdown_summary_has_every_section(admin, batch):
    response = admin.get(reverse("audit:batch_summary", args=[batch.pk]))
    text = response.content.decode()
    assert response.status_code == 200 and response["Content-Type"].startswith("text/markdown")
    assert f'batch_{batch.pk}_' in response["Content-Disposition"]
    for section in ("# Batch summary", "## Claims through the pipeline", "## Review progress",
                    "## Queued claims by provider", "## Queued claims by ICD-11 code", "## Models and settings"):
        assert section in text
    assert "| Cleared by the classifier (logged only) | 1 |" in text and "| reviewer | 0 | 1 | 0 |" in text
    assert "text_screen_threshold: 0.0056" in text and "Classifier version: seed7" in text
    assert ActivityLog.objects.get(action="download").detail == f"batch #{batch.pk}: summary (Markdown)"


def test_reports_csv_has_one_row_per_report(admin, batch):
    response = admin.get(reverse("audit:batch_reports", args=[batch.pk]))
    rows = list(csv.DictReader(io.StringIO(response.content.decode())))
    assert response.status_code == 200 and list(rows[0]) == CSV_COLUMNS and len(rows) == 2
    first = rows[0]
    assert (first["claim_id"], first["lane"], first["verdict"], first["decision"]) == (
        "CLM1", "text_inconsistent", "inconsistent", "pending")
    assert first["shap_top3"] == "claim_amount +0.200; stay_days -0.100"
    assert rows[1]["reviewer"] == "reviewer" and rows[1]["decision"] == "rejected"


def test_csv_cells_cannot_become_formulas(admin, batch):
    report = AuditReport.objects.filter(claim__batch=batch).first()
    report.llm_narrative = "=cmd|' /C calc'!A0"
    report.save()
    text = admin.get(reverse("audit:batch_reports", args=[batch.pk])).content.decode()
    assert "'=cmd" in text and ",=cmd" not in text


def test_auditors_cannot_open_batches_or_their_downloads(auditor, batch):
    for name in ("audit:batch", "audit:batch_summary", "audit:batch_reports"):
        assert auditor.get(reverse(name, args=[batch.pk])).status_code == 403


def test_an_unprocessed_batch_has_no_summary(admin, db):
    batch = Batch.objects.create(name="new.csv")
    assert admin.get(reverse("audit:batch_summary", args=[batch.pk])).status_code == 302
    assert admin.get(reverse("audit:batch_reports", args=[batch.pk])).status_code == 302
    assert "Batch summary" not in admin.get(reverse("audit:batch", args=[batch.pk])).content.decode()
