"""The batch summary report (FR-16, #59): what happened to one batch's claims, from the
database and the job's run log, nothing recomputed. Shown on the batch page and downloadable
as Markdown; the batch's reports download as CSV.

The counts come from the database, so batches imported from the Streamlit-era SQLite, which
have no run log, are summarised too: a claim with a classifier verdict was flagged at stage 1
(the classifier only sees flagged claims), and a flagged claim without a report was cleared by
the classifier. The run log adds what the database does not keep: which detector flagged how
many, failed narratives, and the thresholds the job ran with.
"""
from __future__ import annotations

import csv

from django.db.models import Count, Q, Sum

from .models import AuditReport, Batch, Claim, ClassifierVerdict

DECISIONS = ["approved", "rejected", "escalated"]
TOP = 5
CSV_COLUMNS = ["report_id", "claim_id", "provider_id", "icd11_code", "claim_amount", "anomaly_score",
               "flag_sources", "lane", "verdict", "classifier_score", "model_version", "shap_top3",
               "narrative", "decision", "reviewer", "reviewed_at"]


def batch_summary(batch: Batch) -> dict:
    claims = Claim.objects.filter(batch=batch)
    reports = AuditReport.objects.filter(claim__batch=batch)
    verdicts = ClassifierVerdict.objects.filter(claim__batch=batch)
    flagged = claims.filter(verdicts__isnull=False).distinct().count()
    lanes = reports.aggregate(
        text_inconsistent=Count("id", filter=Q(verdict__verdict=ClassifierVerdict.Verdict.INCONSISTENT)),
        structural_anomaly=Count("id", filter=Q(verdict__verdict=ClassifierVerdict.Verdict.CONSISTENT)))
    queued = sum(lanes.values())
    by_status = dict(reports.values_list("status").annotate(n=Count("id")))
    reviewers = {}
    for row in (reports.exclude(reviewed_by=None).values("reviewed_by__username", "status")
                .annotate(n=Count("id")).order_by("reviewed_by__username")):
        reviewers.setdefault(row["reviewed_by__username"], dict.fromkeys(DECISIONS, 0))[row["status"]] = row["n"]
    run_log = batch.run_log or {}
    summary = run_log.get("summary") or {}
    return {
        "claims": claims.count(),
        "flagged": flagged,
        "flagged_by": summary.get("flagged_by"),
        "cleared": flagged - queued,
        "lanes": lanes,
        "queued": queued,
        "inconsistent": lanes["text_inconsistent"],
        "narratives": reports.exclude(llm_narrative=None).exclude(llm_narrative="").count(),
        "narrative_failures": summary.get("narrative_failures"),
        "amount_claimed": claims.aggregate(s=Sum("claim_amount"))["s"] or 0,
        "amount_under_audit": reports.aggregate(s=Sum("claim__claim_amount"))["s"] or 0,
        "review": {"pending": by_status.get("pending", 0), **{d: by_status.get(d, 0) for d in DECISIONS}},
        "reviewed": queued - by_status.get("pending", 0),
        "reviewers": reviewers,
        "top_providers": list(reports.values("claim__provider_id")
                              .annotate(claims=Count("id"), amount=Sum("claim__claim_amount"))
                              .order_by("-claims", "-amount")[:TOP]),
        "top_codes": list(reports.values("claim__icd11_code")
                          .annotate(claims=Count("id"), amount=Sum("claim__claim_amount"))
                          .order_by("-amount", "-claims")[:TOP]),
        "model_versions": sorted(set(verdicts.values_list("model_version", flat=True))),
        "settings": run_log.get("settings") or {},
        "started_at": run_log.get("started_at"),
    }


def funnel_counts(summary: dict) -> dict:
    """The figures charts.funnel draws, from a batch summary."""
    return {"claims": summary["claims"], "flagged_stage1": summary["flagged"],
            "inconsistent": summary["inconsistent"], "narratives": summary["narratives"]}


def _cell(value) -> str:
    """Text a spreadsheet would read as a formula (=, +, -, @) is quoted, so opening the export
    cannot run anything. Numbers are written as numbers."""
    text = "" if value is None else str(value)
    return "'" + text if text[:1] in ("=", "+", "-", "@") else text


def _shap_top3(attribution: dict) -> str:
    top = sorted((attribution or {}).items(), key=lambda item: abs(item[1]), reverse=True)[:3]
    return "; ".join(f"{feature} {value:+.3f}" for feature, value in top)


def write_reports_csv(batch: Batch, out) -> int:
    writer = csv.writer(out)
    writer.writerow(CSV_COLUMNS)
    rows = (AuditReport.objects.filter(claim__batch=batch).select_related("claim", "verdict", "reviewed_by")
            .order_by("id"))
    n = 0
    for r in rows:
        writer.writerow([
            r.pk, _cell(r.claim.claim_ref), _cell(r.claim.provider_id), _cell(r.claim.icd11_code),
            r.claim.claim_amount, r.claim.anomaly_score, _cell(" ".join(r.flag_sources or [])), r.lane or "",
            r.verdict.verdict if r.verdict else "", r.verdict.confidence if r.verdict else "",
            _cell(r.verdict.model_version) if r.verdict else "", _cell(_shap_top3(r.shap_attribution)),
            _cell(r.llm_narrative), r.status, r.reviewed_by.username if r.reviewed_by else "",
            r.reviewed_at.isoformat() if r.reviewed_at else ""])
        n += 1
    return n
