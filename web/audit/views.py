"""The dashboard's pages (IR-03). The Fraud Auditor reviews (queue, report, decision); the
Administrator monitors (model performance, feedback loop) and manages accounts in the admin
site; both see the overview."""
from __future__ import annotations

from collections import Counter

from django.contrib import messages
from django.db import transaction
from django.db.models import Count, Sum
from django.http import HttpResponseNotAllowed
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils import timezone

from accounts.roles import administrator_required, any_role_required, auditor_required
from activity.models import ActivityLog

from . import charts, services
from .models import AuditReport, ClassifierVerdict

DECISIONS = [AuditReport.Status.APPROVED, AuditReport.Status.REJECTED, AuditReport.Status.ESCALATED]
LANES = {"text_inconsistent": ClassifierVerdict.Verdict.INCONSISTENT,
         "structural_anomaly": ClassifierVerdict.Verdict.CONSISTENT}


def kes(amount) -> str:
    return f"KES {float(amount or 0):,.0f}"


def _config_or_none(request):
    try:
        return services.config()
    except services.ConfigMissing as exc:
        messages.warning(request, f"{exc} Pages that read pipeline outputs are limited until it exists.")
        return None


# --- Overview (both roles) --------------------------------------------------------------------

# KPI tones are whole class names so Tailwind finds them here. Each category keeps one colour (a design token) wherever it appears, and is always named
# beside its count, so no chart relies on colour alone.
STATUS_SEGMENTS = [("pending", "Pending", "sun"), ("approved", "Approved (flag confirmed)", "red"),
                   ("rejected", "Rejected (false positive)", "green"), ("escalated", "Escalated", "violet")]
VERDICT_SEGMENTS = [("inconsistent", "Inconsistent", "red"), ("consistent", "Consistent", "sky"),
                    ("none", "No verdict", "faint")]


def _segments(counts: dict, spec: list[tuple]) -> list[dict]:
    total = sum(counts.values()) or 1
    return [{"key": key, "label": label, "color": color, "reports": counts.get(key, 0),
             "pct": f"{counts.get(key, 0) / total:.0%}"} for key, label, color in spec]


def _stages(summary: dict) -> list[dict]:
    """The pipeline funnel as rows: each stage's count and its share of the claims screened."""
    claims = summary.get("claims") or 0
    rows = [("Screened", "claims in the batch", summary.get("claims"), "line-strong"),
            ("Flagged at stage 1", "anomaly detectors and text screen", summary.get("flagged_stage1"), "sun"),
            ("Inconsistent", "classifier: notes contradict the code", summary.get("inconsistent"), "accent"),
            ("Explained", "LLM narrative written", summary.get("narratives"), "sky")]
    stages = []
    for label, detail, n, color in rows:
        if n is None:
            continue
        share = n / claims if claims else 0
        stages.append({"label": label, "detail": detail, "claims": f"{n:,}", "color": color,
                       "pct": f"{share:.1%}" if claims else "",
                       # the bar keeps a visible sliver for tiny but non-zero stages
                       "width": f"{max(share * 100, 1.5 if n else 0):.1f}"})
    return stages


@any_role_required
def overview(request):
    reports = AuditReport.objects.select_related("claim", "verdict")
    total = reports.count()
    context = {"total": total}
    charts_ = {}
    if total:
        by_status = dict(reports.values_list("status").annotate(n=Count("id")))
        verdicts = Counter(r.verdict.verdict if r.verdict else "none" for r in reports)
        inconsistent = reports.filter(verdict__verdict=ClassifierVerdict.Verdict.INCONSISTENT)
        pending = by_status.get("pending", 0)
        reviewed = total - pending
        context.update(
            reviewed_share=reviewed / total,
            kpis=[
                {"label": "Flagged claims", "value": f"{total:,}", "icon": "flag", "tone": "bg-accent-soft text-accent",
                 "note": f"{kes(reports.aggregate(s=Sum('claim__claim_amount'))['s'])} under audit"},
                {"label": "Pending review", "value": f"{pending:,}", "icon": "inbox", "tone": "bg-warn-soft text-warn",
                 "note": f"{reviewed / total:.0%} of reports reviewed", "progress": reviewed / total},
                {"label": "Inconsistent verdicts", "value": f"{inconsistent.count():,}", "icon": "file-exclamation-point",
                 "tone": "bg-danger-soft text-danger", "note": f"{inconsistent.exclude(llm_narrative=None).count()} with narrative"},
                {"label": "Flags confirmed", "value": f"{by_status.get('approved', 0):,}", "icon": "badge-check",
                 "tone": "bg-sky-soft text-sky", "note": f"{by_status.get('rejected', 0)} false positives, "
                                        f"{by_status.get('escalated', 0)} escalated"},
            ],
            status_segments=_segments(by_status, STATUS_SEGMENTS),
            verdict_segments=_segments(verdicts, VERDICT_SEGMENTS),
        )
        if request.user.is_auditor:
            context["next_up"] = _pending(None, "score")[:5]
        providers = (reports.values("claim__provider_id")
                     .annotate(claims=Count("id"), amount=Sum("claim__claim_amount")).order_by("-claims")[:10])
        codes = (reports.values("claim__icd11_code")
                 .annotate(claims=Count("id"), amount=Sum("claim__claim_amount")).order_by("-amount")[:10])
        charts_ = {
            "status": charts.donut(context["status_segments"]),
            "verdicts": charts.donut(context["verdict_segments"]),
            "scores": charts.anomaly_histogram([
                {"anomaly_score": r.claim.anomaly_score, "verdict": r.verdict.verdict if r.verdict else "none"}
                for r in reports if r.claim.anomaly_score is not None]),
            "shap": charts.shap_frequency([r.shap_attribution for r in reports]),
            "providers": charts.horizontal_bars(
                [{"provider": p["claim__provider_id"], "claims": p["claims"], "amount": float(p["amount"])}
                 for p in providers], "provider", "claims", "Flagged claims",
                tooltip=[{"field": "provider"}, {"field": "claims"}, {"field": "amount", "format": ",.0f", "title": "KES"}]),
            "codes": charts.horizontal_bars(
                [{"code": c["claim__icd11_code"], "claims": c["claims"], "amount": float(c["amount"])}
                 for c in codes], "code", "amount", "KES",
                tooltip=[{"field": "code"}, {"field": "claims"}, {"field": "amount", "format": ",.0f", "title": "KES"}]),
        }
    if _config_or_none(request) is not None:
        run = services.latest_pipeline_run()
        if run:
            context["run"] = run
            context["stages"] = _stages(run.get("summary") or {})
            by_typology = (run.get("evaluation") or {}).get("flagged_by_typology")
            if by_typology:
                charts_["typology"] = charts.share_by_typology(
                    [{"typology": services.TYPOLOGY_NAMES.get(t, t), "share": v["sum"] / v["size"],
                      "flagged": v["sum"], "claims": v["size"]} for t, v in by_typology.items()],
                    "Share flagged at stage 1 (ground truth)")
    context["charts"] = charts_
    context["chart_specs"] = {f"chart-{name}": spec for name, spec in charts_.items()}
    return render(request, "audit/overview.html", context)


# --- Review (Fraud Auditor) ---------------------------------------------------------------------

def _pending(lane: str | None, order: str):
    pending = (AuditReport.objects.filter(status=AuditReport.Status.PENDING)
               .select_related("claim", "verdict"))
    if lane in LANES:
        pending = pending.filter(verdict__verdict=LANES[lane])
    return pending.order_by("-claim__anomaly_score", "id") if order == "score" else pending.order_by("created_at", "id")


@auditor_required
def queue(request):
    lane, order = request.GET.get("lane", ""), request.GET.get("order", "score")
    pending = _pending(lane, order)
    lane_counts = dict(AuditReport.objects.filter(status=AuditReport.Status.PENDING)
                       .values_list("verdict__verdict").annotate(n=Count("id")))
    return render(request, "audit/queue.html", {
        "reports": pending[:200], "pending_count": pending.count(), "lane": lane, "order": order,
        "lane_counts": {"text_inconsistent": lane_counts.get("inconsistent", 0),
                        "structural_anomaly": lane_counts.get("consistent", 0)}})


@auditor_required
def report_detail(request, pk: int):
    """The six panels in pipeline order (NFR-07): the claim's inputs (1-3), then what each
    stage concluded (4-6), on one page with the decision."""
    report = get_object_or_404(AuditReport.objects.select_related("claim", "verdict", "reviewed_by"), pk=pk)
    claim = report.claim
    stay = None
    if claim.discharge_date:
        stay = (claim.discharge_date - claim.service_date).days
    definition = None
    if _config_or_none(request) is not None:
        definition = services.icd11_definition(claim.icd11_code)
    return render(request, "audit/report.html", {
        "report": report, "claim": claim, "stay": stay, "definition": definition, "amount": kes(claim.claim_amount),
        "sources": [services.SOURCE_NAMES.get(s, s) for s in report.flag_sources or []],
        "shap_chart": charts.shap_contributions(report.shap_attribution) if report.shap_attribution else None,
        "decisions": DECISIONS, "query": request.GET.urlencode()})


@auditor_required
def decide(request, pk: int):
    """FR-09: the decision, its time and the auditor; FR-12: approve and reject become
    labelled feedback examples. Only a pending report can be decided, once."""
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])
    decision = request.POST.get("decision")
    if decision not in DECISIONS:
        messages.error(request, "Unknown decision.")
        return redirect("audit:report", pk=pk)
    with transaction.atomic():
        report = get_object_or_404(AuditReport.objects.select_for_update(of=("self",)).select_related("claim", "verdict"), pk=pk)
        if report.status != AuditReport.Status.PENDING:
            messages.warning(request, f"Report #{pk} was already {report.status}.")
            return redirect("audit:queue")
        report.status, report.reviewed_by, report.reviewed_at = decision, request.user, timezone.now()
        report.save(update_fields=["status", "reviewed_by", "reviewed_at"])
        ActivityLog.record(ActivityLog.Action.DECISION, request.user, f"report #{pk}: {decision}")
    if decision in (AuditReport.Status.APPROVED, AuditReport.Status.REJECTED):
        _write_feedback_example(request, report, decision)
    messages.success(request, f"Report #{pk}: {decision}.")
    lane, order = request.POST.get("lane", ""), request.POST.get("order", "score")
    following = _pending(lane, order).first()
    if following:
        return redirect(f"{reverse('audit:report', args=[following.pk])}?lane={lane}&order={order}")
    return redirect(f"{reverse('audit:queue')}?lane={lane}&order={order}")


def _write_feedback_example(request, report: AuditReport, decision: str) -> None:
    try:
        services.feedback_repository().materialize_example(
            claim={"id": report.claim.claim_ref, "clinical_notes": report.claim.clinical_notes,
                   "icd11_code": report.claim.icd11_code},
            verdict={"verdict": report.verdict.verdict if report.verdict else None},
            auditor_decision=decision, icd11_definition=services.code_reference(report.claim.icd11_code))
    except services.ConfigMissing:
        messages.warning(request, "The decision is saved, but no feedback example was written: config.yaml is missing.")


# --- Monitoring (Administrator) ----------------------------------------------------------------

SYSTEM_NAMES = {
    "pipeline_text_lane": "ClaimGuard text lane (stage 1, then the classifier)",
    "zero_shot_llm": "Zero-shot LLM on every claim",
    "zero_shot_llm_text_lane": "Zero-shot LLM in the classifier's place",
    "pipeline": "ClaimGuard pipeline (both lanes)",
    "hybrid_text_rules": "Hybrid: text lane plus five billing rules",
    "rule_based": "Rule-based baseline",
    "isolation_forest": "Isolation Forest alone",
    "zero_shot_llm_funnel": "Pipeline with a zero-shot LLM in the classifier's place",
    "structured_only": "Structured-only model",
}


def _heldout_rows(results: dict) -> list[dict]:
    """One row per system, the reference first; seeded systems (structured-only) are shown
    once as mean ± std over their seeds, with the weakest of their McNemar p-values."""
    reference, mcnemar = results["reference"], results.get("mcnemar", {})
    grouped = {name: g for name, g in (results.get("seed_groups") or {}).items()}
    in_group = {s for g in grouped.values() for s in g["systems"]}
    rows = []
    for name, system in results["systems"].items():
        if name in in_group:
            continue
        m, ci = system["metrics"], system.get("ci95", {})
        rows.append({"name": SYSTEM_NAMES.get(name, name), "reference": name == reference,
                     "precision": m["precision"], "recall": m["recall"], "f1": m["f1"], "fpr": m["fpr"],
                     "f1_ci": ci.get("f1"), "flagged": m["flagged"],
                     "p_value": (mcnemar.get(name) or {}).get("p_value")})
    for name, g in grouped.items():
        rows.append({"name": f"{SYSTEM_NAMES.get(name, name)} ({g['seeds']} seeds, mean ± std)", "reference": False,
                     "precision": g["precision"]["mean"], "recall": g["recall"]["mean"], "f1": g["f1"]["mean"],
                     "f1_std": g["f1"]["std"], "fpr": g["fpr"]["mean"], "flagged": None,
                     "p_value": max((mcnemar.get(s) or {}).get("p_value") or 0 for s in g["systems"]) or None})
    return sorted(rows, key=lambda r: (not r["reference"], -r["f1"]))


@administrator_required
def model_performance(request):
    context = {}
    if _config_or_none(request) is not None:
        path = services.active_model_path()
        metrics = services.read_json(path / "metrics.json")
        context.update(path=path, metrics=metrics, record=services.read_json(path / "config.json") or {})
        evaluation = services.read_json(path / "evaluation_validation.json") if metrics else None
        if evaluation:
            should_flag = {"upcoding", "ghost_billing", "record_fabrication"}
            context["typology_chart"] = charts.share_by_typology(
                [{"typology": services.TYPOLOGY_NAMES.get(t, t), "share": v["rate"], "flagged": v["flagged"],
                  "claims": v["claims"], "should be flagged": "yes" if t in should_flag else "no"}
                 for t, v in evaluation["flagged_by_typology"].items()],
                "Share flagged", color=charts.SHOULD_FLAG_COLOR)
            context["strata"] = [
                ("Clean vs noisy text", {{"False": "Clean", "True": "Noisy"}.get(k, k): v
                                         for k, v in evaluation["text_noise"].items()}),
                ("Common vs rare ICD-11 codes", evaluation["code_frequency"])]
            context["rare_threshold"] = evaluation.get("rare_code_threshold")
        if metrics:
            context["caught"] = round(metrics["recall"] * metrics["positives"])
        text, fraud = services.heldout_evaluation("text"), services.heldout_evaluation("is_fraud")
        if text:
            lane = text["systems"][text["reference"]]
            strata = lane.get("strata", {})
            context["heldout"] = {"run": text, "lane": lane["metrics"], "ci": lane.get("ci95", {}),
                                  "rows": _heldout_rows(text), "strata": strata}
            context["heldout_strata"] = [
                *((f"{'Noisy' if k == 'True' else 'Clean'} text", v) for k, v in strata.get("text_noise", {}).items()),
                *((f"{k.capitalize()} ICD-11 codes", v) for k, v in strata.get("code_frequency", {}).items())]
            if strata.get("flagged_by_typology"):
                should_flag = {"upcoding", "ghost_billing", "record_fabrication"}
                context.setdefault("charts", {})["typology_test"] = charts.share_by_typology(
                    [{"typology": services.TYPOLOGY_NAMES.get(t, t), "share": v["rate"], "flagged": v["flagged"],
                      "claims": v["claims"], "should be flagged": "yes" if t in should_flag else "no"}
                     for t, v in strata["flagged_by_typology"].items()],
                    "Share flagged", color=charts.SHOULD_FLAG_COLOR)
        if fraud:
            context["heldout_fraud"] = {"run": fraud, "rows": _heldout_rows(fraud)}
        context["active_is_evaluated"] = services.active_is_evaluated()
    return render(request, "audit/model.html", context)


@administrator_required
def feedback_loop(request):
    context = {}
    if _config_or_none(request) is not None:
        feedback = services.feedback_repository()
        examples = feedback.read_examples(since_consumed_marker=False)
        trigger = services.retrain_trigger_count()
        unconsumed = feedback.count_unconsumed_examples()
        context.update(examples=examples[-200:][::-1], total=len(examples), unconsumed=unconsumed, trigger=trigger,
                       progress=min(100, round(100 * unconsumed / trigger)) if trigger else 0,
                       due=unconsumed >= trigger,
                       history=[{**row, "checkpoint": str(row.get("candidate_path", "")).rstrip("/").split("/")[-1]}
                                for row in reversed(services.retraining_history())])
    return render(request, "audit/feedback.html", context)

