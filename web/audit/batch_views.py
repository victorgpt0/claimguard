"""Batch input and processing (FR-01, FR-15, phase 3): the Administrator uploads a claims CSV,
which is validated in full and stored only if every row passes, then starts a job on the
local CPU worker or the Colab GPU worker and watches its progress. Every page here is the
Administrator's: the Fraud Auditor reviews the results in the queue and never sees batches
(separation of duties)."""
from __future__ import annotations

import hashlib
import io

import pandas as pd
from django import forms
from django.contrib import messages
from django.db import transaction
from django.http import HttpResponse, HttpResponseNotAllowed
from django.shortcuts import get_object_or_404, redirect, render
from django.template.loader import render_to_string
from django.utils import timezone
from django.views.decorators.http import require_GET

from accounts.roles import administrator_required
from activity.models import ActivityLog

from . import charts, services
from .models import Batch, Claim, Job
from .summary import TOP, batch_summary, funnel_counts, write_reports_csv
from .tasks import process_batch
from .validation import UPLOAD_COLUMNS, validate

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
TEMPLATE_ROWS = [
    ["CLM900001", "PRV0001", "MBR000001", "principal", "34", "female", "4", "DB10.02", "67200",
     "2026-01-10", "2026-01-13", "Presented with right lower quadrant pain and fever; laparoscopic appendicectomy done."],
    ["CLM900002", "PRV0001", "MBR000002", "dependent", "6", "male", "3", "CA07.0", "0",
     "2026-01-11", "", "Child with cough and runny nose for three days, no fever, advised fluids and rest."],
]


class UploadForm(forms.Form):
    file = forms.FileField(label="Claims CSV")
    synthetic = forms.BooleanField(label="This file contains synthetic data only, no real patient or provider data (DR-06).")

    def clean_file(self):
        upload = self.cleaned_data["file"]
        if not upload.name.lower().endswith(".csv"):
            raise forms.ValidationError("Upload a .csv file.")
        if upload.size > MAX_UPLOAD_BYTES:
            raise forms.ValidationError("The file is larger than 10 MB.")
        return upload


@administrator_required
def batch_list(request):
    batches = Batch.objects.select_related("uploaded_by").prefetch_related("jobs")[:100]
    return render(request, "audit/batches.html", {"batches": batches})


@administrator_required
def upload(request):
    form = UploadForm(request.POST or None, request.FILES or None)
    problems = None
    if request.method == "POST" and form.is_valid():
        raw = form.cleaned_data["file"].read()
        try:
            frame = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False)
            raw.decode("utf-8")
        except (pd.errors.ParserError, UnicodeDecodeError, pd.errors.EmptyDataError) as exc:
            form.add_error("file", f"Not a readable CSV file: {exc}")
        else:
            try:
                problems = validate(frame, services.known_codes(), set(Claim.objects.values_list("claim_ref", flat=True)),
                                    services.heldout_ids())
            except (services.ConfigMissing, FileNotFoundError) as exc:
                form.add_error(None, f"The upload cannot be checked: {exc}")
            else:
                name = form.cleaned_data["file"].name
                if problems:
                    ActivityLog.record(ActivityLog.Action.UPLOAD, request.user, f"rejected {name}: {len(problems)} problems")
                else:
                    batch = _store(frame, raw, name, request.user)
                    ActivityLog.record(ActivityLog.Action.UPLOAD, request.user, f"batch #{batch.pk}: {name}, {len(frame)} claims")
                    messages.success(request, f"Stored {len(frame)} claims as batch #{batch.pk}. Start processing below.")
                    return redirect("audit:batch", pk=batch.pk)
    return render(request, "audit/upload.html", {"form": form, "problems": problems, "columns": UPLOAD_COLUMNS,
                                                 "shown": (problems or [])[:200]})


def _number(value):
    return None if str(value).strip() == "" else float(value)


@transaction.atomic
def _store(frame: pd.DataFrame, raw: bytes, name: str, user) -> Batch:
    batch = Batch.objects.create(name=name, uploaded_by=user, claims_count=len(frame),
                                 source_csv=raw.decode("utf-8"), source_sha256=hashlib.sha256(raw).hexdigest())
    Claim.objects.bulk_create([Claim(
        batch=batch, claim_ref=row["claim_id"].strip(), provider_id=row["provider_id"].strip(),
        patient_ref=row["member_id"].strip(), beneficiary_type=row["beneficiary_type"].strip(),
        patient_age=_number(row["patient_age"]), patient_sex=row["patient_sex"].strip(),
        facility_level=int(float(row["facility_level"])), icd11_code=row["icd11_code"].strip(),
        claim_amount=row["claim_amount"].strip(), service_date=pd.Timestamp(row["service_date"]).date(),
        discharge_date=pd.Timestamp(row["discharge_date"]).date() if row["discharge_date"].strip() else None,
        clinical_notes=row["clinical_notes"]) for row in frame.to_dict(orient="records")])
    return batch


@require_GET
@administrator_required
def upload_template(request):
    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = 'attachment; filename="claims_template.csv"'
    pd.DataFrame(TEMPLATE_ROWS, columns=UPLOAD_COLUMNS).to_csv(response, index=False)
    return response


@administrator_required
def batch_detail(request, pk: int):
    batch = get_object_or_404(Batch.objects.select_related("uploaded_by"), pk=pk)
    job = batch.jobs.first()
    reports = batch.claims.filter(reports__isnull=False).count()
    context = {"batch": batch, "job": job, "reports": reports, "queues": Job.Queue.choices,
               "can_start": request.user.is_auditor and batch.status in (Batch.Status.UPLOADED, Batch.Status.FAILED)
               and not (job and job.active), "workers": services.workers_online()}
    if _summarised(batch):
        summary = batch_summary(batch)
        context.update(s=summary, top=TOP, amount_claimed=_kes(summary["amount_claimed"]),
                       amount_under_audit=_kes(summary["amount_under_audit"]),
                       chart_specs={"chart-funnel": charts.funnel(funnel_counts(summary))})
    return render(request, "audit/batch.html", context)


def _summarised(batch: Batch) -> bool:
    return batch.status in (Batch.Status.PROCESSED, Batch.Status.IMPORTED)


def _kes(amount) -> str:
    return f"KES {float(amount or 0):,.0f}"


def _filename(batch: Batch, suffix: str) -> str:
    return f"batch_{batch.pk}_{timezone.localtime():%Y%m%d}_{suffix}"


@require_GET
@administrator_required
def batch_summary_markdown(request, pk: int):
    """FR-16: the summary as a readable Markdown report."""
    batch = get_object_or_404(Batch.objects.select_related("uploaded_by"), pk=pk)
    if not _summarised(batch):
        messages.warning(request, "A summary exists once the batch is processed.")
        return redirect("audit:batch", pk=pk)
    summary = batch_summary(batch)
    text = render_to_string("audit/batch_summary.md", {
        "batch": batch, "s": summary, "top": TOP, "user": request.user, "generated": timezone.localtime(),
        "amount_claimed": _kes(summary["amount_claimed"]), "amount_under_audit": _kes(summary["amount_under_audit"])})
    response = HttpResponse(text, content_type="text/markdown; charset=utf-8")
    response["Content-Disposition"] = f'attachment; filename="{_filename(batch, "summary.md")}"'
    ActivityLog.record(ActivityLog.Action.DOWNLOAD, request.user, f"batch #{pk}: summary (Markdown)")
    return response


@require_GET
@administrator_required
def batch_reports_csv(request, pk: int):
    """FR-16: the batch's audit reports, one row each, with verdict, SHAP top 3 and decision."""
    batch = get_object_or_404(Batch, pk=pk)
    if not _summarised(batch):
        messages.warning(request, "Reports exist once the batch is processed.")
        return redirect("audit:batch", pk=pk)
    response = HttpResponse(content_type="text/csv; charset=utf-8")
    response["Content-Disposition"] = f'attachment; filename="{_filename(batch, "reports.csv")}"'
    n = write_reports_csv(batch, response)
    ActivityLog.record(ActivityLog.Action.DOWNLOAD, request.user, f"batch #{pk}: {n} reports (CSV)")
    return response


@administrator_required
def start_job(request, pk: int):
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])
    queue = request.POST.get("queue")
    if queue not in Job.Queue.values:
        messages.error(request, "Choose where to run the job.")
        return redirect("audit:batch", pk=pk)
    with transaction.atomic():
        batch = get_object_or_404(Batch.objects.select_for_update(), pk=pk)
        if batch.status not in (Batch.Status.UPLOADED, Batch.Status.FAILED) or batch.jobs.filter(
                status__in=[Job.Status.QUEUED, Job.Status.RUNNING]).exists():
            messages.warning(request, "This batch is already processed or being processed.")
            return redirect("audit:batch", pk=pk)
        job = Job.objects.create(batch=batch, queue=queue, created_by=request.user)
        ActivityLog.record(ActivityLog.Action.JOB_STARTED, request.user, f"job #{job.pk}: batch #{pk} on {queue}")
    transaction.on_commit(lambda: process_batch.apply_async(args=[job.pk], queue=queue))
    if not services.workers_online().get(queue):
        messages.warning(request, f"No {queue.upper()} worker is online: the job waits in the queue until one starts.")
    return redirect("audit:batch", pk=pk)


@administrator_required
def job_progress(request, pk: int):
    """The progress fragment the batch page polls (HTMX). Status 286 tells HTMX to stop
    polling once the job has finished."""
    job = get_object_or_404(Job, pk=pk)
    response = render(request, "audit/job_progress.html", {"job": job})
    if not job.active:
        response.status_code = 286
    return response
