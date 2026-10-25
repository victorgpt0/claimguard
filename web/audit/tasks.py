"""Background jobs (phase 3). process_batch runs the unchanged AuditPipeline (src/pipeline.py)
over an uploaded batch on whichever worker took it: stage 1 fitted on the configured
dataset's training split, the active classifier, and on the GPU queue the narrative LLM. Its
progress goes to the Job row the batch page polls as it happens (so the run is not one
transaction, which would hide it); a failed run's partial verdicts and reports are removed,
and the failure is recorded on the job and the batch, never lost."""
from __future__ import annotations

import io
import logging
import socket
import traceback
from pathlib import Path

import pandas as pd
from celery import shared_task
from django.utils import timezone

from . import services
from .models import AuditReport, Batch, ClassifierVerdict, Job
from .repository import DjangoAuditRepository

logger = logging.getLogger(__name__)
_STAGE1 = {}   # fitted once per worker process and configuration: fitting takes minutes


def _update(job_id: int, **fields) -> None:
    Job.objects.filter(pk=job_id).update(**fields)


def build_pipeline(config: dict, with_narratives: bool, repository, progress):
    """The pipeline as src.pipeline.main builds it, from config.yaml. Kept apart so tests can
    replace it with a light fake."""
    import torch

    from src.classification.classifier import SemanticConsistencyClassifier
    from src.detection.text_screen import TextConsistencyScreen
    from src.icd11.cache import ICD11Cache
    from src.pipeline import AuditPipeline, fit_stage1

    paths, thresholds = config["paths"], config.get("thresholds") or {}
    dataset = (config.get("pipeline") or {}).get("dataset")
    if not dataset:
        raise ValueError("config.yaml has no pipeline.dataset: the training split to fit stage 1 on")
    cache = ICD11Cache(Path(paths["artifacts_dir"]) / "icd11_cache.json")
    key = (dataset, thresholds.get("isolation_forest_contamination"), thresholds.get("autoencoder_train_quantile"),
           thresholds.get("text_screen_threshold"))
    if key not in _STAGE1:
        progress("fitting stage 1", 0, 0)
        train = pd.read_parquet(Path(paths["data_dir"]) / "processed" / dataset / "train.parquet")
        screen = TextConsistencyScreen(cache, thresholds["text_screen_threshold"]).fit(train)
        _STAGE1[key] = fit_stage1(train, thresholds["isolation_forest_contamination"],
                                  autoencoder_quantile=thresholds.get("autoencoder_train_quantile"), screen=screen)
    progress("loading the classifier", 0, 0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = SemanticConsistencyClassifier(str(Path(paths["model_dir"]) / "classifier_lora" / "active"),
                                               config["models"]["classifier_base_model"], device)
    narrator = None
    if with_narratives:
        from src.narrative.narrative_generator import AuditNarrativeGenerator
        narrator = AuditNarrativeGenerator(config["models"]["llm_provider"], config["models"]["llm_model_name"])
    return AuditPipeline(_STAGE1[key], classifier, cache, repository, narrator)


@shared_task(bind=True)
def process_batch(self, job_id: int) -> str:
    job = Job.objects.select_related("batch").filter(pk=job_id).first()
    if job is None:   # e.g. a message left in the broker by a database that no longer has the job
        logger.warning("Job %s does not exist; message dropped", job_id)
        return "missing"
    batch = job.batch
    _update(job_id, status=Job.Status.RUNNING, started_at=timezone.now(), task_id=self.request.id or "",
            worker=self.request.hostname or socket.gethostname(), stage="starting")
    Batch.objects.filter(pk=batch.pk).update(status=Batch.Status.PROCESSING, error="")

    def progress(stage: str, done: int, total: int) -> None:
        _update(job_id, stage=stage, done=done, total=total)

    try:
        config = services.config()
        claims = pd.read_csv(io.StringIO(batch.source_csv), dtype={"claim_id": str, "provider_id": str, "member_id": str})
        pipeline = build_pipeline(config, job.queue == Job.Queue.GPU, DjangoAuditRepository(batch), progress)
        log = pipeline.run(claims, {"batch": f"upload:{batch.pk}", "job": job_id, "queue": job.queue},
                           progress=progress)
    except Exception as exc:
        logger.exception("Job %s failed", job_id)
        AuditReport.objects.filter(claim__batch=batch).delete()
        ClassifierVerdict.objects.filter(claim__batch=batch).delete()
        message = f"{type(exc).__name__}: {exc}"
        _update(job_id, status=Job.Status.FAILED, finished_at=timezone.now(),
                message=message + "\n\n" + traceback.format_exc(limit=5))
        Batch.objects.filter(pk=batch.pk).update(status=Batch.Status.FAILED, error=message)
        return "failed"
    summary = log.summary()
    Batch.objects.filter(pk=batch.pk).update(
        status=Batch.Status.PROCESSED, processed_at=timezone.now(),
        run_log={"started_at": log.started_at, "settings": log.settings, "summary": summary})
    _update(job_id, status=Job.Status.SUCCEEDED, finished_at=timezone.now(), stage="done",
            message=f"{summary['flagged_stage1']} flagged, {sum(summary['reported'].values())} queued for review, "
                    f"{summary['narratives']} narratives")
    return "succeeded"
