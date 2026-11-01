"""The web application's view of the ML side (src/): the pipeline's config.yaml (paths and
thresholds, FR-10), the ICD-11 cache, the Feedback Dataset, the active checkpoint's metrics,
pipeline run logs and the retraining history. Nothing here loads a model."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from django.conf import settings

from src.classification.export import reference_text
from src.config import load_config
from src.feedback.feedback_repository import FeedbackRepository
from src.feedback.run_retraining import history_path, read_history
from src.icd11.cache import ICD11Cache

TYPOLOGY_NAMES = {
    "none": "Clean", "upcoding": "Upcoding", "ghost_billing": "Ghost billing",
    "record_fabrication": "Record fabrication", "outpatient_to_inpatient": "Outpatient conversion",
    "anomalous_dependents": "Anomalous dependents", "inflated_caesarean": "Inflated caesarean",
}
SOURCE_NAMES = {"isolation_forest": "Isolation Forest", "autoencoder": "autoencoder",
                "text_screen": "text-consistency screen"}


class ConfigMissing(Exception):
    pass


def config() -> dict:
    try:
        return load_config(settings.CLAIMGUARD_CONFIG)
    except FileNotFoundError as exc:
        raise ConfigMissing(str(exc)) from None


def read_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@lru_cache(maxsize=4)
def _cache(path: str) -> ICD11Cache:
    return ICD11Cache(path)


def icd11_definition(code: str):
    return _cache(str(Path(config()["paths"]["artifacts_dir"]) / "icd11_cache.json")).get(code)


def code_reference(code: str) -> str:
    """The same reference text the classifier judged the claim against (export.py)."""
    return reference_text(icd11_definition(code))


def feedback_repository() -> FeedbackRepository:
    return FeedbackRepository(str(Path(config()["paths"]["artifacts_dir"]) / "feedback_dataset" / "examples.jsonl"))


def active_model_path() -> Path:
    return Path(config()["paths"]["model_dir"]) / "classifier_lora" / "active"


def latest_pipeline_run() -> dict | None:
    """The newest run on the Deployment/Demonstration Pool, the only split whose reports reach
    the review queue (DR-08). Evaluation runs on validation or test are left out: their reports
    are not in this database, so their funnel would misdescribe it."""
    experiments = config()["paths"].get("experiments_dir")
    if not experiments:
        return None
    for path in sorted((Path(experiments) / "pipeline_runs").glob("*.json"), reverse=True):
        run = read_json(path)
        if (run.get("settings") or {}).get("batch", "deployment_pool") == "deployment_pool":
            return run
    return None


def retraining_history() -> list[dict]:
    return read_history(history_path(config()))


def retrain_trigger_count() -> int:
    return int(config()["thresholds"]["retrain_trigger_count"])


def known_codes() -> set[str]:
    """Codes the ICD-11 cache holds a reference definition for: an upload's codes must be
    among them, since the classifier judges notes against that definition."""
    data = json.loads((Path(config()["paths"]["artifacts_dir"]) / "icd11_cache.json").read_text(encoding="utf-8"))
    return set(data)


@lru_cache(maxsize=8)
def _test_split_ids(path: str, mtime: float) -> frozenset[str]:
    import pandas as pd
    return frozenset(pd.read_parquet(path, columns=["claim_id"])["claim_id"].astype(str))


def heldout_ids() -> set[str]:
    """Claim ids of the held-out test split (DR-08) of the dataset the models are evaluated on
    (config.yaml pipeline.dataset); of every dataset version when that is not set. Claim ids
    are reused across versions, so checking all of them would refuse claims that only share
    an id with an older version's test split. Raises ConfigMissing when the split is not
    reachable: an upload is refused rather than left unchecked."""
    cfg = config()
    processed = Path(cfg["paths"]["data_dir"]) / "processed"
    dataset = (cfg.get("pipeline") or {}).get("dataset")
    paths = [processed / dataset / "test.parquet"] if dataset else sorted(processed.glob("*/test.parquet"))
    if not paths or not all(path.exists() for path in paths):
        raise ConfigMissing(f"The test split under {processed} is not reachable, so uploads cannot be checked against it.")
    ids: set[str] = set()
    for path in paths:
        ids |= _test_split_ids(str(path), path.stat().st_mtime)
    return ids


def workers_online() -> dict[str, list[str]]:
    """Which queues have a live worker right now ({"cpu": [...], "gpu": [...]}), from Celery's
    own broadcast; empty when the broker is unreachable."""
    from claimguard_web.celery import app
    try:
        active = app.control.inspect(timeout=1.0).active_queues() or {}
    except Exception:
        return {}
    online: dict[str, list[str]] = {}
    for worker, queues in active.items():
        for queue in queues:
            online.setdefault(queue["name"], []).append(worker)
    return online
