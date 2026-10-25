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
