import json
from datetime import date
from decimal import Decimal

import pandas as pd
import pytest
import yaml
from django.test import Client

from audit import services
from audit.models import AuditReport, Batch, Claim, ClassifierVerdict


@pytest.fixture
def pipeline_files(tmp_path, settings):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "icd11_cache.json").write_text(json.dumps({"JB22.1": {
        "code": "JB22.1", "title": "Delivery by emergency caesarean section", "definition": "",
        "inclusion": [], "exclusion": []}, "DB10.02": {
        "code": "DB10.02", "title": "Acute appendicitis", "definition": "", "inclusion": [], "exclusion": []}}))
    config = {"paths": {"artifacts_dir": str(artifacts), "model_dir": str(tmp_path / "models"),
                        "experiments_dir": str(tmp_path / "experiments"), "data_dir": str(tmp_path)},
              "thresholds": {"retrain_trigger_count": 1}}
    processed = tmp_path / "processed" / "v0.6-20k"
    processed.mkdir(parents=True)
    pd.DataFrame({"claim_id": ["CLM_TEST_1"]}).to_parquet(processed / "test.parquet")
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    settings.CLAIMGUARD_CONFIG = tmp_path / "config.yaml"
    services._cache.cache_clear()
    return tmp_path


@pytest.fixture
def reports(db):
    batch = Batch.objects.create(name="pool", status=Batch.Status.PROCESSED)
    made = []
    for ref, verdict, narrative, score in (("CLM1", "inconsistent", "Notes describe a forceps delivery.", 0.6),
                                           ("CLM2", "consistent", None, 0.8)):
        claim = Claim.objects.create(batch=batch, claim_ref=ref, provider_id="PRV001", patient_ref="M1",
                                     icd11_code="JB22.1", claim_amount=Decimal("30000"),
                                     service_date=date(2026, 1, 1), clinical_notes="Forceps-assisted vaginal delivery.",
                                     anomaly_score=score)
        v = ClassifierVerdict.objects.create(claim=claim, verdict=verdict, confidence=0.9, model_version="seed7")
        made.append(AuditReport.objects.create(claim=claim, verdict=v, llm_narrative=narrative,
                                               shap_attribution={"claim_amount": 0.2, "stay_days": -0.1},
                                               flag_sources=["isolation_forest", "text_screen"]))
    return made


def _client(user, password):
    client = Client()
    assert client.login(username=user.username, password=password)
    return client


@pytest.fixture
def auditor(make_user):
    return _client(make_user(), "auditor-pass-1")


@pytest.fixture
def admin(make_user):
    return _client(make_user("admin", role="Administrator", password="admin-pass-12"), "admin-pass-12")


