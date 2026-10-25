"""AuditPipeline orchestration with fake classifier and narrator: checks the funnel's
gating (only flagged claims reach the classifier, only inconsistent verdicts reach the
LLM), the NFR-02 retry-then-log behaviour, and what is persisted."""
import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from src.classification.classifier import ClassifierVerdict
from src.data.generator import GeneratorConfig, generate_base_claims, load_catalogue
from src.icd11.client import ICD11Definition
from src.pipeline import AuditPipeline, evaluate_run, fit_stage1, run_predictions
from src.reporting.repository import AuditReportRepository


class _FakeCache:
    def get(self, code):
        return ICD11Definition(code, f"Title of {code}", "", [], [])


class _FakeClassifier:
    """Calls every other claim it sees inconsistent."""
    def __init__(self):
        self.seen = []

    def predict(self, claim_id, clinical_notes, icd11_definition):
        self.seen.append(claim_id)
        verdict = "inconsistent" if len(self.seen) % 2 else "consistent"
        return ClassifierVerdict(claim_id, verdict, 0.9, "fake-v1")


class _FlakyNarrator:
    def __init__(self, failures_before_success):
        self.failures_left = failures_before_success
        self.calls = 0

    def generate(self, clinical_notes, verdict, icd11_definition):
        self.calls += 1
        assert verdict.verdict == "inconsistent"
        if self.failures_left:
            self.failures_left -= 1
            raise ConnectionError("provider unavailable")
        return "Narrative."


@pytest.fixture(scope="module")
def data():
    claims, _ = generate_base_claims(load_catalogue(), GeneratorConfig(n_claims=400, n_providers=20, seed=3))
    claims["clinical_notes"] = "Patient reviewed."
    claims["is_fraud"] = (claims["is_inflated_csection"] | claims["is_anomalous_dependent"]).astype(int)
    claims["label"] = "consistent"
    claims["fraud_typology"] = "none"
    return claims.iloc[:300], claims.iloc[300:]


@pytest.fixture(scope="module")
def stage1(data):
    return fit_stage1(data[0], contamination=0.1, background_size=5)


def _pipeline(stage1, tmp_path, narrator):
    classifier = _FakeClassifier()
    repository = AuditReportRepository(str(tmp_path / "audit.sqlite"))
    return AuditPipeline(stage1, classifier, _FakeCache(), repository, narrator), classifier, repository


class _FakeScreen:
    """Flags every fifth claim of the batch, whatever its notes say."""
    def flag(self, claims):
        return np.arange(len(claims)) % 5 == 0


class _ConsistentClassifier(_FakeClassifier):
    def predict(self, claim_id, clinical_notes, icd11_definition):
        self.seen.append(claim_id)
        return ClassifierVerdict(claim_id, "consistent", 0.8, "fake-v1")


def test_only_flagged_claims_reach_the_classifier_and_get_reports(stage1, data, tmp_path, monkeypatch):
    monkeypatch.setattr("src.pipeline.time.sleep", lambda s: None)
    pipeline, classifier, repository = _pipeline(stage1, tmp_path, _FlakyNarrator(0))
    log = pipeline.run(data[1])
    flagged = [c for c in log.claims if c["flagged"]]
    assert 0 < len(flagged) < len(data[1])
    assert len(classifier.seen) == len(flagged)
    assert len(repository.get_pending_reports()) == len(flagged)


def test_screen_only_flags_with_consistent_notes_are_not_queued(stage1, data, tmp_path):
    """#46: a screen-only flag the classifier finds consistent is a logged false
    positive; a structured-detector flag with consistent notes is queued in the
    structural-anomaly lane."""
    classifier = _ConsistentClassifier()
    repository = AuditReportRepository(str(tmp_path / "audit.sqlite"))
    pipeline = AuditPipeline(dataclasses.replace(stage1, screen=_FakeScreen()), classifier,
                             _FakeCache(), repository)
    log = pipeline.run(data[1])
    flagged = [c for c in log.claims if c["flagged"]]
    screen_only = [c for c in flagged if c["flag_sources"] == ["text_screen"]]
    structural = [c for c in flagged if set(c["flag_sources"]) & {"isolation_forest", "autoencoder"}]
    assert screen_only and structural
    assert len(classifier.seen) == len(flagged)  # every flag still reaches the classifier
    assert not any(c["reported"] for c in screen_only)
    assert all(c["reported"] and c["lane"] == "structural_anomaly" for c in structural)
    assert len(repository.get_pending_reports()) == len(structural)
    report = repository.get_report(structural[0]["report_id"])
    assert json.loads(report["flag_sources"]) == structural[0]["flag_sources"]
    assert log.summary()["screen_false_positives"] == len(screen_only)


def test_inconsistent_verdicts_go_to_the_text_lane(stage1, data, tmp_path):
    pipeline, _, _ = _pipeline(dataclasses.replace(stage1, screen=_FakeScreen()), tmp_path, None)
    log = pipeline.run(data[1])
    for entry in (c for c in log.claims if c["flagged"]):
        if entry["verdict"] == "inconsistent":
            assert entry["reported"] and entry["lane"] == "text_inconsistent"


def test_run_predictions_shared_format(stage1, data, tmp_path):
    pipeline, _, _ = _pipeline(stage1, tmp_path, None)
    log = pipeline.run(data[1])
    predictions = run_predictions(log)
    full, text = predictions["pipeline"], predictions["pipeline_text_lane"]
    assert len(full) == len(text) == len(data[1])
    assert full["flagged"].sum() == sum(bool(c.get("reported")) for c in log.claims)
    assert text["flagged"].sum() == sum(c.get("verdict") == "inconsistent" for c in log.claims)
    unseen = [not c["flagged"] for c in log.claims]
    assert (text.loc[unseen, "score"] == 0).all()
    # Stage-1 flags, for putting another stage-2 model in the classifier's place (#52).
    assert predictions["pipeline_stage1"]["flagged"].sum() == sum(c["flagged"] for c in log.claims)
    structural = predictions["pipeline_stage1_structural"]["flagged"]
    assert structural.sum() == sum(bool({"isolation_forest", "autoencoder"} & set(c["flag_sources"]))
                                   for c in log.claims)
    # The pipeline's lane rule: every reported claim is text-inconsistent or structurally flagged.
    assert (full["flagged"] == ((text["flagged"] & predictions["pipeline_stage1"]["flagged"]) | structural)).all()


def test_narrative_only_for_inconsistent_verdicts(stage1, data, tmp_path, monkeypatch):
    monkeypatch.setattr("src.pipeline.time.sleep", lambda s: None)
    narrator = _FlakyNarrator(0)
    pipeline, _, repository = _pipeline(stage1, tmp_path, narrator)
    log = pipeline.run(data[1])
    flagged = [c for c in log.claims if c["flagged"]]
    inconsistent = [c for c in flagged if c["verdict"] == "inconsistent"]
    assert narrator.calls == len(inconsistent)
    for entry in flagged:
        report = repository.get_report(entry["report_id"])
        assert (report["llm_narrative"] is not None) == (entry["verdict"] == "inconsistent")
        assert report["shap_attribution"]


def test_narrative_failure_is_retried_then_logged(stage1, data, tmp_path, monkeypatch):
    monkeypatch.setattr("src.pipeline.time.sleep", lambda s: None)
    narrator = _FlakyNarrator(failures_before_success=10**6)
    pipeline, _, repository = _pipeline(stage1, tmp_path, narrator)
    log = pipeline.run(data[1])
    inconsistent = [c for c in log.claims if c.get("verdict") == "inconsistent"]
    assert log.narrative_failures == len(inconsistent)
    assert narrator.calls == 3 * len(inconsistent)
    assert not any(c["narrative_ok"] for c in inconsistent)


def test_evaluate_run_reports_stage1_recall(stage1, data, tmp_path):
    pipeline, _, _ = _pipeline(stage1, tmp_path, None)
    batch = data[1]
    result = evaluate_run(pipeline.run(batch), batch)
    assert result["fraud_claims"] == int(batch["is_fraud"].sum())
    assert 0.0 <= result["stage1_recall"] <= 1.0
    assert result["end_to_end_recall"] <= result["stage1_recall"]


@pytest.mark.parametrize("argv, message", [
    (["--batch", "test", "--db", "x/evaluation.sqlite"], "pass --final"),
    (["--batch", "validation", "--db", "x/sha_fraud.sqlite"], "review database"),
    (["--batch", "test", "--final", "--db", "x/sha_fraud.sqlite"], "review database"),
])
def test_held_out_claims_never_reach_the_review_database(argv, message, monkeypatch, capsys, tmp_path):
    """DR-08: the test split needs --final, and only the pool may write to sha_fraud.sqlite."""
    from src import pipeline
    monkeypatch.setattr("sys.argv", ["pipeline", "--data", str(tmp_path), "--cache", "c.json", *argv])
    with pytest.raises(SystemExit):
        pipeline.main()
    assert message in capsys.readouterr().err


def test_progress_is_reported(stage1, data, tmp_path):
    pipeline, _, _ = _pipeline(stage1, tmp_path, None)
    calls = []
    log = pipeline.run(data[1], progress=lambda stage, done, total: calls.append((stage, done, total)))
    n = len(data[1])
    assert calls[0] == ("explaining", 0, sum(c["flagged"] for c in log.claims))
    assert [c[1] for c in calls[1:]] == list(range(1, n + 1)) and calls[-1] == ("auditing", n, n)
