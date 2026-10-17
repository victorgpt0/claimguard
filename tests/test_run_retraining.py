import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.feedback.feedback_repository import FeedbackRepository
from src.feedback.retrainer import ClassifierRetrainer
from src.feedback.run_retraining import (
    classifier_dir,
    feedback_path,
    history_path,
    read_history,
    run,
    simulate_decisions,
)
from src.icd11.cache import ICD11Cache
from src.icd11.client import ICD11Definition
from src.reporting.repository import AuditReportRepository


def _example(i, label="consistent"):
    return {"claim_id": f"CLM{i:06d}", "icd11_code": "1A00", "clinical_notes": f"notes {i}",
            "icd11_definition": "Cholera", "label": label}


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.fixture
def config(tmp_path):
    return {"paths": {"artifacts_dir": str(tmp_path / "artifacts"),
                      "model_dir": str(tmp_path / "models"),
                      "experiments_dir": str(tmp_path / "experiments")},
            "models": {"classifier_base_model": "base-model"}}


@pytest.fixture
def classifier_data(tmp_path):
    data = tmp_path / "classifier"
    _write_jsonl(data / "train.jsonl", [_example(i) for i in range(3)])
    _write_jsonl(data / "validation.jsonl", [_example(10)])
    _write_jsonl(data / "test.jsonl", [_example(20, "inconsistent")])
    return data


def _add_feedback(config, n, simulated=False):
    repo = FeedbackRepository(str(feedback_path(config)))
    for i in range(n):
        repo.materialize_example({"id": 100 + i, "clinical_notes": f"fb {i}", "icd11_code": "1A00"},
                                 {"verdict": "inconsistent"}, "rejected", "Cholera",
                                 extra={"simulated": True} if simulated else None)
    return repo


class _FakeTrainer:
    """Stands in for train_one_seed: records its inputs and writes a fake adapter."""

    def __init__(self, repo=None, arriving=0):
        self.calls = []
        self.repo, self.arriving = repo, arriving

    def __call__(self, seed, train, validation, base_model, output_dir, **kwargs):
        self.calls.append({"seed": seed, "train": train, "validation": validation,
                           "base_model": base_model, "output_dir": output_dir})
        (output_dir / "adapter_model.bin").write_text("weights")
        # Feedback recorded by the dashboard while the run trains.
        for i in range(self.arriving):
            self.repo.materialize_example({"id": 900 + i, "clinical_notes": "late", "icd11_code": "1A00"},
                                          {"verdict": "consistent"}, "rejected", "Cholera")
        return {"f1": 0.7, "recall": 0.8, "epoch": 2}


def _metrics(*results):
    calls = iter(results)
    return patch.object(ClassifierRetrainer, "_evaluate_on_test_set",
                        side_effect=lambda *a, **k: next(calls))


def test_run_trains_on_split_plus_feedback_and_promotes_first_candidate(config, classifier_data):
    _add_feedback(config, 2, simulated=True)
    trainer = _FakeTrainer()
    with patch("src.classification.classifier.SemanticConsistencyClassifier"), \
         _metrics({"f1": 0.6, "recall": 0.9}):
        row = run(config, classifier_data, seed=7, trainer=trainer)

    call = trainer.calls[0]
    assert call["seed"] == 7 and call["base_model"] == "base-model"
    assert [e["clinical_notes"] for e in call["train"]] == ["notes 0", "notes 1", "notes 2", "fb 0", "fb 1"]
    assert call["output_dir"].parent.name == "candidates"
    assert row["feedback_examples"] == 2 and row["simulated_examples"] == 2
    assert row["promoted"] is True
    assert row["active_metrics"] is None  # nothing was deployed before
    assert row["candidate_metrics"] == {"f1": 0.6, "recall": 0.9}
    active = classifier_dir(config) / "active"
    assert (active / "adapter_model.bin").exists()
    assert json.loads((active / "metrics.json").read_text())["f1"] == 0.7
    assert read_history(history_path(config)) == [row]


def test_run_records_rejected_candidate_and_keeps_active(config, classifier_data):
    active = classifier_dir(config) / "active"
    active.mkdir(parents=True)
    (active / "adapter_model.bin").write_text("active weights")
    _add_feedback(config, 1)
    with patch("src.classification.classifier.SemanticConsistencyClassifier"), \
         _metrics({"f1": 0.5, "recall": 0.6}, {"f1": 0.9, "recall": 0.9}):
        row = run(config, classifier_data, trainer=_FakeTrainer())

    assert row["promoted"] is False
    assert row["candidate_metrics"] == {"f1": 0.5, "recall": 0.6}
    assert row["active_metrics"] == {"f1": 0.9, "recall": 0.9}
    assert (active / "adapter_model.bin").read_text() == "active weights"
    assert (Path(row["candidate_path"]) / "adapter_model.bin").exists()  # kept for inspection


def test_run_marks_consumed_at_start_but_keeps_examples_arriving_mid_run(config, classifier_data):
    repo = _add_feedback(config, 3)
    trainer = _FakeTrainer(repo=repo, arriving=2)
    with patch("src.classification.classifier.SemanticConsistencyClassifier"), \
         _metrics({"f1": 0.6, "recall": 0.9}):
        run(config, classifier_data, trainer=trainer)

    assert repo.count_unconsumed_examples() == 2  # the two late ones wait for the next run
    assert [e["clinical_notes"] for e in repo.read_examples()] == ["late", "late"]


def test_run_marks_consumed_even_when_training_fails(config, classifier_data):
    repo = _add_feedback(config, 2)

    def failing_trainer(*args, **kwargs):
        raise RuntimeError("CUDA out of memory")

    with pytest.raises(RuntimeError):
        run(config, classifier_data, trainer=failing_trainer)
    assert repo.count_unconsumed_examples() == 0
    assert read_history(history_path(config)) == []


def test_run_without_feedback_does_nothing(config, classifier_data):
    trainer = _FakeTrainer()
    assert run(config, classifier_data, trainer=trainer) is None
    assert trainer.calls == []


def test_read_history_is_empty_before_first_run(tmp_path):
    assert read_history(tmp_path / "missing.jsonl") == []


# --- simulated auditor ----------------------------------------------------------------

def _pool_claim(claim_id, notes, is_fraud, code="1A00"):
    return {"claim_id": claim_id, "provider_id": "PRV0001", "member_id": f"MBR{claim_id}",
            "icd11_code": code, "claim_amount": 0.0, "service_date": "2025-05-01",
            "discharge_date": "2025-05-01", "clinical_notes": notes, "is_fraud": is_fraud}


def _save_report(repository, claim, verdict):
    db_claim = repository.save_claim({
        "provider_id": claim["provider_id"], "patient_ref": claim["member_id"],
        "icd11_code": claim["icd11_code"].strip().upper(), "claim_amount": claim["claim_amount"],
        "service_date": claim["service_date"], "discharge_date": claim["discharge_date"],
        "clinical_notes": claim["clinical_notes"], "anomaly_score": 0.5})
    verdict_id = repository.save_verdict(db_claim, verdict, 0.9, "test")
    return repository.save_report(db_claim, verdict_id, {}, None)


def test_simulate_decisions_decides_from_ground_truth(tmp_path):
    data = tmp_path / "processed"
    data.mkdir()
    claims = [_pool_claim("C1", "fraud notes", 1), _pool_claim("C2", "clean notes", 0, code=" 1a00"),
              _pool_claim("C3", "fraud but consistent", 1), _pool_claim("C4", "not decided", 0)]
    pd.DataFrame(claims).to_parquet(data / "deployment_pool.parquet", index=False)

    db = str(tmp_path / "audit.sqlite")
    repository = AuditReportRepository(db)
    ids = [_save_report(repository, claims[0], "inconsistent"),
           _save_report(repository, claims[1], "inconsistent"),
           _save_report(repository, claims[2], "consistent"),
           _save_report(repository, claims[3], "consistent")]
    unmatched = _save_report(repository, _pool_claim("CX", "not in the split", 0), "consistent")

    cache = ICD11Cache(tmp_path / "icd11_cache.json")
    cache._entries["1A00"] = ICD11Definition("1A00", "Cholera", "An infection.", [], [])
    cache.save()
    feedback_file = tmp_path / "feedback" / "examples.jsonl"

    # Order is report id: the four matched ones come first, so n=3 leaves C4 pending.
    counts = simulate_decisions(db, data, "deployment_pool", 3, feedback_file, cache.cache_path)

    assert counts == {"decided": 3, "approved": 2, "rejected": 1, "examples": 2, "unmatched": 0}
    statuses = {r["id"]: r["status"] for r in map(repository.get_report, [*ids, unmatched])}
    assert statuses == {ids[0]: "approved", ids[1]: "rejected", ids[2]: "approved",
                        ids[3]: "pending", unmatched: "pending"}
    examples = FeedbackRepository(str(feedback_file)).read_examples(since_consumed_marker=False)
    # Approving a consistent verdict yields no example (feedback_label).
    assert [(e["claim_ref"], e["label"]) for e in examples] == [("C1", "inconsistent"), ("C2", "consistent")]
    assert all(e["simulated"] is True for e in examples)
    assert examples[0]["icd11_definition"] == "Cholera. An infection."


def test_simulate_decisions_skips_reports_matching_no_claim(tmp_path):
    data = tmp_path / "processed"
    data.mkdir()
    pd.DataFrame([_pool_claim("C1", "notes", 0)]).to_parquet(data / "deployment_pool.parquet", index=False)
    db = str(tmp_path / "audit.sqlite")
    repository = AuditReportRepository(db)
    _save_report(repository, _pool_claim("CX", "other notes", 0), "consistent")

    counts = simulate_decisions(db, data, "deployment_pool", 5, tmp_path / "fb.jsonl",
                                tmp_path / "icd11_cache.json")
    assert counts["decided"] == 0 and counts["unmatched"] == 1
    assert len(repository.get_pending_reports()) == 1
