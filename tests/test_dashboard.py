"""Drives dashboard/app.py headless (Streamlit AppTest): the overview statistics, the
review queue and its decisions, and what reaches SQLite and the Feedback Dataset."""
import json
import sqlite3
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.reporting.repository import AuditReportRepository

APP = str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py")


@pytest.fixture
def config(tmp_path, monkeypatch):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "icd11_cache.json").write_text(json.dumps({
        "JB22.1": {"code": "JB22.1", "title": "Delivery by emergency caesarean section",
                   "definition": "", "inclusion": [], "exclusion": []}}))
    db = str(tmp_path / "audit.sqlite")
    repository = AuditReportRepository(db)
    reports = (("inconsistent", "Notes describe a forceps delivery.", 0.6),
               ("consistent", None, 0.8))
    for verdict, narrative, score in reports:
        claim_id = repository.save_claim({
            "provider_id": "PRV001", "patient_ref": "M1", "icd11_code": "JB22.1",
            "claim_amount": 30000.0, "service_date": "2026-01-01",
            "clinical_notes": "Forceps-assisted vaginal delivery.", "anomaly_score": score})
        verdict_id = repository.save_verdict(claim_id, verdict, 0.9, "test")
        repository.save_report(claim_id, verdict_id, {"claim_amount": 0.2, "stay_days": -0.1}, narrative,
                               ["isolation_forest", "text_screen"])

    runs = tmp_path / "experiments" / "pipeline_runs"
    runs.mkdir(parents=True)
    (runs / "run.json").write_text(json.dumps({
        "started_at": "2026-10-12T08:00:00", "settings": {},
        "summary": {"claims": 500, "flagged_stage1": 44, "inconsistent": 1, "narratives": 1,
                    "narrative_failures": 0},
        "evaluation": {"stage1_recall": 0.35, "inconsistent_caught": 1, "inconsistent_total": 25,
                       "flagged_by_typology": {"upcoding": {"size": 8, "sum": 1}}}}))

    config = {"paths": {"db_path": db, "artifacts_dir": str(artifacts),
                        "model_dir": str(tmp_path / "models"),
                        "experiments_dir": str(tmp_path / "experiments")},
              "thresholds": {"retrain_trigger_count": 50}}
    monkeypatch.setattr("src.config.load_config", lambda *a, **k: config)
    st.cache_resource.clear()  # get_config is cached across AppTest runs in one process
    return config


def _app():
    return AppTest.from_file(APP, default_timeout=30).run()


def _report_picker(app):
    return next(s for s in app.selectbox if s.label.startswith("Report"))


def test_opens_without_sign_in_and_shows_statistics(config):
    app = _app()
    assert not app.exception
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Flagged claims"] == "2"
    assert metrics["Pending review"] == "2"
    assert metrics["Inconsistent verdicts"] == "1"
    assert metrics["Amount under audit"] == "KES 60,000"
    assert any("batch of 500 claims" in c.value for c in app.caption)
    # No checkpoint yet: the model tab says so instead of failing.
    assert any("No classifier checkpoint" in w.value for w in app.warning)


def test_review_queue_orders_and_filters(config):
    app = _app()
    assert _report_picker(app).options[0].startswith("#2")  # highest anomaly score first
    app = next(s for s in app.selectbox if s.label == "Lane").select("Text inconsistent").run()
    assert [o[:2] for o in _report_picker(app).options] == ["#1"]
    markdown = " ".join(m.value for m in app.markdown)
    assert "Notes describe a forceps delivery." in markdown  # panel 6, the narrative
    assert "Delivery by emergency caesarean section" in markdown  # panel 3, the ICD-11 title
    assert "Flagged by: Isolation Forest, text-consistency screen" in markdown  # panel 4
    assert app.text[0].value == "Forceps-assisted vaginal delivery."  # panel 2, the notes
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Service date"] == "2026-01-01"
    assert metrics["Discharge date"] == "— (outpatient)"


def test_consistent_verdict_explains_the_missing_narrative(config):
    app = _app()
    app = next(s for s in app.selectbox if s.label == "Lane").select("Structural anomaly").run()
    assert any("No narrative: the classifier found the notes consistent" in i.value for i in app.info)


def test_approving_a_report_records_it_and_writes_feedback(config):
    app = _app()
    app = next(s for s in app.selectbox if s.label == "Lane").select("Text inconsistent").run()
    app = next(b for b in app.button if b.label == "Approved").click().run()
    with sqlite3.connect(config["paths"]["db_path"]) as conn:
        assert conn.execute("SELECT status FROM AuditReport WHERE id = 1").fetchone()[0] == "approved"
    assert {m.label: m.value for m in app.metric}["Pending review"] == "1"
    feedback = Path(config["paths"]["artifacts_dir"]) / "feedback_dataset" / "examples.jsonl"
    example = json.loads(feedback.read_text())
    assert example["label"] == "inconsistent"
    assert example["icd11_definition"] == "Delivery by emergency caesarean section"
    assert {m.label: m.value for m in app.metric}["Feedback examples"] == "1"


def test_retraining_notice_and_history_on_the_feedback_tab(config):
    """#49: the notice is on the feedback tab, not shown to the deciding auditor; #47: the
    history lists every retraining run."""
    config["thresholds"]["retrain_trigger_count"] = 1
    history = Path(config["paths"]["experiments_dir"]) / "retraining_history.jsonl"
    history.write_text(json.dumps({
        "run_at": "2026-10-20T09:30:00+00:00", "feedback_examples": 52, "simulated_examples": 40,
        "candidate_path": "models/classifier_lora/candidates/20261020T093000Z",
        "candidate_metrics": {"f1": 0.95, "recall": 0.97}, "active_metrics": {"f1": 0.94, "recall": 0.97},
        "promoted": True}) + "\n")
    app = _app()
    assert not any("Retraining due" in w.value for w in app.warning)
    app = next(s for s in app.selectbox if s.label == "Lane").select("Text inconsistent").run()
    app = next(b for b in app.button if b.label == "Approved").click().run()
    assert not any("Retraining" in t.value for t in app.toast)  # the auditor is not told
    assert any("Retraining due" in w.value for w in app.warning)  # the feedback tab is
    table = next(d.value for d in app.dataframe if "promoted" in d.value.columns)
    assert table.iloc[0]["checkpoint"] == "20261020T093000Z"
    assert table.iloc[0]["promoted"] == "yes"
