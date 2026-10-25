"""Drives dashboard/app.py headless (Streamlit AppTest): sign-in and role-based tabs, the
overview statistics, the review queue and its decisions, and what reaches SQLite and the
Feedback Dataset."""
import json
import sqlite3
import time
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.auth.user import User
from src.db.seed import seed_user
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
    seed_user(db, "auditor", "auditor-pass-1", "Auditor")  # id 1
    seed_user(db, "admin", "admin-pass-12", "Administrator")  # id 2
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


AUDITOR = User(id=1, username="auditor", role="Auditor")
ADMIN = User(id=2, username="admin", role="Administrator")


def _app(user: User | None = AUDITOR, last_active: float | None = None):
    """The app with `user` already signed in (None: nobody signed in)."""
    app = AppTest.from_file(APP, default_timeout=30)
    if user is not None:
        app.session_state["user"] = user
        app.session_state["last_active"] = time.time() if last_active is None else last_active
    return app.run()


def _tab_labels(app):
    return [t.label for t in app.tabs]


def _sign_in(app, username, password):
    app.text_input[0].input(username)
    app.text_input[1].input(password)
    return app.button[0].click().run()


def _report_picker(app):
    return next(s for s in app.selectbox if s.label.startswith("Report"))


def test_nothing_but_the_sign_in_form_without_a_session(config):
    app = _app(user=None)
    assert not app.exception
    assert not app.tabs and not app.metric
    assert [t.label for t in app.text_input] == ["Username", "Password"]


def test_wrong_password_and_unknown_user_get_the_same_message(config):
    for username, password in (("auditor", "wrong-password"), ("nobody", "auditor-pass-1")):
        app = _sign_in(_app(user=None), username, password)
        assert [e.value for e in app.error] == ["Incorrect username or password."]
        assert not app.tabs


def test_signing_in_opens_the_role_tabs_and_signing_out_closes_them(config):
    app = _sign_in(_app(user=None), "auditor", "auditor-pass-1")
    assert not app.exception
    assert _tab_labels(app) == ["Overview", "Review queue"]
    app = next(b for b in app.sidebar.button if b.label == "Sign out").click().run()
    assert not app.tabs
    assert "user" not in app.session_state


def test_each_role_sees_only_its_own_tabs(config):
    assert _tab_labels(_app(AUDITOR)) == ["Overview", "Review queue"]
    admin = _app(ADMIN)
    assert _tab_labels(admin) == ["Overview", "Model performance", "Feedback loop"]
    assert not any(b.label == "Approved" for b in admin.button)  # no decisions for the admin


def test_an_idle_session_expires(config):
    app = _app(AUDITOR, last_active=time.time() - 31 * 60)
    assert not app.tabs
    assert any("session expired" in i.value for i in app.info)


def test_auditor_sees_statistics(config):
    app = _app()
    assert not app.exception
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Flagged claims"] == "2"
    assert metrics["Pending review"] == "2"
    assert metrics["Inconsistent verdicts"] == "1"
    assert metrics["Amount under audit"] == "KES 60,000"
    assert any("batch of 500 claims" in c.value for c in app.caption)


def test_admin_model_tab_without_a_checkpoint(config):
    # No checkpoint yet: the model tab says so instead of failing.
    assert any("No classifier checkpoint" in w.value for w in _app(ADMIN).warning)


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
        status, reviewer = conn.execute("SELECT status, reviewed_by FROM AuditReport WHERE id = 1").fetchone()
    assert (status, reviewer) == ("approved", AUDITOR.id)  # FR-09: the reviewer's identity
    assert {m.label: m.value for m in app.metric}["Pending review"] == "1"
    feedback = Path(config["paths"]["artifacts_dir"]) / "feedback_dataset" / "examples.jsonl"
    example = json.loads(feedback.read_text())
    assert example["label"] == "inconsistent"
    assert example["icd11_definition"] == "Delivery by emergency caesarean section"
    assert {m.label: m.value for m in _app(ADMIN).metric}["Feedback examples"] == "1"


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
    assert not any("Retraining due" in w.value for w in _app(ADMIN).warning)
    app = _app(AUDITOR)
    app = next(s for s in app.selectbox if s.label == "Lane").select("Text inconsistent").run()
    app = next(b for b in app.button if b.label == "Approved").click().run()
    assert not any("Retraining" in t.value for t in app.toast)  # the auditor is not told
    assert not any("Retraining" in w.value for w in app.warning)
    app = _app(ADMIN)
    assert any("Retraining due" in w.value for w in app.warning)  # the administrator is
    table = next(d.value for d in app.dataframe if "promoted" in d.value.columns)
    assert table.iloc[0]["checkpoint"] == "20261020T093000Z"
    assert table.iloc[0]["promoted"] == "yes"
