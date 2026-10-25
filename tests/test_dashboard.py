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
    # Streamlit reads .streamlit/secrets.toml from the working directory: run away from the
    # developer's own (Google credentials, #61) so each test sets exactly the secrets it needs.
    monkeypatch.chdir(tmp_path)
    st.cache_resource.clear()  # get_config is cached across AppTest runs in one process
    return config


AUDITOR = User(id=1, username="auditor", role="Auditor")
ADMIN = User(id=2, username="admin", role="Administrator")
AUDITOR_TABS = ["Overview", "Review queue", "Account"]
ADMIN_TABS = ["Overview", "Model performance", "Feedback loop", "Users", "Activity log", "Account"]


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
    return next(b for b in app.button if b.label == "Sign in").click().run()


def _report_picker(app):
    return next(s for s in app.selectbox if s.label.startswith("Report"))


def test_nothing_but_the_sign_in_form_without_a_session(config):
    app = _app(user=None)
    assert not app.exception
    assert not app.tabs and not app.metric
    assert [t.label for t in app.text_input] == ["Username", "Password"]


def test_google_button_only_when_configured(config):
    assert not any(b.label == "Sign in with Google" for b in _app(user=None).button)
    app = AppTest.from_file(APP, default_timeout=30)
    app.secrets["auth"] = {"redirect_uri": "http://localhost:8501/oauth2callback", "cookie_secret": "x",
                           "google": {"client_id": "id", "client_secret": "secret",
                                      "server_metadata_url": "https://accounts.google.com/.well-known/openid-configuration"}}
    app.run()
    assert not app.exception
    assert any(b.label == "Sign in with Google" for b in app.button)
    assert not app.tabs  # offering Google opens nothing by itself


def test_wrong_password_and_unknown_user_get_the_same_message(config):
    for username, password in (("auditor", "wrong-password"), ("nobody", "auditor-pass-1")):
        app = _sign_in(_app(user=None), username, password)
        assert [e.value for e in app.error] == ["Incorrect username or password."]
        assert not app.tabs


def test_signing_in_opens_the_role_tabs_and_signing_out_closes_them(config):
    app = _sign_in(_app(user=None), "auditor", "auditor-pass-1")
    assert not app.exception
    assert _tab_labels(app) == AUDITOR_TABS
    app = next(b for b in app.sidebar.button if b.label == "Sign out").click().run()
    assert not app.tabs
    assert "user" not in app.session_state


def test_each_role_sees_only_its_own_tabs(config):
    assert _tab_labels(_app(AUDITOR)) == AUDITOR_TABS
    admin = _app(ADMIN)
    assert _tab_labels(admin) == ADMIN_TABS
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


# --- Accounts and activity log (#56) ----------------------------------------------------

def _activity(config):
    with sqlite3.connect(config["paths"]["db_path"]) as conn:
        return conn.execute("SELECT user_id, action, detail FROM ActivityLog ORDER BY id").fetchall()


def _form_input(app, label, value):
    next(t for t in app.text_input if t.label.startswith(label)).input(value)


def _submit(app, label):
    return next(b for b in app.button if b.label == label).click().run()


def test_sign_ins_and_decisions_are_logged_without_passwords(config):
    _sign_in(_app(user=None), "auditor", "wrong-password")
    app = _sign_in(_app(user=None), "auditor", "auditor-pass-1")
    app = next(s for s in app.selectbox if s.label == "Lane").select("Text inconsistent").run()
    app = next(b for b in app.button if b.label == "Approved").click().run()
    next(b for b in app.sidebar.button if b.label == "Sign out").click().run()
    assert _activity(config) == [(None, "sign_in_failed", "username: auditor"),
                                 (1, "sign_in", "password"),
                                 (1, "decision", "report #1: approved"),
                                 (1, "sign_out", None)]
    logged = " ".join(str(v) for row in _activity(config) for v in row)
    assert "pass" not in logged.replace("password", "")  # neither password appears
    assert "forceps" not in logged.lower()  # nor note text


def test_auditor_changes_own_password(config):
    app = _app(AUDITOR)
    _form_input(app, "Current password", "wrong-password")
    _form_input(app, "New password", "new-auditor-pass")
    _form_input(app, "Repeat", "new-auditor-pass")
    app = _submit(app, "Change password")
    assert "The current password is incorrect." in [e.value for e in app.error]
    _form_input(app, "Current password", "auditor-pass-1")
    _form_input(app, "New password", "new-auditor-pass")
    _form_input(app, "Repeat", "new-auditor-pass")
    app = _submit(app, "Change password")
    assert "Password changed." in [s.value for s in app.success]
    assert _activity(config)[-1] == (1, "password_changed", None)
    assert not _sign_in(_app(user=None), "auditor", "auditor-pass-1").tabs
    assert _sign_in(_app(user=None), "auditor", "new-auditor-pass").tabs


def test_admin_creates_an_account_that_can_sign_in(config):
    app = _app(ADMIN)
    _form_input(app, "Username", "auditor2")
    _form_input(app, "Initial password", "short")
    app = _submit(app, "Create account")
    assert any("at least 10 characters" in e.value for e in app.error)
    _form_input(app, "Username", "auditor2")
    _form_input(app, "Initial password", "auditor2-pass-1")
    _form_input(app, "Google account email", "Auditor2@Example.com")
    app = _submit(app, "Create account")
    assert not app.exception
    users = next(d.value for d in app.dataframe if "role" in d.value.columns)
    row = users[users["username"] == "auditor2"].iloc[0]
    assert (row["role"], row["email"], row["active"]) == ("Auditor", "auditor2@example.com", "yes")
    assert _activity(config)[-1] == (2, "user_created", "auditor2 (Auditor)")
    assert _tab_labels(_sign_in(_app(user=None), "auditor2", "auditor2-pass-1")) == AUDITOR_TABS


def test_admin_deactivates_an_auditor_who_is_then_signed_out(config):
    app = _app(ADMIN)
    app = next(s for s in app.selectbox if s.label == "Account").select("auditor").run()
    app = _submit(app, "Deactivate")
    assert _activity(config)[-1] == (2, "user_deactivated", "auditor")
    assert not _app(AUDITOR).tabs  # the open session ends on its next run
    assert not _sign_in(_app(user=None), "auditor", "auditor-pass-1").tabs
    app = next(s for s in _app(ADMIN).selectbox if s.label == "Account").select("admin").run()
    app = _submit(app, "Deactivate")
    assert "You cannot deactivate your own account." in [e.value for e in app.error]


def test_activity_log_tab_filters_by_user(config):
    _sign_in(_app(user=None), "auditor", "auditor-pass-1")
    _sign_in(_app(user=None), "admin", "admin-pass-12")
    app = _app(ADMIN)
    app = next(s for s in app.selectbox if s.label == "User").select("auditor").run()
    log = next(d.value for d in app.dataframe if "action" in d.value.columns)
    assert list(log["username"]) == ["auditor"] and list(log["action"]) == ["sign_in"]
