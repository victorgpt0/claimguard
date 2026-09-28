"""Streamlit dashboard entrypoint (IR-03). Role-based views:
- Fraud Auditor: pending-report queue + review (FR-08, FR-09) on a single view showing
  SHAP attribution, classifier verdict, and LLM narrative together (NFR-07).
- Administrator: detection performance metrics + classifier retraining history only
  (FR-11) — threshold configuration is a file (config/config.yaml, FR-10), not a
  dashboard control; user management is out of scope (FR-11 scope note).
"""
from __future__ import annotations

import json

import streamlit as st

from src.auth.user import authenticate
from src.classification.classifier import SemanticConsistencyClassifier
from src.config import load_config
from src.feedback.feedback_repository import FeedbackRepository
from src.feedback.retrainer import ClassifierRetrainer
from src.reporting.repository import AuditReportRepository


@st.cache_resource
def get_config():
    return load_config()


@st.cache_resource
def get_repository(db_path: str) -> AuditReportRepository:
    return AuditReportRepository(db_path)


def login_form():
    st.title("ClaimGuard — Sign in")
    with st.form("login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")
    if submitted:
        config = get_config()
        user = authenticate(config["paths"]["db_path"], username, password)
        if user is None:
            st.error("Invalid username or password.")
        else:
            st.session_state["user"] = user
            st.rerun()


def auditor_view(repository: AuditReportRepository, config: dict):
    st.header("Pending Audit Reports")
    pending = repository.get_pending_reports()
    if not pending:
        st.info("No pending reports.")
        return

    report_options = {f"#{r['id']} — {r['provider_id']} — {r['icd11_code']}": r["id"] for r in pending}
    selected_label = st.selectbox("Select a report to review", list(report_options.keys()))
    report = repository.get_report(report_options[selected_label])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Claim")
        st.write(f"**Provider:** {report['provider_id']}")
        st.write(f"**Billed ICD-11 code:** {report['icd11_code']}")
        st.write(f"**Claim amount:** {report['claim_amount']}")
        st.write("**Clinical notes:**")
        st.write(report["clinical_notes"])
    with col2:
        st.subheader("Classifier verdict")
        st.write(f"**Verdict:** {report.get('verdict', 'n/a')} "
                 f"(confidence: {report.get('confidence', 0):.2f})")
        st.subheader("SHAP attribution")
        st.json(json.loads(report["shap_attribution"] or "{}"))
        if report.get("llm_narrative"):
            st.subheader("Audit narrative")
            st.write(report["llm_narrative"])

    st.subheader("Decision")
    decision_cols = st.columns(3)
    for col, decision in zip(decision_cols, ["approved", "rejected", "escalated"]):
        if col.button(decision.capitalize(), key=f"decide_{decision}"):
            _record_decision_and_maybe_feedback(repository, report, decision, config)
            st.success(f"Recorded: {decision}")
            st.rerun()


def _record_decision_and_maybe_feedback(repository, report, decision, config):
    reviewer_id = st.session_state["user"].id
    repository.record_decision(report["id"], decision, reviewer_id)

    if decision in ("approved", "rejected"):
        feedback_repo = FeedbackRepository(config["paths"]["artifacts_dir"] + "/feedback_dataset/examples.jsonl")
        feedback_repo.materialize_example(
            claim={"id": report["claim_id"], "clinical_notes": report["clinical_notes"],
                   "icd11_code": report["icd11_code"]},
            verdict={"verdict": report.get("verdict")},
            auditor_decision=decision,
        )
        retrainer = ClassifierRetrainer(
            feedback_repo,
            active_model_path=config["paths"]["model_dir"] + "/classifier_lora/active",
            candidate_dir=config["paths"]["model_dir"] + "/classifier_lora/candidates",
        )
        if retrainer.maybe_trigger_retraining(config["thresholds"]["retrain_trigger_count"]):
            st.info("Retraining trigger reached — run src/classification/train.py on "
                    "Colab against the current Feedback Dataset (see implementation_roadmap.md).")


def admin_view(config: dict):
    st.header("Detection Performance Metrics")
    active_model_path = config["paths"]["model_dir"] + "/classifier_lora/active"
    metrics = SemanticConsistencyClassifier.load_metrics(active_model_path)
    if metrics is None:
        st.warning("No trained classifier checkpoint found yet — nothing to report.")
    else:
        st.json(metrics)


def main():
    st.set_page_config(page_title="ClaimGuard — SHA Fraud Audit", layout="wide")

    try:
        config = get_config()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    if "user" not in st.session_state:
        login_form()
        return

    user = st.session_state["user"]
    st.sidebar.write(f"Signed in as **{user.username}** ({user.role})")
    if st.sidebar.button("Sign out"):
        del st.session_state["user"]
        st.rerun()

    repository = get_repository(config["paths"]["db_path"])
    if user.role == "Auditor":
        auditor_view(repository, config)
    else:
        admin_view(config)


if __name__ == "__main__":
    main()
