"""Streamlit dashboard entrypoint (IR-03): sign-in first (NFR-03), then the tabs of the
signed-in user's role. The Fraud Auditor sees Overview, Review queue and Account; the
Administrator sees Overview, Model performance, Feedback loop, Users, Activity log and
Account. Sign-ins, decisions and account changes are written to the activity log (#56).

- Overview: statistics and charts over every audit report in the database, plus the
  latest pipeline run's funnel when its log is available.
- Review queue: pending reports with SHAP attribution, classifier verdict and LLM
  narrative together (FR-08, NFR-07), and the approve/reject/escalate decision (FR-09).
- Model performance: the active classifier's validation metrics and stratified breakdown.
- Feedback loop: the Feedback Dataset and the retraining trigger (FR-12, FR-13).

Thresholds come from config/config.yaml (FR-10), not a dashboard control. The dashboard
never loads torch or transformers: it reads what the pipeline and training runs wrote.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.auth import activity_log as log
from src.auth.activity_log import ActivityLog
from src.auth.user import MIN_PASSWORD_LENGTH, ROLES, User, authenticate, user_for_google_identity
from src.auth.user_repository import UserRepository
from src.classification.export import reference_text
from src.config import load_config
from src.feedback.feedback_repository import FeedbackRepository
from src.feedback.retrainer import ClassifierRetrainer
from src.feedback.run_retraining import history_path, read_history
from src.icd11.cache import ICD11Cache
from src.reporting.repository import AuditReportRepository

DECISIONS = ["approved", "rejected", "escalated"]
TYPOLOGY_NAMES = {
    "none": "Clean", "upcoding": "Upcoding", "ghost_billing": "Ghost billing",
    "record_fabrication": "Record fabrication", "outpatient_to_inpatient": "Outpatient conversion",
    "anomalous_dependents": "Anomalous dependents", "inflated_caesarean": "Inflated caesarean",
}


@st.cache_resource
def get_config():
    return load_config()


@st.cache_resource
def get_repository(db_path: str) -> AuditReportRepository:
    return AuditReportRepository(db_path)


@st.cache_resource
def get_users(db_path: str) -> UserRepository:
    return UserRepository(db_path)


@st.cache_resource
def get_activity_log(db_path: str) -> ActivityLog:
    return ActivityLog(db_path)


def record_activity(config: dict, action: str, user: User | None = None, detail: str = ""):
    get_activity_log(config["paths"]["db_path"]).record(action, user.id if user else None, detail)


@st.cache_resource
def get_icd11_cache(artifacts_dir: str) -> ICD11Cache:
    return ICD11Cache(artifacts_dir + "/icd11_cache.json")


def code_reference(config: dict, code: str) -> str:
    """The same reference text the classifier judged the claim against (export.py)."""
    return reference_text(get_icd11_cache(config["paths"]["artifacts_dir"]).get(code))


def feedback_repository(config: dict) -> FeedbackRepository:
    return FeedbackRepository(config["paths"]["artifacts_dir"] + "/feedback_dataset/examples.jsonl")


def active_model_path(config: dict) -> Path:
    return Path(config["paths"]["model_dir"]) / "classifier_lora" / "active"


def read_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def latest_pipeline_run(config: dict) -> dict | None:
    """The newest log written by src.pipeline, if one has been copied from Drive."""
    experiments = config["paths"].get("experiments_dir")
    if not experiments:
        return None
    runs = sorted((Path(experiments) / "pipeline_runs").glob("*.json"))
    return read_json(runs[-1]) if runs else None


def kes(amount: float) -> str:
    return f"KES {amount:,.0f}"


# --- Overview ---------------------------------------------------------------------------

def overview_tab(reports: pd.DataFrame, config: dict):
    if reports.empty:
        st.info("The database has no audit reports yet. Run the pipeline (notebook 03) first.")
        return

    reviewed = reports[reports["status"] != "pending"]
    inconsistent = reports[reports["verdict"] == "inconsistent"]
    cols = st.columns(5)
    cols[0].metric("Flagged claims", len(reports))
    cols[1].metric("Pending review", int((reports["status"] == "pending").sum()))
    cols[2].metric("Reviewed", len(reviewed),
                   f"{len(reviewed) / len(reports):.0%} of reports", delta_color="off")
    cols[3].metric("Inconsistent verdicts", len(inconsistent),
                   f"{int(reports['has_narrative'].sum())} with narrative", delta_color="off")
    cols[4].metric("Amount under audit", kes(reports["claim_amount"].sum()))

    run = latest_pipeline_run(config)
    if run:
        funnel_chart(run)

    left, right = st.columns(2)
    with left:
        st.subheader("Review status")
        status = reports["status"].value_counts().reindex(["pending", *DECISIONS], fill_value=0)
        st.altair_chart(donut(status.rename_axis("status").reset_index(name="reports"), "status"),
                        width="stretch")
    with right:
        st.subheader("Classifier verdicts")
        verdicts = reports["verdict"].fillna("none").value_counts()
        st.altair_chart(donut(verdicts.rename_axis("verdict").reset_index(name="reports"), "verdict"),
                        width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader("Anomaly scores of flagged claims")
        st.caption("Isolation Forest score: higher is more anomalous.")
        st.altair_chart(
            alt.Chart(reports.dropna(subset=["anomaly_score"])).mark_bar().encode(
                alt.X("anomaly_score:Q", bin=alt.Bin(maxbins=20), title="Anomaly score"),
                alt.Y("count():Q", title="Claims"),
                alt.Color("verdict:N", title="Verdict"),
                tooltip=["count():Q"]),
            width="stretch")
    with right:
        st.subheader("Why claims were flagged")
        st.caption("How often each feature is among a claim's top 3 SHAP attributions.")
        shap = shap_frequency(reports["shap_attribution"])
        st.altair_chart(
            alt.Chart(shap).mark_bar().encode(
                alt.X("claims:Q", title="Claims"),
                alt.Y("feature:N", sort="-x", title=None),
                tooltip=["feature", "claims", alt.Tooltip("mean_abs_shap:Q", format=".3f")]),
            width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader("Providers with the most flagged claims")
        providers = (reports.groupby("provider_id")
                     .agg(claims=("id", "size"), amount=("claim_amount", "sum"))
                     .sort_values("claims", ascending=False).head(10).reset_index())
        st.altair_chart(
            alt.Chart(providers).mark_bar().encode(
                alt.X("claims:Q", title="Flagged claims"),
                alt.Y("provider_id:N", sort="-x", title=None),
                tooltip=["provider_id", "claims", alt.Tooltip("amount:Q", format=",.0f", title="KES")]),
            width="stretch")
    with right:
        st.subheader("Flagged amount by ICD-11 code")
        codes = (reports.groupby("icd11_code")
                 .agg(claims=("id", "size"), amount=("claim_amount", "sum"))
                 .sort_values("amount", ascending=False).head(10).reset_index())
        st.altair_chart(
            alt.Chart(codes).mark_bar().encode(
                alt.X("amount:Q", title="KES"),
                alt.Y("icd11_code:N", sort="-x", title=None),
                tooltip=["icd11_code", "claims", alt.Tooltip("amount:Q", format=",.0f", title="KES")]),
            width="stretch")


def donut(frame: pd.DataFrame, category: str) -> alt.Chart:
    return alt.Chart(frame).mark_arc(innerRadius=50).encode(
        alt.Theta("reports:Q"), alt.Color(f"{category}:N", title=None),
        tooltip=[category, "reports"]).properties(height=220)


def shap_frequency(attributions: pd.Series) -> pd.DataFrame:
    counts, magnitude = Counter(), Counter()
    for raw in attributions.dropna():
        for feature, value in json.loads(raw).items():
            counts[feature] += 1
            magnitude[feature] += abs(value)
    return pd.DataFrame([{"feature": f, "claims": n, "mean_abs_shap": magnitude[f] / n}
                         for f, n in counts.items()])


def funnel_chart(run: dict):
    """The latest run's funnel: screened -> flagged -> inconsistent -> narrative."""
    summary, evaluation = run["summary"], run.get("evaluation", {})
    st.subheader("Latest pipeline run")
    st.caption(f"Started {run['started_at'][:16].replace('T', ' ')} UTC, "
               f"batch of {summary['claims']} claims.")
    left, right = st.columns(2)
    with left:
        stages = pd.DataFrame({
            "stage": ["Screened", "Flagged (stage 1)", "Inconsistent (classifier)", "Narrative (LLM)"],
            "claims": [summary["claims"], summary["flagged_stage1"], summary["inconsistent"],
                       summary["narratives"]]})
        st.altair_chart(
            alt.Chart(stages).mark_bar().encode(
                alt.X("claims:Q", title="Claims", scale=alt.Scale(type="symlog")),
                alt.Y("stage:N", sort=None, title=None),
                tooltip=["stage", "claims"]) +
            alt.Chart(stages).mark_text(align="left", dx=4).encode(
                alt.X("claims:Q", scale=alt.Scale(type="symlog")), alt.Y("stage:N", sort=None),
                text="claims:Q"),
            width="stretch")
        st.caption("Log scale.")
    with right:
        by_typology = evaluation.get("flagged_by_typology")
        reported = summary.get("reported")
        if reported:
            st.caption(f"Queued for review: {reported['text_inconsistent']} text inconsistent, "
                       f"{reported['structural_anomaly']} structural anomaly; "
                       f"{summary.get('screen_false_positives', 0)} text-screen flags the "
                       "classifier cleared were logged only.")
        if by_typology:
            frame = pd.DataFrame([
                {"typology": TYPOLOGY_NAMES.get(t, t), "share flagged": v["sum"] / v["size"],
                 "flagged": v["sum"], "claims": v["size"]}
                for t, v in by_typology.items()])
            st.altair_chart(
                alt.Chart(frame).mark_bar().encode(
                    alt.X("share flagged:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1]),
                          title="Share flagged at stage 1 (ground truth)"),
                    alt.Y("typology:N", sort="-x", title=None),
                    tooltip=["typology", "flagged", "claims"]),
                width="stretch")
            st.caption(f"Stage-1 recall {evaluation['stage1_recall']:.0%}; "
                       f"{evaluation['inconsistent_caught']} of {evaluation['inconsistent_total']} "
                       "text-inconsistent claims reached a narrative. Ground truth exists only "
                       "because the data is synthetic.")


# --- Review queue -----------------------------------------------------------------------

# Review lanes (#46, src/pipeline.py): the lane follows from the classifier's verdict.
LANES = {"Text inconsistent": "inconsistent", "Structural anomaly": "consistent"}
LANE_OF_VERDICT = {verdict: lane for lane, verdict in LANES.items()}
SOURCE_NAMES = {"isolation_forest": "Isolation Forest", "autoencoder": "autoencoder",
                "text_screen": "text-consistency screen"}

def review_tab(repository: AuditReportRepository, config: dict):
    pending = pd.DataFrame(repository.get_all_reports())
    pending = pending[pending["status"] == "pending"] if not pending.empty else pending
    if pending.empty:
        st.info("No pending reports.")
        return

    filters = st.columns([2, 2, 3])
    lane = filters[0].selectbox("Lane", ["All", *LANES],
                                help="Text inconsistent: the notes do not support the billed code. "
                                     "Structural anomaly: the notes are consistent, but the billing "
                                     "fields look anomalous.")
    order = filters[1].selectbox("Order", ["Highest anomaly score", "Oldest first"])
    if lane != "All":
        pending = pending[pending["verdict"] == LANES[lane]]
    if order == "Highest anomaly score":
        pending = pending.sort_values("anomaly_score", ascending=False)
    if pending.empty:
        st.info("No pending reports match.")
        return

    labels = {f"#{r.id} · {LANE_OF_VERDICT.get(r.verdict, 'no verdict')} · {r.provider_id} · {r.icd11_code}": r.id
              for r in pending.itertuples()}
    selected = filters[2].selectbox(f"Report ({len(labels)} pending)", list(labels))
    report = repository.get_report(labels[selected])

    claim_panel(report)
    left, right = st.columns(2)
    with left:
        notes_panel(report)
    with right:
        code_panel(config, report["icd11_code"])
    left, right = st.columns(2)
    with left:
        shap_panel(report)
    with right:
        verdict_panel(report)
    narrative_panel(report)

    st.subheader("Decision")
    for col, decision in zip(st.columns(3), DECISIONS):
        if col.button(decision.capitalize(), key=f"decide_{decision}", width="stretch"):
            record_decision(repository, report, decision, config, st.session_state["user"])
            record_activity(config, log.DECISION, st.session_state["user"], f"report #{report['id']}: {decision}")
            st.toast(f"Report #{report['id']}: {decision}")
            st.rerun()


# The review screen shows a report as five numbered panels in pipeline order: the claim's
# inputs (1-3), then what each stage concluded (4-6). Inputs and outputs are labelled as
# such so a reviewer can tell submitted data from generated text.

def claim_panel(report: dict):
    with st.container(border=True):
        st.markdown("#### 1 · Structured claim fields")
        st.caption("Input: the billing record as submitted.")
        stay = "—"
        if report.get("discharge_date"):
            days = (pd.Timestamp(report["discharge_date"]) - pd.Timestamp(report["service_date"])).days
            stay = f"{days} day{'s' if days != 1 else ''}"
        fields = [("Provider", report["provider_id"]), ("Member", report.get("patient_ref") or "—"),
                  ("Billed ICD-11 code", report["icd11_code"]), ("Amount", kes(report["claim_amount"])),
                  ("Service date", report.get("service_date") or "—"),
                  ("Discharge date", report.get("discharge_date") or "— (outpatient)"),
                  ("Length of stay", stay)]
        for row in (fields[:4], fields[4:]):
            for col, (label, value) in zip(st.columns(4), row):
                col.metric(label, value)


def notes_panel(report: dict):
    with st.container(border=True, height="stretch"):
        st.markdown("#### 2 · Clinical notes")
        st.caption("Input: the provider's free-text justification, exactly as submitted.")
        st.text(report["clinical_notes"])


def code_panel(config: dict, code: str):
    definition = get_icd11_cache(config["paths"]["artifacts_dir"]).get(code)
    with st.container(border=True, height="stretch"):
        st.markdown(f"#### 3 · ICD-11 reference for {code}")
        st.caption("Reference: WHO ICD-11 title and definition the notes are checked against.")
        st.markdown(f"**{definition.title}**")
        if definition.definition:
            st.markdown(definition.definition)
            if definition.definition_source and definition.definition_source != code:
                st.caption(f"Definition inherited from parent category {definition.definition_source}.")
        else:
            st.caption("ICD-11 gives no definition for this code; the title alone is used.")


def shap_panel(report: dict):
    with st.container(border=True, height="stretch"):
        st.markdown("#### 4 · Stage 1: why the claim was flagged")
        st.caption(f"Output: anomaly score {report.get('anomaly_score') or 0:.3f} from the anomaly "
                   "detectors, with SHAP's top 3 contributing fields. Red pushes the claim towards "
                   "anomalous, blue away.")
        sources = json.loads(report.get("flag_sources") or "null")
        if sources:
            st.markdown("Flagged by: " + ", ".join(SOURCE_NAMES.get(s, s) for s in sources))
        attribution = json.loads(report["shap_attribution"] or "{}")
        if attribution:
            frame = pd.DataFrame({"feature": list(attribution), "shap": list(attribution.values())})
            st.altair_chart(
                alt.Chart(frame).mark_bar().encode(
                    alt.X("shap:Q", title="Contribution to the anomaly score"),
                    alt.Y("feature:N", sort="-x", title=None),
                    color=alt.condition("datum.shap > 0", alt.value("#d62728"), alt.value("#1f77b4")),
                    tooltip=["feature", alt.Tooltip("shap:Q", format=".3f")]).properties(height=140),
                width="stretch")


def verdict_panel(report: dict):
    verdict = report.get("verdict")
    with st.container(border=True, height="stretch"):
        st.markdown("#### 5 · Stage 2: classifier verdict")
        st.caption("Output: do the notes (2) support the billed code (3)?")
        message = f"**{verdict or 'no verdict'}** · {report.get('confidence') or 0:.0%} confidence"
        if verdict == "inconsistent":
            st.error(message)
        elif verdict == "consistent":
            st.success(message)
        else:
            st.warning(message)
        st.caption(f"Model: {report.get('model_version') or 'unknown'}")


def narrative_panel(report: dict):
    with st.container(border=True):
        st.markdown("#### 6 · Stage 3: audit narrative")
        if report.get("llm_narrative"):
            st.caption("Output: generated by the LLM explanation layer from (2), (3) and the "
                       "verdict. It explains the verdict; it does not decide it.")
            st.markdown(f"> {report['llm_narrative']}")
        elif report.get("verdict") == "consistent":
            st.info("No narrative: the classifier found the notes consistent with the billed "
                    "code, and narratives are written only for inconsistent verdicts. The claim "
                    "is in the structural-anomaly lane because its billing fields looked "
                    "anomalous (panel 4).")
        else:
            st.warning("No narrative: generation failed or no LLM was available for this run.")


def record_decision(repository, report, decision, config, user: User):
    repository.record_decision(report["id"], decision, reviewer_id=user.id)
    if decision not in ("approved", "rejected"):
        return
    feedback = feedback_repository(config)
    feedback.materialize_example(
        claim={"id": report["claim_id"], "clinical_notes": report["clinical_notes"],
               "icd11_code": report["icd11_code"]},
        verdict={"verdict": report.get("verdict")},
        auditor_decision=decision,
        icd11_definition=code_reference(config, report["icd11_code"]),
    )
    # The retraining-due notice is the administrator's, shown on the feedback tab (#49).


# --- Model performance ------------------------------------------------------------------

def model_tab(config: dict):
    path = active_model_path(config)
    metrics = read_json(path / "metrics.json")
    if metrics is None:
        st.warning(f"No classifier checkpoint at {path}. Copy the active run folder there.")
        return
    record = read_json(path / "config.json") or {}
    st.caption(f"Active checkpoint: {record.get('run_id', path.name)} · "
               f"validation split · epoch {metrics.get('epoch', '?')}")
    cols = st.columns(4)
    cols[0].metric("Precision", f"{metrics['precision']:.3f}")
    cols[1].metric("Recall", f"{metrics['recall']:.3f}", "NFR-05 floor 0.80", delta_color="off")
    cols[2].metric("F1", f"{metrics['f1']:.3f}")
    cols[3].metric("Caught", f"{round(metrics['recall'] * metrics['positives'])} / {metrics['positives']}")

    evaluation = read_json(path / "evaluation_validation.json")
    if evaluation is None:
        st.info("Run src.classification.evaluate for the stratified breakdown.")
        return
    st.subheader("Flagged by fraud typology")
    st.caption("For upcoding, ghost billing and record fabrication this is recall; for the "
               "others the notes match the code, so any flag is a false positive.")
    inconsistent_types = {"upcoding", "ghost_billing", "record_fabrication"}
    frame = pd.DataFrame([
        {"typology": TYPOLOGY_NAMES.get(t, t), "rate": v["rate"], "flagged": v["flagged"],
         "claims": v["claims"],
         "should be flagged": "yes" if t in inconsistent_types else "no"}
        for t, v in evaluation["flagged_by_typology"].items()])
    st.altair_chart(
        alt.Chart(frame).mark_bar().encode(
            alt.X("rate:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1]), title="Share flagged"),
            alt.Y("typology:N", sort="-x", title=None),
            alt.Color("should be flagged:N", scale=alt.Scale(domain=["yes", "no"], range=["#2ca02c", "#d62728"])),
            tooltip=["typology", "flagged", "claims"]),
        width="stretch")

    left, right = st.columns(2)
    left.subheader("Clean vs noisy text")
    left.dataframe(strata_table(evaluation["text_noise"], {"False": "Clean", "True": "Noisy"}))
    right.subheader("Common vs rare ICD-11 codes")
    right.dataframe(strata_table(evaluation["code_frequency"], {}))
    right.caption(f"Rare: {evaluation['rare_code_threshold']}.")


def strata_table(strata: dict, names: dict) -> pd.DataFrame:
    return pd.DataFrame({names.get(k, k): v for k, v in strata.items()}).T[
        ["claims", "inconsistent", "precision", "recall", "f1"]]


# --- Feedback loop ----------------------------------------------------------------------

def feedback_tab(config: dict):
    feedback = feedback_repository(config)
    trigger = config["thresholds"]["retrain_trigger_count"]
    unconsumed = feedback.count_unconsumed_examples()
    examples = pd.DataFrame(feedback.read_examples(since_consumed_marker=False))
    cols = st.columns(3)
    cols[0].metric("Feedback examples", len(examples))
    cols[1].metric("Since last retraining", unconsumed)
    cols[2].metric("Retraining trigger", trigger)
    st.progress(min(unconsumed / trigger, 1.0), f"{unconsumed} of {trigger} examples toward the next retraining")
    retrainer = ClassifierRetrainer(feedback, active_model_path=str(active_model_path(config)),
                                    candidate_dir=str(active_model_path(config).parent / "candidates"))
    if retrainer.maybe_trigger_retraining(trigger):  # IR-06
        st.warning("Retraining due: run `python -m src.feedback.run_retraining` on Colab. It "
                   "trains a candidate on the training split plus this feedback and promotes it "
                   "only if it scores at least as well on the held-out test set (FR-14).")
    st.caption("Approve and reject decisions become labelled examples for the classifier: a "
               "rejection is labelled consistent, an approval of an inconsistent verdict "
               "inconsistent. Approving a consistent verdict and escalating add nothing.")
    if not examples.empty:
        st.dataframe(examples[["claim_id", "icd11_code", "classifier_verdict", "auditor_decision", "label"]],
                     hide_index=True)
    retraining_history(config)


def retraining_history(config: dict):
    """FR-11 (#47): every retraining run, from experiments/retraining_history.jsonl."""
    st.subheader("Retraining history")
    history = read_history(history_path(config))
    if not history:
        st.caption("No retraining run yet.")
        return
    rows = []
    for run in reversed(history):
        candidate, active = run.get("candidate_metrics") or {}, run.get("active_metrics") or {}
        rows.append({"run at (UTC)": run["run_at"][:16].replace("T", " "),
                     "checkpoint": Path(run["candidate_path"]).name,
                     "feedback examples": run["feedback_examples"],
                     "simulated": run.get("simulated_examples", 0),
                     "candidate F1": candidate.get("f1"), "candidate recall": candidate.get("recall"),
                     "active F1": active.get("f1"), "active recall": active.get("recall"),
                     "promoted": "yes" if run["promoted"] else "no"})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption("F1 and recall on the held-out test set, the promotion gate (FR-14). Simulated "
               "examples come from ground-truth decisions, not an auditor.")


# --- Accounts and activity log (#56) ----------------------------------------------------------

def account_tab(config: dict, user: User):
    users = get_users(config["paths"]["db_path"])
    me = users.get(user.id)
    st.markdown(f"**{me['username']}** · {me['role']} · Google account: {me['email'] or 'not linked'}")
    st.subheader("Change password")
    with st.form("change_password", clear_on_submit=True):
        current = st.text_input("Current password", type="password")
        new = st.text_input(f"New password (at least {MIN_PASSWORD_LENGTH} characters)", type="password")
        repeat = st.text_input("Repeat the new password", type="password")
        submitted = st.form_submit_button("Change password")
    if submitted:
        try:
            if new != repeat:
                raise ValueError("The two new passwords differ.")
            users.change_password(user.id, current, new)
        except ValueError as exc:
            st.error(str(exc))
        else:
            record_activity(config, log.PASSWORD_CHANGED, user)
            st.success("Password changed.")


def users_tab(config: dict, admin: User):
    users = get_users(config["paths"]["db_path"])
    accounts = pd.DataFrame(users.list_users())
    accounts["active"] = accounts["active"].map({1: "yes", 0: "no"})
    st.dataframe(accounts[["username", "role", "email", "active", "created_at"]], hide_index=True)

    st.subheader("Create an account")
    with st.form("create_user", clear_on_submit=True):
        cols = st.columns(2)
        username = cols[0].text_input("Username")
        role = cols[1].selectbox("Role", ROLES)
        password = cols[0].text_input(f"Initial password (at least {MIN_PASSWORD_LENGTH} characters)",
                                      type="password")
        email = cols[1].text_input("Google account email (optional)")
        submitted = st.form_submit_button("Create account")
    if submitted:
        try:
            users.create_user(username, password, role, email or None)
        except ValueError as exc:
            st.error(str(exc))
        else:
            record_activity(config, log.USER_CREATED, admin, f"{username.strip()} ({role})")
            st.success(f"Created {role} '{username.strip()}'.")
            st.rerun()

    st.subheader("Manage an account")
    by_name = {row["username"]: row for row in users.list_users()}
    target = by_name[st.selectbox("Account", list(by_name))]
    cols = st.columns(3)
    with cols[0]:
        label = "Deactivate" if target["active"] else "Reactivate"
        if st.button(label, key="toggle_active"):
            try:
                users.set_active(target["id"], not target["active"], acting_user_id=admin.id)
            except ValueError as exc:
                st.error(str(exc))
            else:
                action = log.USER_DEACTIVATED if target["active"] else log.USER_REACTIVATED
                record_activity(config, action, admin, target["username"])
                st.rerun()
    with cols[1], st.form("reset_password", clear_on_submit=True):
        new = st.text_input("New password", type="password")
        if st.form_submit_button("Reset password"):
            try:
                users.reset_password(target["id"], new)
            except ValueError as exc:
                st.error(str(exc))
            else:
                record_activity(config, log.PASSWORD_RESET, admin, target["username"])
                st.success(f"Password reset for '{target['username']}'.")
    with cols[2], st.form("link_email", clear_on_submit=True):
        email = st.text_input("Google account email", placeholder=target["email"] or "blank unlinks")
        if st.form_submit_button("Link Google account"):
            try:
                users.link_email(target["id"], email)
            except ValueError as exc:
                st.error(str(exc))
            else:
                record_activity(config, log.EMAIL_LINKED, admin, f"{target['username']}: {email.strip() or 'unlinked'}")
                st.rerun()


def activity_tab(config: dict, admin: User):
    activity = get_activity_log(config["paths"]["db_path"])
    usernames = sorted(row["username"] for row in get_users(config["paths"]["db_path"]).list_users())
    cols = st.columns(2)
    username = cols[0].selectbox("User", ["All", *usernames])
    action = cols[1].selectbox("Action", ["All", *sorted(log.ACTIONS)])
    entries = pd.DataFrame(activity.entries(None if username == "All" else username,
                                            None if action == "All" else action),
                           columns=["id", "at", "username", "action", "detail"])
    st.caption(f"{len(entries)} entries, newest first. Times are UTC.")
    st.dataframe(entries.drop(columns="id"), hide_index=True)
    st.download_button("Download as CSV", entries.to_csv(index=False), "activity_log.csv", "text/csv",
                       on_click=record_activity, args=(config, log.LOG_DOWNLOADED, admin))


# --- Sign-in (NFR-03) ---------------------------------------------------------------------

AUDITOR, ADMINISTRATOR = "Auditor", "Administrator"
DEFAULT_SESSION_TIMEOUT_MINUTES = 30


def session_timeout_seconds(config: dict) -> float:
    minutes = (config.get("auth") or {}).get("session_timeout_minutes", DEFAULT_SESSION_TIMEOUT_MINUTES)
    return float(minutes) * 60


def google_configured() -> bool:
    """#61: Google sign-in is offered only when .streamlit/secrets.toml has [auth.google]."""
    try:
        return "google" in (st.secrets.get("auth") or {})
    except (FileNotFoundError, KeyError, AttributeError):  # no secrets file at all
        return False


def google_signed_in() -> bool:
    return google_configured() and bool(st.user.get("is_logged_in", False))


def sign_out(config: dict, action: str = log.SIGN_OUT):
    if st.session_state.get("user") is not None:
        record_activity(config, action, st.session_state["user"])
    via_google = st.session_state.get("via") == "google"
    for key in ("user", "last_active", "via"):
        st.session_state.pop(key, None)
    if via_google and google_signed_in():
        st.logout()  # also clears Streamlit's identity cookie, or the next run signs back in


def start_session(user: User, via: str):
    st.session_state.update(user=user, last_active=time.time(), via=via)


def current_user(config: dict) -> User | None:
    """The signed-in user, or None. A session idle for longer than the timeout is signed out.
    A Google identity becomes a session only for a linked account (user_for_google_identity)."""
    user = st.session_state.get("user")
    if user is None and google_signed_in() and not st.session_state.get("google_refused"):
        user = user_for_google_identity(config["paths"]["db_path"], dict(st.user))
        if user is None:
            st.session_state["google_refused"] = st.user.get("email") or "this Google account"
            record_activity(config, log.SIGN_IN_REFUSED, detail=st.session_state["google_refused"])
            return None
        start_session(user, "google")
        record_activity(config, log.SIGN_IN, user, "google")
    if user is None:
        return None
    if time.time() - st.session_state.get("last_active", 0) > session_timeout_seconds(config):
        st.session_state["expired"] = True
        sign_out(config, log.SESSION_EXPIRED)
        return None
    if not get_users(config["paths"]["db_path"]).is_active(user.id):
        sign_out(config)  # deactivated by the Administrator while signed in
        return None
    st.session_state["last_active"] = time.time()
    return user


def sign_in_form(config: dict):
    st.title("ClaimGuard")
    st.caption("SHA post-payment fraud audit · sign in to continue")
    if st.session_state.pop("expired", False):
        st.info("Your session expired. Sign in again.")
    refused = st.session_state.get("google_refused")
    if refused:
        st.error(f"{refused} is not linked to a ClaimGuard account. Ask the Administrator to "
                 "link it, or sign in with a username and password.")
        if st.button("Use another Google account"):
            st.session_state.pop("google_refused")
            st.logout()
    elif google_configured():
        st.button("Sign in with Google", on_click=st.login, args=("google",))
        st.caption("or with a username and password:")
    with st.form("sign_in"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")
    if submitted:
        # One message for an unknown user and a wrong password, so usernames can't be probed.
        user = authenticate(config["paths"]["db_path"], username.strip(), password)
        if user is None:
            record_activity(config, log.SIGN_IN_FAILED, detail=f"username: {username.strip()[:64]}")
            st.error("Incorrect username or password.")
            return
        st.session_state.pop("google_refused", None)
        start_session(user, "password")
        record_activity(config, log.SIGN_IN, user, "password")
        st.rerun()


def main():
    st.set_page_config(page_title="ClaimGuard — SHA Fraud Audit", layout="wide")
    try:
        config = get_config()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    repository = get_repository(config["paths"]["db_path"])  # creates the schema on a new database
    user = current_user(config)
    if user is None:
        sign_in_form(config)
        return

    with st.sidebar:
        st.markdown(f"Signed in as **{user.username}** ({user.role})")
        if st.button("Sign out"):
            sign_out(config)
            st.rerun()

    st.title("ClaimGuard")
    st.caption(f"SHA post-payment fraud audit")
    reports = pd.DataFrame(repository.get_all_reports())
    # The role decides which tabs exist at all, so a tab's data is never read for the
    # wrong role (IR-03, NFR-03).
    tabs = {AUDITOR: ["Overview", "Review queue", "Account"],
            ADMINISTRATOR: ["Overview", "Model performance", "Feedback loop", "Users", "Activity log",
                            "Account"]}[user.role]
    for name, tab in zip(tabs, st.tabs(tabs)):
        with tab:
            if name == "Overview":
                overview_tab(reports, config)
            elif name == "Review queue":
                review_tab(repository, config)
            elif name == "Model performance":
                model_tab(config)
            elif name == "Feedback loop":
                feedback_tab(config)
            elif name == "Users":
                users_tab(config, user)
            elif name == "Activity log":
                activity_tab(config, user)
            elif name == "Account":
                account_tab(config, user)


if __name__ == "__main__":
    main()
