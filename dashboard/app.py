"""Streamlit dashboard entrypoint (IR-03). Role-based views:
- Fraud Auditor: pending-report queue + review (FR-08, FR-09) on a single view showing
  SHAP attribution, classifier verdict, and LLM narrative together (NFR-07).
- Administrator: detection performance metrics + classifier retraining history only
  (FR-11) — threshold configuration is a file (config/config.yaml, FR-10), not a
  dashboard control; user management is out of scope (FR-11 scope note).

Not yet implemented — this is the target shape for Sprint 8 (dashboard implementation,
Section 3.4.1 Product Backlog), after the detection/classification/explanation modules
exist to call into.
"""
import streamlit as st


def main():
    st.set_page_config(page_title="ClaimGuard — SHA Fraud Audit", layout="wide")
    st.title("ClaimGuard")
    st.caption(
        "LLM-Assisted Anomaly Detection for Post-Payment Fraud Audit — "
        "Kenya's Social Health Authority"
    )
    st.info("Dashboard not yet implemented. See implementation_roadmap.md for setup, "
            "and the Product Backlog (Section 3.4.1) for build order.")


if __name__ == "__main__":
    main()
