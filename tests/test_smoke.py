"""Smoke test confirming the package structure imports cleanly. Real tests replace/extend
this as each module in tests/test_plan.md gets implemented."""


def test_package_imports():
    import src.config  # noqa: F401
    import src.data.ingestion  # noqa: F401
    import src.data.noise_injection  # noqa: F401
    import src.data.preprocessing  # noqa: F401
    import src.detection.base  # noqa: F401
    import src.detection.isolation_forest  # noqa: F401
    import src.detection.autoencoder  # noqa: F401
    import src.explainability.shap_explainer  # noqa: F401
    import src.classification.classifier  # noqa: F401
    import src.classification.train  # noqa: F401
    import src.icd11.cache  # noqa: F401
    import src.icd11.client  # noqa: F401
    import src.narrative.narrative_generator  # noqa: F401
    import src.reporting.report_generator  # noqa: F401
    import src.reporting.repository  # noqa: F401
    import src.feedback.feedback_repository  # noqa: F401
    import src.feedback.retrainer  # noqa: F401
