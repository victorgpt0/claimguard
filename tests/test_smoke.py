"""Smoke test confirming the package structure imports cleanly. Real tests replace/extend
this as each module in tests/test_plan.md gets implemented."""


def test_package_imports():
    import src.data.ingestion  # noqa: F401
    import src.data.preprocessing  # noqa: F401
    import src.detection.base  # noqa: F401
    import src.classification.classifier  # noqa: F401
    import src.icd11.client  # noqa: F401
    import src.narrative.narrative_generator  # noqa: F401
    import src.reporting.report_generator  # noqa: F401
    import src.feedback.retrainer  # noqa: F401
    import src.auth.user  # noqa: F401
