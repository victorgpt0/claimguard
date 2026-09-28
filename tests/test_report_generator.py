from unittest.mock import MagicMock

from src.reporting.report_generator import AuditReportGenerator


def test_assemble_and_save_delegates_to_repository():
    fake_repo = MagicMock()
    fake_repo.save_report.return_value = 42

    generator = AuditReportGenerator(fake_repo)
    report_id = generator.assemble_and_save(
        claim_id=1, verdict_id=2, shap_attribution={"claim_amount": 0.5}, llm_narrative="explanation"
    )

    assert report_id == 42
    fake_repo.save_report.assert_called_once_with(1, 2, {"claim_amount": 0.5}, "explanation")
