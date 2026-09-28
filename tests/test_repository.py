from src.reporting.repository import AuditReportRepository


def test_save_and_retrieve_claim_and_report(tmp_db_path):
    repo = AuditReportRepository(tmp_db_path)
    claim_id = repo.save_claim({
        "provider_id": "P1", "patient_ref": "PT1", "icd11_code": "1A00",
        "claim_amount": 1000.0, "service_date": "2026-01-01",
        "discharge_date": "2026-01-01", "clinical_notes": "Patient had fever.",
        "anomaly_score": 0.9,
    })
    verdict_id = repo.save_verdict(claim_id, "inconsistent", 0.85, "v1")
    report_id = repo.save_report(claim_id, verdict_id, {"claim_amount": 0.4}, "This is inconsistent because...")

    report = repo.get_report(report_id)
    assert report["provider_id"] == "P1"
    assert report["verdict"] == "inconsistent"
    assert report["status"] == "pending"


def test_get_pending_reports_only_returns_pending(tmp_db_path):
    repo = AuditReportRepository(tmp_db_path)
    claim_id = repo.save_claim({
        "provider_id": "P1", "patient_ref": "PT1", "icd11_code": "1A00",
        "claim_amount": 1000.0, "service_date": "2026-01-01",
        "discharge_date": "2026-01-01", "clinical_notes": "notes", "anomaly_score": 0.5,
    })
    report_id = repo.save_report(claim_id, None, {}, None)
    assert len(repo.get_pending_reports()) == 1

    repo.record_decision(report_id, "approved", reviewer_id=1)
    assert len(repo.get_pending_reports()) == 0


def test_record_decision_rejects_invalid_status(tmp_db_path):
    repo = AuditReportRepository(tmp_db_path)
    claim_id = repo.save_claim({
        "provider_id": "P1", "patient_ref": "PT1", "icd11_code": "1A00",
        "claim_amount": 1000.0, "service_date": "2026-01-01",
        "discharge_date": "2026-01-01", "clinical_notes": "notes", "anomaly_score": 0.5,
    })
    report_id = repo.save_report(claim_id, None, {}, None)
    try:
        repo.record_decision(report_id, "not-a-real-decision", reviewer_id=1)
        assert False, "expected ValueError"
    except ValueError:
        pass
