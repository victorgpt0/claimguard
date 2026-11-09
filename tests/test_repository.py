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


def test_old_database_gains_new_columns(tmp_path):
    """Databases written before claim_ref and flag_sources (e.g. the TC-028 run) still open."""
    import sqlite3
    path = str(tmp_path / "old.sqlite")
    with sqlite3.connect(path) as conn:
        conn.executescript("""
            CREATE TABLE Claim (id INTEGER PRIMARY KEY AUTOINCREMENT, provider_id TEXT NOT NULL,
                patient_ref TEXT NOT NULL, icd11_code TEXT NOT NULL, claim_amount REAL NOT NULL,
                service_date DATE NOT NULL, discharge_date DATE, clinical_notes TEXT NOT NULL,
                anomaly_score REAL);
            CREATE TABLE AuditReport (id INTEGER PRIMARY KEY AUTOINCREMENT, claim_id INTEGER NOT NULL,
                verdict_id INTEGER, shap_attribution TEXT, llm_narrative TEXT,
                status TEXT NOT NULL DEFAULT 'pending', reviewed_by INTEGER, reviewed_at DATETIME,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP);""")
    repository = AuditReportRepository(path)
    claim_id = repository.save_claim({"claim_ref": "C-17", "provider_id": "P1", "patient_ref": "M1",
                                      "icd11_code": "JB20", "claim_amount": 0.0,
                                      "service_date": "2025-11-01", "clinical_notes": "Delivered."})
    report_id = repository.save_report(claim_id, None, {"a": 1.0}, None, ["isolation_forest"])
    report = repository.get_report(report_id)
    assert report["claim_ref"] == "C-17"
    assert report["flag_sources"] == '["isolation_forest"]'
