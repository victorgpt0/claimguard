"""import_sqlite: the Streamlit-era database comes across with accounts, passwords, claims,
reports, decisions, reviewers and activity. Claims and reports are written with the pipeline's
own repository; accounts and activity as the removed Streamlit sign-in code wrote them."""
import hashlib
import os
import sqlite3

import pytest
from django.core.management import CommandError, call_command

from accounts.models import User
from activity.models import ActivityLog
from audit.models import AuditReport, Batch, Claim


def _old_hash(password: str) -> str:
    """The removed src/auth/user.py format: salt hex, "$", PBKDF2-SHA256 hex, 600,000 rounds."""
    salt = os.urandom(16)
    return f"{salt.hex()}${hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 600_000).hex()}"


@pytest.fixture
def old_db(tmp_path):
    from src.reporting.repository import AuditReportRepository
    path = str(tmp_path / "sha_fraud.sqlite")
    repo = AuditReportRepository(path)
    with sqlite3.connect(path) as conn:
        conn.executemany("INSERT INTO User (username, password_hash, role, email, created_at) "
                         "VALUES (?, ?, ?, ?, '2026-10-25T00:00:00+00:00')",
                         [("admin", _old_hash("admin-pass-12"), "Administrator", None),
                          ("auditor", _old_hash("auditor-pass-1"), "Auditor", "auditor@example.com")])
    for ref, verdict, narrative in (("CLM1", "inconsistent", "Notes describe a vaginal delivery."),
                                    ("CLM2", "consistent", None), ("CLM1", "consistent", None)):
        claim_id = repo.save_claim({"claim_ref": ref, "provider_id": "PRV1", "patient_ref": "M1",
                                    "icd11_code": "JB22.1", "claim_amount": 30000.0, "service_date": "2025-06-01",
                                    "discharge_date": "2025-06-03", "clinical_notes": "note", "anomaly_score": 0.6})
        verdict_id = repo.save_verdict(claim_id, verdict, 0.9, "seed7")
        repo.save_report(claim_id, verdict_id, {"claim_amount": 0.2}, narrative, ["isolation_forest"])
    repo.record_decision(1, "approved", reviewer_id=2)
    with sqlite3.connect(path) as conn:
        conn.execute("INSERT INTO ActivityLog (user_id, action, detail, at) "
                     "VALUES (2, 'sign_in', 'password', '2026-10-25T00:00:00+00:00')")
    return path


def test_imports_everything(old_db, db):
    call_command("import_sqlite", old_db)
    auditor = User.objects.get(username="auditor")
    assert auditor.check_password("auditor-pass-1") and auditor.email == "auditor@example.com"
    assert User.objects.get(username="admin").is_administrator
    # CLM1 appears twice in the old database: the repeat goes into a second batch.
    assert list(Batch.objects.order_by("id").values_list("claims_count", flat=True)) == [2, 1]
    assert Claim.objects.count() == 3
    first = AuditReport.objects.order_by("id").first()
    assert (first.status, first.reviewed_by, first.lane) == ("approved", auditor, "text_inconsistent")
    assert first.llm_narrative.startswith("Notes describe") and first.flag_sources == ["isolation_forest"]
    assert first.shap_attribution == {"claim_amount": 0.2}
    assert AuditReport.objects.filter(status="pending").count() == 2
    assert ActivityLog.objects.get().user == auditor


def test_refuses_to_import_twice(old_db, db):
    call_command("import_sqlite", old_db)
    with pytest.raises(CommandError, match="already"):
        call_command("import_sqlite", old_db)
