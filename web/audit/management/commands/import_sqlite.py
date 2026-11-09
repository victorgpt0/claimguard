"""Imports the Streamlit-era SQLite database (src/db/schema.sql) into this application's
database: accounts (with their existing passwords, accounts.hashers), claims, verdicts,
audit reports with their decisions and reviewers, and the activity log.

The old database records no batches, and the same claim id can appear in more than one
earlier run (e.g. the v0.5 and v0.6 pools), so claims go into "Imported" batches: a claim
id's first occurrence into the first, a repeat into the next. Runs in one transaction and
refuses to run twice. Rows are inserted in bulk, so a hosted database far away (Neon) gets a
few queries rather than one per row.

    python web/manage.py import_sqlite ~/gdrive/claimguard/db/sha_fraud.sqlite
"""
from __future__ import annotations

import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from accounts.hashers import from_legacy
from accounts.models import User
from activity.models import ActivityLog
from audit.models import AuditReport, Batch, Claim, ClassifierVerdict


def _when(value) -> datetime | None:
    """ISO strings with or without an offset, and SQLite's CURRENT_TIMESTAMP (UTC)."""
    if not value:
        return None
    moment = datetime.fromisoformat(str(value).replace(" ", "T"))
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _json(value, empty):
    if value in (None, ""):
        return empty
    return json.loads(value) if isinstance(value, str) else value


def _restore_times(objects, rows, field, column):
    """auto_now_add overwrites a timestamp on insert, so the original one is set afterwards."""
    changed = []
    for obj, row in zip(objects, rows):
        moment = _when(row[column])
        if moment:
            setattr(obj, field, moment)
            changed.append(obj)
    if changed:
        type(changed[0]).objects.bulk_update(changed, [field], batch_size=500)


class Command(BaseCommand):
    help = "Import the Streamlit-era SQLite database (accounts, claims, reports, decisions, activity)."

    def add_arguments(self, parser):
        parser.add_argument("sqlite_path", type=Path)

    def handle(self, sqlite_path: Path, **options):
        if not sqlite_path.exists():
            raise CommandError(f"{sqlite_path} not found")
        if Batch.objects.filter(status=Batch.Status.IMPORTED).exists():
            raise CommandError("an import has already been done; refusing to import twice")
        source = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
        source.row_factory = sqlite3.Row
        try:
            with transaction.atomic():
                counts = self._import(source)
        finally:
            source.close()
        self.stdout.write(self.style.SUCCESS(", ".join(f"{n} {what}" for what, n in counts.items())))

    def _tables(self, source) -> set[str]:
        return {row[0] for row in source.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}

    def _columns(self, source, table) -> set[str]:
        return {row[1] for row in source.execute(f"PRAGMA table_info({table})")}

    def _import(self, source) -> dict:
        tables = self._tables(source)
        users = self._users(source) if "User" in tables else {}
        claims, batches = self._claims(source)
        rows = list(source.execute("SELECT * FROM ClassifierVerdict ORDER BY id"))
        created = ClassifierVerdict.objects.bulk_create([ClassifierVerdict(
            claim=claims[row["claim_id"]], verdict=row["verdict"], confidence=row["confidence"],
            model_version=row["model_version"]) for row in rows])
        verdicts = {row["id"]: verdict for row, verdict in zip(rows, created)}
        has_sources = "flag_sources" in self._columns(source, "AuditReport")
        rows = list(source.execute("SELECT * FROM AuditReport ORDER BY id"))
        reports = AuditReport.objects.bulk_create([AuditReport(
            claim=claims[row["claim_id"]], verdict=verdicts.get(row["verdict_id"]),
            shap_attribution=_json(row["shap_attribution"], {}), llm_narrative=row["llm_narrative"],
            flag_sources=_json(row["flag_sources"], []) if has_sources else [],
            status=row["status"], reviewed_by=users.get(row["reviewed_by"]),
            reviewed_at=_when(row["reviewed_at"])) for row in rows])
        _restore_times(reports, rows, "created_at", "created_at")
        activity = self._activity(source, users) if "ActivityLog" in tables else 0
        return {"users": len(users), "batches": len(batches), "claims": len(claims),
                "verdicts": len(verdicts), "reports": len(reports), "activity entries": activity}

    def _users(self, source) -> dict[int, User]:
        columns = self._columns(source, "User")
        mapped = {}
        for row in source.execute("SELECT * FROM User ORDER BY id"):
            if User.objects.filter(username=row["username"]).exists():
                raise CommandError(f"user {row['username']!r} already exists here; import into an empty database")
            user = User(username=row["username"], role=row["role"], password=from_legacy(row["password_hash"]),
                        email=(row["email"] or "") if "email" in columns else "",
                        is_active=bool(row["active"]) if "active" in columns else True)
            user.save()
            joined = _when(row["created_at"]) if "created_at" in columns else None
            if joined:
                User.objects.filter(pk=user.pk).update(date_joined=joined)
            mapped[row["id"]] = user
        return mapped

    def _claims(self, source) -> tuple[dict[int, Claim], list[Batch]]:
        has_ref = "claim_ref" in self._columns(source, "Claim")
        seen: Counter = Counter()
        batches: list[Batch] = []
        rows = list(source.execute("SELECT * FROM Claim ORDER BY id"))
        pending = []
        for row in rows:
            ref = (row["claim_ref"] if has_ref else None) or f"legacy-{row['id']}"
            occurrence = seen[ref]
            seen[ref] += 1
            while len(batches) <= occurrence:
                batches.append(Batch.objects.create(name=f"Imported from SQLite ({len(batches) + 1})",
                                                    status=Batch.Status.IMPORTED))
            pending.append(Claim(
                batch=batches[occurrence], claim_ref=ref, provider_id=row["provider_id"],
                patient_ref=row["patient_ref"], icd11_code=row["icd11_code"],
                claim_amount=Decimal(str(row["claim_amount"])), service_date=row["service_date"][:10],
                discharge_date=(row["discharge_date"] or "")[:10] or None,
                clinical_notes=row["clinical_notes"], anomaly_score=row["anomaly_score"]))
        created = Claim.objects.bulk_create(pending)
        claims = {row["id"]: claim for row, claim in zip(rows, created)}
        per_batch = Counter(claim.batch_id for claim in created)
        for batch in batches:
            batch.claims_count = per_batch[batch.pk]
        Batch.objects.bulk_update(batches, ["claims_count"])
        return claims, batches

    def _activity(self, source, users) -> int:
        rows = list(source.execute("SELECT * FROM ActivityLog ORDER BY id"))
        entries = ActivityLog.objects.bulk_create([ActivityLog(
            user=users.get(row["user_id"]), action=row["action"], detail=(row["detail"] or "")[:255])
            for row in rows])
        _restore_times(entries, rows, "at", "at")
        return len(entries)
