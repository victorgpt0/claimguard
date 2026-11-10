"""Django-side helper for 08-ops-check.sh. deploy/ is kept out of the images, so the script
pipes this file into the web container:

    docker compose exec -T web python - <command> [args] < scripts/ops_helper.py

Commands (each prints one line of JSON, nothing secret except where stated):
  session <role>      a signed-in session for the first active user with that role; prints
                      {"key": ..., "user": ...}. The key is a credential: the script keeps it in
                      a 0600 file and deletes the session at the end. No account is created.
  expire <key>        moves that session's expiry into the past (the expired-session test)
  drop <key>...       deletes those sessions
  config              the settings the checks depend on
  counts              rows the restart test compares before and after
  batch               a processed batch to aim the duplicate-submission test at
  report              a report id for the timing run
  jobs <batch>        how many jobs that batch has
  active              jobs queued or running (the disruptive checks refuse to run if any)
  workers             which queues have a worker online
  stats [days]        job and claims-feed outcomes over the last days (default 30)
"""
from __future__ import annotations

import json
import os
import sys
from datetime import timedelta

sys.path.insert(0, "/app/web")
sys.path.insert(0, "/app")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "claimguard_web.settings")

import django  # noqa: E402

django.setup()

from django.conf import settings  # noqa: E402
from django.contrib.auth import BACKEND_SESSION_KEY, HASH_SESSION_KEY, SESSION_KEY, get_user_model  # noqa: E402
from django.contrib.sessions.backends.db import SessionStore  # noqa: E402
from django.contrib.sessions.models import Session  # noqa: E402
from django.db.models import Avg, Count, F  # noqa: E402
from django.utils import timezone  # noqa: E402

from audit import services  # noqa: E402
from audit.models import AuditReport, Batch, Claim, Job  # noqa: E402
from ingestion.models import IngestionRun  # noqa: E402


def out(value) -> None:
    print(json.dumps(value, default=str))


def session(role: str) -> None:
    user = get_user_model().objects.filter(role=role, is_active=True).order_by("pk").first()
    if user is None:
        out({"error": f"no active {role}"})
        return
    store = SessionStore()
    store[SESSION_KEY] = str(user.pk)
    store[BACKEND_SESSION_KEY] = "django.contrib.auth.backends.ModelBackend"
    store[HASH_SESSION_KEY] = user.get_session_auth_hash()
    store.set_expiry(settings.SESSION_COOKIE_AGE)
    store.create()
    out({"key": store.session_key, "user": user.get_username()})


def expire(key: str) -> None:
    n = Session.objects.filter(session_key=key).update(expire_date=timezone.now() - timedelta(minutes=1))
    out({"expired": n})


def drop(*keys: str) -> None:
    out({"deleted": Session.objects.filter(session_key__in=keys).delete()[0]})


def config() -> None:
    out({"session_timeout_minutes": settings.SESSION_COOKIE_AGE // 60,
         "session_save_every_request": settings.SESSION_SAVE_EVERY_REQUEST,
         "debug": settings.DEBUG, "conn_max_age": settings.DATABASES["default"].get("CONN_MAX_AGE"),
         "conn_health_checks": settings.DATABASES["default"].get("CONN_HEALTH_CHECKS")})


def counts() -> None:
    out({"batches": Batch.objects.count(), "claims": Claim.objects.count(), "reports": AuditReport.objects.count(),
         "jobs": Job.objects.count(), "ingestion_runs": IngestionRun.objects.count()})


def batch() -> None:
    b = Batch.objects.filter(status=Batch.Status.PROCESSED).order_by("pk").first()
    out({"batch": b.pk if b else None})


def report() -> None:
    r = AuditReport.objects.order_by("pk").first()
    out({"report": r.pk if r else None})


def jobs(pk: str) -> None:
    out({"jobs": Job.objects.filter(batch_id=int(pk)).count()})


def active() -> None:
    out({"active": Job.objects.filter(status__in=[Job.Status.QUEUED, Job.Status.RUNNING]).count()})


def workers() -> None:
    out(services.workers_online())


def stats(days: str = "30") -> None:
    since = timezone.now() - timedelta(days=int(days))
    finished = Job.objects.filter(created_at__gte=since, finished_at__isnull=False)
    by_queue = {}
    for row in (finished.values("queue", "status").annotate(n=Count("pk"))):
        by_queue.setdefault(row["queue"], {})[row["status"]] = row["n"]
    durations = {}
    for row in (finished.filter(status=Job.Status.SUCCEEDED, started_at__isnull=False)
                .values("queue").annotate(mean=Avg(F("finished_at") - F("started_at")), n=Count("pk"))):
        durations[row["queue"]] = {"succeeded": row["n"], "mean_seconds": round(row["mean"].total_seconds(), 1)}
    per_claim = {}
    for j in finished.filter(status=Job.Status.SUCCEEDED, started_at__isnull=False).select_related("batch"):
        if j.batch.claims_count:
            per_claim.setdefault(j.queue, []).append((j.finished_at - j.started_at).total_seconds() / j.batch.claims_count)
    for queue, xs in per_claim.items():
        durations.setdefault(queue, {})["mean_seconds_per_claim"] = round(sum(xs) / len(xs), 3)
        durations[queue]["max_seconds_per_claim"] = round(max(xs), 3)
    failures = [{"job": j.pk, "queue": j.queue, "at": j.finished_at, "reason": j.message.splitlines()[0][:160]}
                for j in finished.filter(status=Job.Status.FAILED).order_by("-finished_at")[:10]]
    runs = IngestionRun.objects.filter(started_at__gte=since)
    out({"days": int(days), "jobs": by_queue, "job_durations": durations, "recent_job_failures": failures,
         "ingestion_runs": {r["status"]: r["n"] for r in runs.values("status").annotate(n=Count("pk"))}})


COMMANDS = {f.__name__: f for f in (session, expire, drop, config, counts, batch, report, jobs, active, workers, stats)}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    COMMANDS[sys.argv[1]](*sys.argv[2:])
