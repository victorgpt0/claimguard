"""Celery for the web application's background jobs (docs/platform_plan.md, phase 3). Two
queues: "cpu" for a local worker, "gpu" for the Colab worker notebook. Progress is kept in
the Job table, so no result backend is needed.

    cd web && celery -A claimguard_web worker -Q cpu -c 1 -n cpu@%h     # local
    cd web && celery -A claimguard_web worker -Q gpu -c 1 -n gpu@%h     # Colab (notebook 05)
"""
import os
import sys
from pathlib import Path

from celery import Celery

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # the repository, for src/
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "claimguard_web.settings")

app = Celery("claimguard_web")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()
