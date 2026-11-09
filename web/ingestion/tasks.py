"""The scheduled pull (#66): Celery beat sends sync_claims_feed every FEED_SYNC_SECONDS to the
"ingest" queue, which a small worker of its own serves, so a pull never waits behind a
long batch job on the cpu queue:

    cd web && celery -A claimguard_web worker -Q ingest -c 1 -n ingest@%h -B    # -B runs beat in it
"""
from __future__ import annotations

from celery import shared_task

from . import sync


@shared_task(ignore_result=True)
def sync_claims_feed() -> str:
    try:
        run = sync.sync(scheduled=True)
    except sync.Busy:
        return "busy"
    return "paused" if run is None else run.status
