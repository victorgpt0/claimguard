"""A stand-in for SHA's Centralized Digital Platform (#65, docs/ingestion_plan.md).

In SHA's system the audit side does not own the claims: providers lodge them on the
Centralized Digital Platform, where they are reviewed, approved and paid (Social Health
Insurance Regulations 2024, LN 49/2024, reg. 57(4)), and a post-payment audit reads the paid
ones from there. This package plays that platform for the demonstration: a separate service
with its own database, which ClaimGuard reaches only through a read-only, token-protected API
(IR-09), the way it would reach SHA's.

    db.py     its database: the claims, with their lodged/approved/paid times, and the clock
    clock.py  the simulation clock that releases claims as simulated time passes
    seed.py   loads a claims set (the live feed, live.parquet) without its ground-truth labels
    app.py    the API: paid claims after a cursor, platform statistics, clock control

Run it apart from the web application:

    python -m sha_cdp.seed <live.parquet>
    uvicorn sha_cdp.app:app --port 8100

Nothing here imports the audit system (src/, web/), and nothing there imports this: the API
is the only link.
"""
