# Claims ingestion: the simulated SHA claims platform and the claims feed (3 Nov)

Issues #64 (live data set), #65 (claims platform), #66 (ingestion pipeline and Data pipeline page).

## The problem

The dashboard showed batches an auditor uploaded by hand, but not where claims come from.
In SHA's system the claims do not belong to the audit side. Providers lodge them on the
Centralized Digital Platform, where they are reviewed, approved and paid, and a post-payment
audit reads the paid ones from there. ClaimGuard runs on synthetic claims generated on Colab,
so the demonstration needs a stand-in for that platform and a data pipeline from it, and the
dashboard has to show that pipeline.

Detection must also stay batch-based. The provider- and member-level features
(`provider_caesarean_share`, `member_claim_count`, `member_dependent_claims`;
`src/data/preprocessing.py`) are computed across a batch and its history. A claim scored alone
would hide the provider-level typologies (inflated caesareans, anomalous dependants).

## What SHA's rules say (checked 3 Nov against the primary source)

| Rule | Source |
|---|---|
| All claims are lodged, reviewed, processed, validated, appraised and paid through the Centralized Digital Platform | Social Health Insurance Regulations 2024, Legal Notice 49 of 2024 (Kenya Gazette Supplement 60, 8 March 2024), reg. 57(4) |
| A provider lodges a claim within 7 days of the patient's discharge | LN 49/2024, reg. 58(1) |
| The Claims Management Office reviews each claim: approves it for payment, returns it for amendment, or rejects it (notice within 14 days) | LN 49/2024, reg. 59(1) |
| Approved claims are paid within 90 days of approval | Social Health Insurance (Amendment) Regulations 2024; Oraro & Company Advocates, *Highlights of the Social Health Insurance (Amendment) Regulations, 2024* (legal alert, September 2024) |

Neither instrument sets an audit cycle. The audit period is therefore a configurable design
choice, monthly by default.

## Design

```
 Simulated SHA platform (sha_cdp/)          ClaimGuard (web/ingestion/)               existing engine
 ┌──────────────────────────────┐   HTTPS    ┌──────────────────────────────────┐    ┌────────────────┐
 │ own DB: claims + times       │  read-only │ beat every FEED_SYNC_SECONDS     │    │ Batch → Job →  │
 │ (no fraud labels)            │ ─────────▶ │ pull after cursor (paid_at, seq) │──▶ │ AuditPipeline  │
 │ clock: releases paid claims  │  API key   │ check per claim → quarantine     │    │ → review queue │
 │ API: /claims/paid, /stats,   │            │ dedupe → open batch of period    │    └────────────────┘
 │      /clock (operator token) │            │ period ended → close → queue job │
 └──────────────────────────────┘            └──────────────────────────────────┘
      ▲ python -m sha_cdp.seed live.parquet          Data pipeline page (/pipeline/)
      │ (labels dropped here)
  notebooks/06_generate_live_feed.ipynb (Colab): live-v1, seed 2026, Oct 2025 – Mar 2026
```

**Live data (#64).** A fresh set: seed 2026, 2,400 claims, service dates from October 2025 to
March 2026 (after the training window), v0.6's 240 providers, `L`-prefixed claim and member
ids. Every claim is unseen input. See `docs/data_generation.md`, "Live claims feed".

**The platform (#65), `sha_cdp/`.** A separate FastAPI service with its own database
(`SHA_CDP_DATABASE_URL`: a second Neon database, or a local SQLite file). Nothing in `src/` or
`web/` imports it. The seeder assigns SHA's processing times: lodged 0–7 days after discharge,
approved 1–14 days later, paid 14–60 days after approval (inside the 90-day limit). It keeps
only the platform's columns, so the ground truth never enters it. A simulation clock (start,
advance, play, pause) decides which claims have been paid "so far". Only paid claims are
exported, because a post-payment audit sees a claim once SHA has paid it. Paging uses the
cursor (paid_at, seq).

**The feed (#66), `web/ingestion/`.**
- Celery beat sends `sync_claims_feed` to its own `ingest` queue, so a pull never waits
  behind a batch job.
- A pull reads pages after the cursor and checks each claim with the upload rules (FR-15).
  Bad claims are quarantined with their problems. Claim ids already stored are counted and
  skipped.
- Each claim goes into the open batch of its audit period, by payment date. Each page is
  stored with its cursor in one transaction, so a failed pull resumes where it stopped.
- A period closes only once everything paid up to the platform's clock has been pulled. Its
  CSV is then written exactly as an upload's is, and (with `ingestion.auto_process`) a job is
  queued. Manual upload stays as the ad hoc path.

**Configuration** (`config.yaml`, all optional):

```yaml
ingestion:
  audit_period: monthly   # monthly | fortnightly | weekly
  auto_process: true      # queue a job when a period closes
  queue: cpu              # cpu (no narratives) | gpu (Colab worker, with narratives)
```

**Page (`/pipeline/`, "Data pipeline").** It shows:
- the flow: platform paid/lodged → ingested (quarantined) → collecting → detection → review
  queue;
- the platform's status and clock;
- the feed's cursor and health;
- the audit periods with their claims, flags, reports and job;
- the pulls and the quarantine.

Auditors can pull now and close a period early. Administrators also pause scheduled pulls
and move the platform's clock; every move is followed by a pull. The page refreshes every
5 seconds.

## Running the demonstration

```bash
# 1. once: generate live-v1 on Colab (notebook 06), then load it into the platform
.venv/bin/python -m sha_cdp.seed ~/gdrive/claimguard/data/processed/live-v1/live.parquet
# 2. the platform, the web app, and two workers (ingest with beat; cpu for batches)
.venv/bin/uvicorn sha_cdp.app:app --port 8100
cd web && ../.venv/bin/python manage.py runserver
cd web && ../.venv/bin/celery -A claimguard_web worker -Q ingest -c 1 -n ingest@%h -B
cd web && ../.venv/bin/celery -A claimguard_web worker -Q cpu -c 1 -n cpu@%h
# 3. Data pipeline page, as Administrator: Advance 30 days (or Play) and watch periods close
```

Both sides need the same `SHA_CDP_API_KEY` and `SHA_CDP_ADMIN_TOKEN` (`.env.example`).

## Chapter 3 and 4 changes this needs

Paste-ready text is in `compare/changes.txt`, section E.
- **FR-01** becomes scheduled, incremental ingestion of paid claims from the claims platform,
  per audit period, with manual upload kept for ad hoc batches.
- **New IR-09:** the claims platform interface.
- **Figure 4.1** (architecture) gains the platform and the ingestion component.
- **Figure 4.9** (Batch state machine) gains `open`.
- **Chapter 5** states that the platform is simulated, because there is no access to SHA's.

## Limitations to state

- The platform is a simulation. Its processing times are drawn within the regulatory limits,
  not measured from SHA.
- Live claims reuse v0.6's providers, so a provider that inflates caesareans in the live set
  is diluted by its normal training-period deliveries (`docs/data_generation.md`).
- The first and last audit periods are partial, because payments ramp up and tail off.
