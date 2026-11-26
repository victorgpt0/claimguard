# ClaimGuard

Fine-Tuned Clinical Language Models for Explainable Post-Payment Fraud Audit: A Case Study of Kenya's Social Health Authority

This repo is the implementation only; the written project documentation (methodology,
system design, results) is kept outside it.

- **Live system:** https://claimsguard.tech (Oracle Cloud VM, `deploy/README.md`)
- **Project page:** https://victorgpt0.github.io/claimguard/ (`site/`, published by
  `.github/workflows/pages.yml`)
- **Frozen classifier:** Hugging Face Hub `victorgpt0/claimguard-consistency-classifier`,
  revision tag `v1.0-evaluated` (the synthetic dataset is not published)

## Status (25 Nov 2026)

Implemented and evaluated: the three-stage pipeline, the v0.6-20k dataset, the classifier
(five seeds, F1 0.975 ± 0.004 on validation), the baselines, the held-out test evaluation
(read once, 27 Oct), the feedback retraining run, the narrative and SHAP checks, the Django
platform with the simulated SHA claims feed, escalation and fraud cases, and the deployment
with its operational checks. Every result is recorded in `tests/test_plan.md` (TC-001 to
TC-056). Still open: one uploaded batch processed by the Colab GPU worker against the
deployed release (TC-040), a usability test with a representative user (TC-015), and the
`v1.0` tag (#44). The checklist is `docs/sprint_plan.md`.

## Problem

Kenya's Social Health Authority (SHA) lost an estimated KES 11 billion to medical
insurance fraud within six months of launch. Existing detection relies on static,
rule-based filters over structured billing fields, which can't check whether a claim's
free-text clinical justification is semantically consistent with its billed ICD-11
procedure code. ClaimGuard is a hybrid pipeline: unsupervised anomaly detection
(Isolation Forest + Autoencoder) filters structured billing outliers; a fine-tuned
clinical transformer classifier (transfer learning, not a prompted LLM) predicts
whether the flagged claim's clinical justification is consistent with its billed code; an
LLM, invoked only for claims the classifier flags inconsistent, converts the verdict into
a plain-language audit narrative grounded in the WHO ICD-11 API.

## Architecture

Each module below maps directly to a class in the system design (Chapter 4, Class
Diagram), so the OOAD-to-code mapping stays visible as the project develops:

```
src/
├── config.py          # loads config/config.yaml + .env
├── data/              # dataset generation (stages A-C), ClaimsPreprocessor, Table 3.1 injection
├── detection/         # IsolationForestDetector, AutoencoderDetector
├── explainability/     # SHAPExplainer
├── classification/     # SemanticConsistencyClassifier + LoRA/PEFT fine-tuning
├── icd11/               # ICD11APIClient + ICD11Cache (see docs/environment.md) + refresh_cache CLI
├── narrative/            # AuditNarrativeGenerator (LLM, explanation-only)
├── reporting/            # AuditReportGenerator, AuditReportRepository
├── feedback/              # FeedbackRepository (FR-12), ClassifierRetrainer (FR-13/FR-14)
├── baselines/             # rule-based, structured-only and zero-shot LLM baselines
├── evaluation/            # held-out comparison, cost, faithfulness, rubric, SHAP coherence
├── release/               # publishing the frozen classifier to the Hugging Face Hub
└── db/                    # SQLite schema the pipeline writes reports to (imported into Django)
web/                         # Django application (IR-03, IR-04, IR-07, IR-08, IR-09)
├── claimguard_web/          # settings (env-driven), URLs, Celery app
├── accounts/                # User with role, allauth adapters (Google, no sign-up), admin site
├── activity/                # ActivityLog (FR-18)
├── audit/                   # Batch, Claim, ClassifierVerdict, AuditReport, Job, FraudCase; pages;
│                            # upload validation (FR-15); process_batch task; batch summary (FR-16);
│                            # review, escalation and fraud cases
└── ingestion/               # scheduled pull of paid claims from sha_cdp, Data pipeline page
sha_cdp/                     # simulated SHA claims platform: own service, database and API
notebooks/                   # Colab notebooks: dataset, training, baselines, GPU worker, live feed
deploy/                      # Oracle Cloud deployment: compose file, scripts, runbook
site/                        # project page (GitHub Pages)
docker-compose.yml           # PostgreSQL, Redis, the web service, WHO ICD-11 API
docs/diagrams/               # Chapter 4 diagrams for the Django platform (PlantUML + PNG)
```

## Setup

1. **Clone and install.** Dependencies are locked in `uv.lock` (`pyproject.toml` groups:
   `web`, `ml`, `dev`; torch comes from the CPU-only index locally):
   ```bash
   git clone git@github.com:victorgpt0/claimguard.git
   cd claimguard
   python3 -m venv .venv && .venv/bin/pip install uv
   .venv/bin/uv sync --frozen        # everything; or: --no-default-groups --group web --group dev
   ```
   `requirements.txt` is kept for the older `venv/`; `requirements/colab-worker.txt` is what
   the Colab GPU worker installs. Every command below assumes `.venv`.

2. **Configure:**
   ```bash
   cp config/config.example.yaml config/config.yaml   # paths to the Drive folders; keep pipeline.dataset
   cp .env.example .env                               # without DATABASE_URL the web app runs on SQLite
   ```
   Fill in `.env` as `.env.example` describes (HF_TOKEN / OPENAI_API_KEY, database, Redis,
   Google sign-in). `config.yaml` and `.env` are gitignored — never commit them. On Colab,
   use Colab Secrets (`google.colab.userdata`) instead of `.env`.

3. **WHO ICD-11 API (self-hosted, only needed to refresh the cache):**
   ```bash
   docker compose --profile icd11 up -d icd11-api
   curl http://localhost:8081/icd/release/11/2026-01/mms  # sanity check it's up
   .venv/bin/python -m src.icd11.refresh_cache --codes-file data/processed/icd11_codes_used.txt \
       --cache-path artifacts/icd11_cache.json
   ```
   No OAuth2 credentials are needed for a local deployment. Everything else reads the cache
   file, never the live API, so it also runs on Colab (see `docs/environment.md`).

## Web application (Django)

The Django application in `web/` replaces the Streamlit dashboard (`docs/platform_plan.md`):
accounts with the two roles and Google sign-in for linked accounts, batch upload with full
validation, processing as background jobs on a local CPU worker or a Colab GPU worker, the
review queue and report pages, a summary report per batch, user management and an activity
log. The Streamlit dashboard it replaced was removed on 1 Nov (it is in the git history).

After Setup, the quickest run is SQLite plus a local Redis:

```bash
docker compose up -d redis
cd web
../.venv/bin/python manage.py migrate
../.venv/bin/python manage.py createsuperuser    # then set the Administrator role in /admin
../.venv/bin/python manage.py runserver          # http://localhost:8000
# second terminal, from the repo root: the CPU worker (see Batches and jobs)
```

**Database and queue.** Set in `.env` (`.env.example` explains each variable):

| Setup | `DATABASE_URL` | `REDIS_URL` |
|---|---|---|
| Quick local run | unset (SQLite) | `redis://localhost:6379/0` (`docker compose up -d redis`) |
| Local PostgreSQL | `postgres://claimguard:claimguard-dev@localhost:5432/claimguard` (`docker compose up -d db redis`) | as above |
| Deployed (claimsguard.tech) | PostgreSQL on the Oracle VM, set up by `deploy/` | Redis on the Oracle VM; the Colab GPU worker reaches both over TLS |
| Hosted, earlier setup (before 9 Nov) | Neon: `neon deploy` writes `DATABASE_URL` (pooled) and `DATABASE_URL_UNPOOLED` into `.env` | Upstash: `rediss://default:<password>@<db>.upstash.io:6379` |

The settings still support Neon and Upstash: they verify Upstash's TLS certificate and turn
off server-side cursors on Neon's pooled endpoint. On Neon, run migrations and the one-off
import on the direct endpoint:

```bash
cd web
export $(grep '^DATABASE_URL_UNPOOLED=' ../.env | tr -d '"')     # .env is not a shell file
DATABASE_URL=$DATABASE_URL_UNPOOLED ../.venv/bin/python manage.py migrate
DATABASE_URL=$DATABASE_URL_UNPOOLED ../.venv/bin/python manage.py import_sqlite ~/gdrive/claimguard/db/sha_fraud.sqlite   # once
../.venv/bin/python manage.py runserver                       # http://localhost:8000
```

Imported accounts keep their passwords. A new database without an import needs
`manage.py createsuperuser`, then the Administrator role set in the admin site.
`docker compose up -d --build db redis web` runs the web service in a container against the
local PostgreSQL and Redis; the container cannot see the rclone Drive mount (FUSE, see
docker-compose.yml), so uploads, ICD-11 definitions and the monitoring pages need the
`.venv` setup, which reads Drive directly.

**Roles** (a separation of duties: whoever decides what is audited does not judge the
results). The Fraud Auditor sees Overview and Review queue, where each report is decided as
Confirm fraud, Dismiss flag or Escalate. The Administrator sees Overview, Data pipeline,
Batches (upload, processing, summaries), the Escalated queue (the senior reviewer decides
escalated reports), Fraud cases, Model performance, Feedback loop, and Users and Activity
log in Django's admin site, where accounts are
created, deactivated (never deleted), given a role and linked to a Google email, and the
log is filtered and exported. Everyone changes their own password from their name in the
top bar. A session idle for 30 minutes (`SESSION_TIMEOUT_MINUTES`) is signed out.

**Fraud cases.** Confirming fraud (directly or after escalation) opens one fraud case with
the claim amount as the recommended recovery and a log of enforcement actions drawn from
the measures SHA's law provides (recovery, suspension, deregistration, referral). Dismissing
opens none. Cases record recommendations, not real enforcement.

**Google sign-in** needs `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET` in `.env` and, on the
OAuth client in Google Cloud Console, every address the app is opened at as an authorised
redirect URI, e.g. `http://localhost:8000/accounts/google/login/callback/` and the same with
`127.0.0.1` (Google treats them as different; a missing one gives `redirect_uri_mismatch`).
Only accounts whose email the Administrator has linked can sign in with Google.

**Batches and jobs.** The Administrator uploads a claims CSV under Batches (a template is
offered, at most 2,000 claims and 10 MB); every line is checked and nothing is stored unless
all pass, and held-out test claims are always refused (DR-08). Processing runs as a Celery
job on one of two queues, with live progress on the batch page:

```bash
# Local CPU worker (stage 1, SHAP, classifier; no narratives), with the same .env as the web
# process. config.yaml needs pipeline.dataset (see config.example.yaml).
.venv/bin/celery --workdir web -A claimguard_web worker -Q cpu -c 1 -n cpu@%h --without-gossip --without-mingle -l INFO
```

The GPU queue (with narratives) is served by `notebooks/05_gpu_worker.ipynb` on Colab, with
the pooled `DATABASE_URL` and the `REDIS_URL` in the notebook's Colab Secrets and this branch
pushed. The uploaded CSV is kept in the database, so either worker can process it. With
Upstash's free tier, stop idle workers: an idle worker still polls Redis. A failed job
leaves no partial reports, and the batch can be started again.

**Outputs.** A processed (or imported) batch shows its summary: claims flagged by each
detector, cleared and queued per lane, narratives, amount under audit, review progress per
reviewer, top providers and ICD-11 codes among queued claims, and the classifier version and
thresholds. It downloads as Markdown, and the batch's reports as CSV (`web/audit/summary.py`).

Tests: see Testing below. Design diagrams for Chapter 4 are in
`docs/diagrams/`.

## Simulated SHA claims platform (`sha_cdp/`)

ClaimGuard does not own the claims it audits. In SHA's system, providers lodge claims on the
Centralized Digital Platform, where they are reviewed, approved and paid (LN 49/2024,
regs. 57(4), 58, 59), and a post-payment audit reads the paid ones from there. `sha_cdp/`
stands in for that platform (#65): a **separate service with its own database**, which the
audit system reaches only through a read-only API (IR-09). Nothing in `src/` or `web/`
imports it.

```bash
# its database: SHA_CDP_DATABASE_URL (a second Neon database), else data/sha_cdp.sqlite
.venv/bin/python -m sha_cdp.seed ~/gdrive/claimguard/data/processed/live-v1/live.parquet
.venv/bin/uvicorn sha_cdp.app:app --port 8100        # API docs at http://localhost:8100/docs
```

The seeder gives each claim SHA's processing times (lodged within 7 days of discharge,
approved after review, paid within 90 days of approval) and **drops the ground-truth
labels**. A simulation clock releases claims as simulated time passes. It starts paused
before the first payment and is moved from the Data pipeline page, or with
`POST /api/v1/clock` and the operator token.

| Endpoint | Token | Returns |
|---|---|---|
| `GET /api/v1/claims/paid?after_paid_at=&after_seq=&limit=` | `SHA_CDP_API_KEY` | claims paid by the clock's time, after the cursor, oldest first |
| `GET /api/v1/stats` | `SHA_CDP_API_KEY` | claims lodged, approved and paid so far; next payment |
| `POST /api/v1/clock` `{action: advance/play/pause/set}` | `SHA_CDP_ADMIN_TOKEN` | the clock |
| `GET /health` | none | status and clock |

The live claims set comes from `notebooks/06_generate_live_feed.ipynb` (`docs/data_generation.md`).

### The claims feed and the Data pipeline page (#66)

The web app pulls paid claims from the platform on a schedule, checks each one (bad ones are
quarantined), and collects them into one batch per **audit period** (monthly by default;
`config.yaml` `ingestion.audit_period`: monthly, fortnightly or weekly). When the platform's
clock passes a period's last day, the batch is closed and queued for detection like an
upload. Detection stays per batch because provider- and member-level patterns only show
across many claims. The **Data pipeline** page shows the whole flow. Design and the SHA rules
behind it: `docs/ingestion_plan.md`.

```bash
cd web && ../.venv/bin/celery -A claimguard_web worker -Q ingest -c 1 -n ingest@%h -B   # pulls (beat inside)
cd web && ../.venv/bin/celery -A claimguard_web worker -Q cpu -c 1 -n cpu@%h            # closed periods
```

Settings (`.env`): `SHA_CDP_URL`, `SHA_CDP_API_KEY`, `SHA_CDP_ADMIN_TOKEN` (the same values the
platform uses) and `FEED_SYNC_SECONDS` (30).

## Dataset

The synthetic claims dataset is built in three stages (Section 3.2.1); all three run from
`notebooks/01_generate_dataset.ipynb` on a Colab T4, writing to Google Drive.

| Stage | Command | Runs on |
|---|---|---|
| A. Structured claims, provider-level typologies | `python -m src.data.generator --n-claims 5000 --out data/raw` | CPU |
| B. LLM clinical notes (data synthesis only) | `python -m src.data.note_synthesis --requests data/raw/note_requests.jsonl --out data/raw/notes.jsonl --boilerplate-out data/raw/boilerplate_notes.jsonl` | GPU (vLLM, or `--backend transformers`) |
| C. Claim-level typologies, labels, splits, noise | `python -m src.data.build_dataset --raw data/raw --out data/processed` | CPU |

- **Conditions and tariffs:** `src/data/sha_catalogue.yaml`. 37 ICD-11 codes verified
  against the WHO ICD-11 API (MMS 2026-01), priced from Legal Notice No. 56 of 2025
  (SHA Tariffs for Healthcare Services). Outpatient care is capitation-paid (KES 0 per
  claim), medical admissions are per diem by facility level, and maternity and surgery
  are case rates.
- **Fraud:** ~10% of claims, split evenly across the six Table 3.1 typologies, one
  typology per claim. `is_fraud` marks any typology, and `label` is the classifier target:
  `inconsistent` for upcoding, ghost billing and record fabrication, and `consistent`
  otherwise.
- **Splits:** a 10% Deployment/Demonstration Pool (DR-08) is carved off first, then the
  rest is split 60/20/20, stratified by typology. The test split and pool get
  shorthand-noise forms and fabrication notes never seen in training (DR-03).
- `data/processed/manifest.json` records the git commit, input file hashes, parameters
  and per-split counts.

## Running the pipeline and the evaluation (command line)

Claims uploaded in the web application are processed as jobs by the CPU worker or the
Colab GPU worker, which write straight to the shared database (see Web application). The
research runs use the modules below instead; each has a usage block in its docstring and
`--help`. The notebooks (`notebooks/README.md`) run the GPU steps on Colab. Run from the
repo root with `.venv/bin/python -m ...`.

| Step | Command |
|---|---|
| Export classifier input | `python -m src.classification.export --data data/processed/v0.6-20k --cache artifacts/icd11_cache.json --out data/classifier/v0.6-20k` |
| Fine-tune the classifier, 5 seeds (GPU) | `python -m src.classification.train --train-file <dir>/train.jsonl --val-file <dir>/validation.jsonl --experiments-dir experiments/classifier` |
| Seed summary | `python -m src.classification.summarize_seeds --runs experiments/classifier` |
| Stratified classifier evaluation | `python -m src.classification.evaluate --checkpoint <run dir> --data data/classifier/v0.6-20k --split validation` |
| Tune stage 1 (writes `thresholds:` values) | `python -m src.detection.tune_stage1 --data data/processed/v0.6-20k --cache artifacts/icd11_cache.json` |
| Run the pipeline on a batch | `python -m src.pipeline --data data/processed/v0.6-20k --batch deployment_pool --cache artifacts/icd11_cache.json --classifier <adapter dir> --db audit.sqlite --config config/config.yaml` |
| Baselines | `src.baselines.rule_based`, `src.baselines.structured_only`, `src.baselines.zero_shot_llm`, `src.detection.evaluate_baseline` |
| Held-out comparison, cost, Configuration C | `src.evaluation.heldout`, `src.evaluation.cost`, `src.evaluation.hybrid` |
| Narrative and SHAP checks | `src.evaluation.faithfulness`, `src.evaluation.rubric`, `src.evaluation.shap_coherence` |
| Feedback retraining (FR-12 to FR-14, GPU) | `python -m src.feedback.run_retraining --classifier-data data/classifier/v0.6-20k` |

**The test split has already been read** (Session B, #33): anything that scores it needs
`--final` and must not be rerun. Do not rerun `run_retraining --simulate-auditor`.
`src.pipeline --db` writes a SQLite file, which the web application does not read
(`import_sqlite` carried the Streamlit-era review database over once and refuses to run
again).

## Testing

```bash
.venv/bin/python -m pytest -q     # ML and web suites
```

`pyproject.toml` sets the pytest paths and the Django settings, so this runs from the repo
root. The settings load `.env`, so if `.env` sets `DATABASE_URL` to a PostgreSQL that is not
running, the web tests error; override it for a run:
`DATABASE_URL=sqlite:////tmp/cg_test.sqlite3 .venv/bin/python -m pytest -q` (312 passed,
25 Nov 2026). CI runs the suite before every deploy (`.github/workflows/deploy.yml`).

Tests mock anything that would otherwise need a GPU, a real model download, or a live
external API (HF/OpenAI calls, the ICD-11 container, transformer weight loading) — they
check *this project's* logic, not the correctness of the libraries it depends on.
`tests/test_plan.md` tracks every functional/non-functional requirement (FR-XX/NFR-XX)
against its test case, expected result, and actual result + evidence — the same format
Chapter 5's Table 5.3 uses, kept live from the start rather than reconstructed later.

## Plans and status

`docs/sprint_plan.md` is the live checklist ("Remaining implementation" and Sprint 8).
`docs/completion_plan.md`, `docs/remaining_work_plan.md` and `docs/platform_plan.md` record
how the work was planned and the decisions behind it; where they disagree with
`docs/sprint_plan.md` or `tests/test_plan.md`, the latter two are current.

## Deployment

The platform runs on one Oracle Cloud VM (`VM.Standard.E5.Flex`, UK South) at
https://claimsguard.tech behind Cloudflare: Caddy, the web process (gunicorn), PostgreSQL,
Redis, the simulated claims platform and the ingest and CPU workers under Docker Compose
(`deploy/compose.yaml`). **`main` is the deployed branch:** a pull request from `dev` into
`main` runs the tests, and merging it deploys (`.github/workflows/deploy.yml`). Every step
is a script you run yourself; the runbook is `deploy/README.md`, and the tested setup is in
`docs/environment.md`. Operational evidence (failure scenarios, response times, restarts):
`deploy/scripts/08-ops-check.sh`, TC-053 and TC-054. `docs/hosting_guide.md` records the
free-hosting options considered earlier.

## Development process

Scrum for One (a single-developer Agile variant): weekly sprints tracked in the GitHub
project "ClaimGuard: Scrum for One", one issue per backlog item; sprint ends that produced
a working increment are tagged (`v0.4-data` to `v0.9-frozen`). `docs/ai_use_log.md` is the
running AI-assistance log the course requires.

## License

MIT — see `LICENSE`.
