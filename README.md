# ClaimGuard

LLM-Assisted Anomaly Detection for Post-Payment Fraud Audit: A Case Study of Kenya's
Social Health Authority

Full research documentation (Chapters 1–4, methodology, system design) lives in the
project's separate documentation repository, not here — this repo is the implementation
only. See that repo's `SHA_Fraud_Audit_Project_Documentation.docx` for the complete
design rationale this code follows.

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
└── db/                    # SQLite schema the pipeline writes reports to (imported into Django)
web/                         # Django application (IR-03, IR-04, IR-07, IR-08)
├── claimguard_web/          # settings (env-driven), URLs, Celery app
├── accounts/                # User with role, allauth adapters (Google, no sign-up), admin site
├── activity/                # ActivityLog (FR-18)
└── audit/                   # Batch, Claim, ClassifierVerdict, AuditReport, Job; pages; upload
                             # validation (FR-15); process_batch task; batch summary (FR-16)
notebooks/05_gpu_worker.ipynb  # Colab GPU worker for the "gpu" queue
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
   docker compose up -d icd11-api
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
| Hosted, needed for the Colab GPU worker | Neon: `neon deploy` writes `DATABASE_URL` (pooled) and `DATABASE_URL_UNPOOLED` into `.env` | Upstash: `rediss://default:<password>@<db>.upstash.io:6379` |

The settings verify Upstash's TLS certificate and turn off server-side cursors on Neon's
pooled endpoint. Run migrations and the one-off import on Neon's direct endpoint:

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

**Roles.** The Fraud Auditor sees Overview, Review queue and Batches (upload, processing,
summaries). The Administrator sees Overview, Batches (summaries only), Model performance,
Feedback loop, and Users and Activity log in Django's admin site, where accounts are
created, deactivated (never deleted), given a role and linked to a Google email, and the
log is filtered and exported. Everyone changes their own password from their name in the
top bar. A session idle for 30 minutes (`SESSION_TIMEOUT_MINUTES`) is signed out.

**Google sign-in** needs `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET` in `.env` and, on the
OAuth client in Google Cloud Console, every address the app is opened at as an authorised
redirect URI, e.g. `http://localhost:8000/accounts/google/login/callback/` and the same with
`127.0.0.1` (Google treats them as different; a missing one gives `redirect_uri_mismatch`).
Only accounts whose email the Administrator has linked can sign in with Google.

**Batches and jobs.** The Fraud Auditor uploads a claims CSV under Batches (a template is
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

Tests: `.venv/bin/python -m pytest` runs the ML and web suites together (SQLite by default;
set `DATABASE_URL` to run the web tests on PostgreSQL). Design diagrams for Chapter 4 are in
`docs/diagrams/`.

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
root. Set `DATABASE_URL` to run the web tests on PostgreSQL instead of SQLite.

Tests mock anything that would otherwise need a GPU, a real model download, or a live
external API (HF/OpenAI calls, the ICD-11 container, transformer weight loading) — they
check *this project's* logic, not the correctness of the libraries it depends on.
`tests/test_plan.md` tracks every functional/non-functional requirement (FR-XX/NFR-XX)
against its test case, expected result, and actual result + evidence — the same format
Chapter 5's Table 5.3 uses, kept live from the start rather than reconstructed later.

## Completion plan

`docs/completion_plan.md` is the single reference for what's actually left before this
implementation is done: what's trained and where (Colab vs. local), where every
artifact is stored (Drive during development, Hugging Face Hub as the deployment
hand-off), the deployed dashboard's own storage/persistence decision, an end-to-end
data-flow diagram, and the remaining Sprint-by-Sprint checklist. Current status is in
`docs/remaining_work_plan.md` and `docs/platform_plan.md`.

## Deployment

The Django application uses hosted Neon PostgreSQL and Upstash Redis (see Web application);
only the web process itself still needs a host (#32, #39). `docker compose up -d --build db
redis web` runs it in a container (gunicorn). The hosting guide below was written for the
Streamlit dashboard; its options still apply to the web process.

Free hosting on your own domain: see `docs/hosting_guide.md` for two options (Cloudflare
Tunnel for a self-hosted demo, or Hugging Face Spaces + a Cloudflare Worker reverse proxy
for something always-on) — both checked against current provider docs before writing.
See `docs/completion_plan.md` §3c for a persistence caveat specific to the free-tier
Spaces option that affects the feedback loop (FR-12/13/14).

## Development process

Scrum for One (single-developer Agile variant, Section 3.3): 2-week sprints tracked via
GitHub Projects/Issues mapped 1:1 to the Product Backlog. See `implementation_roadmap.md`
in the documentation repo for full environment setup (Colab, Google Drive folder layout,
experiment logging) and `docs/ai_use_log.md` for the running AI-assistance disclosure log
required by the course guide.

## License

MIT — see `LICENSE`.
