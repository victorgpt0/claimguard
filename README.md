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
├── auth/                  # User entity, authenticate() (NFR-03, PBKDF2-hashed passwords)
└── db/                    # Streamlit-era SQLite schema + seed script (imported into Django)
web/                         # Django application (IR-03, IR-04, IR-07, IR-08)
├── claimguard_web/          # settings (env-driven), URLs, Celery app
├── accounts/                # User with role, allauth adapters (Google, no sign-up), admin site
├── activity/                # ActivityLog (FR-18)
└── audit/                   # Batch, Claim, ClassifierVerdict, AuditReport, Job; pages; upload
                             # validation (FR-15); process_batch task; batch summary (FR-16)
dashboard/app.py             # Streamlit dashboard, kept until the django branch is merged
notebooks/05_gpu_worker.ipynb  # Colab GPU worker for the "gpu" queue
docker-compose.yml           # PostgreSQL, Redis, the web service, WHO ICD-11 API
docs/diagrams/               # Chapter 4 diagrams for the Django platform (PlantUML + PNG)
```

## Setup

1. **Clone and create a virtual environment:**
   ```bash
   git clone git@github.com:victorgpt0/claimguard.git
   cd claimguard
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure:**
   ```bash
   cp config/config.example.yaml config/config.yaml
   cp .env.example .env
   # fill in .env: HF_TOKEN / OPENAI_API_KEY (no ICD-11 credentials needed — see step 3)
   ```

3. **Start the WHO ICD-11 API (self-hosted, not the cloud API):**
   ```bash
   docker compose up -d
   curl http://localhost:8081/icd/release/11/2026-01/mms  # sanity check it's up
   ```
   No OAuth2 credentials needed for a local deployment. See `docs/environment.md` for the
   Colab-reachability caveat (Colab can't reach this container directly) before Sprint 3.
   `config.yaml` and `.env` are gitignored — never commit them. On Colab, prefer
   `google.colab.userdata` for secrets instead of `.env` (see `implementation_roadmap.md`).

4. **Initialise the database:**
   ```bash
   python -m src.db.seed --db-path /path/to/sha_fraud.sqlite [--auditor auditor]
   ```
   Creates the schema (Figure 4.7) and the first Administrator account, `admin` by default
   (`--admin` changes the name); the password is asked for and must be at least 10
   characters. `--auditor` also creates one Auditor account. Re-running leaves existing
   accounts unchanged, so it never resets a password.

## Web application (Django, branch `django`)

The Django application in `web/` replaces the Streamlit dashboard (`docs/platform_plan.md`):
accounts with the two roles and Google sign-in for linked accounts, batch upload with full
validation, processing as background jobs on a local CPU worker or a Colab GPU worker, the
review queue and report pages, a summary report per batch, user management and an activity
log. The Streamlit dashboard in `dashboard/` (see Running) is kept until the branch is merged.

Dependencies are locked in `uv.lock` (`pyproject.toml` groups: `web`, `ml`, `streamlit`,
`dev`; torch comes from the CPU-only index locally):

```bash
python3 -m venv .venv && .venv/bin/pip install uv
.venv/bin/uv sync --frozen        # everything; or: --no-default-groups --group web --group dev
cp .env.example .env              # without DATABASE_URL it runs on SQLite (web/db.sqlite3)
cp config/config.example.yaml config/config.yaml   # paths to the Drive folders; keep pipeline.dataset
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

## Running

The pipeline runs on Colab (`notebooks/03_run_pipeline.ipynb`) and writes its audit
reports to `MyDrive/claimguard/db/sha_fraud.sqlite`. The dashboard runs on a laptop
against a downloaded copy of that database:

1. Download `MyDrive/claimguard/db/sha_fraud.sqlite` to `db/sha_fraud.sqlite`, and
   `MyDrive/claimguard/artifacts/icd11_cache.json` to `artifacts/` (the review screen
   shows each billed code's ICD-11 reference text from it).
2. Create `config/config.yaml` from `config/config.example.yaml` with repo-relative paths
   (`db_path: db/sha_fraud.sqlite`, `artifacts_dir: artifacts`, `model_dir: models`).
   Relative paths resolve against the repo root.
3. Optionally, for the latest run's funnel chart, download the newest log in
   `MyDrive/claimguard/experiments/pipeline_runs/` to `experiments/pipeline_runs/`.
4. From the repo root:

```bash
venv/bin/python -m streamlit run dashboard/app.py
```

Use `python -m streamlit`, not the bare `streamlit` command: it puts the repo root on the
import path, so `src` can be imported. Decisions are written to the local copy, so upload
it back to Drive to keep them. Approve and reject decisions also append labelled
examples to `artifacts/feedback_dataset/examples.jsonl` (FR-12). For the Model performance
tab, copy the active classifier run folder to `models/classifier_lora/active/`.

The dashboard opens on a sign-in form (NFR-03); create the accounts first with
`src.db.seed` (step 4 above) against the same database. Each role sees its own tabs (IR-03):
the Fraud Auditor sees Overview, Review queue and Account, and every decision is stored with
the auditor's identity (FR-09); the Administrator sees Overview, Model performance, Feedback
loop, Users, Activity log and Account. In Users the Administrator creates accounts,
deactivates or reactivates them (never deletes, so decisions stay attributable; the last
active Administrator is protected), resets passwords and links Google accounts. Everyone can
change their own password under Account. Sign-ins (including failed ones), sign-outs,
decisions and account changes are recorded in the activity log, which the Administrator can
filter and download as CSV; it never holds a password or note text. A session idle for `auth.session_timeout_minutes` (default 30) is signed out.

**Sign in with Google (optional, #61).** Copy `.streamlit/secrets.example.toml` to
`.streamlit/secrets.toml` (gitignored) and fill in an OAuth client from Google Cloud Console
(APIs & Services, Credentials, OAuth client ID, type Web application, authorised redirect
URI `http://localhost:8501/oauth2callback`). Google only proves who someone is: a Google
account gets in only after it is linked to an existing user, and then with that user's role.
```bash
python -m src.db.seed --db-path <db> --link-email auditor someone@gmail.com
```
(or, once signed in as the Administrator, in the Users tab).
Without `secrets.toml` the button is hidden and password sign-in works as before. Passwords
are stored only as salted PBKDF2-SHA256 hashes.

**Before Sprint 3** (ICD-11 integration), refresh the local ICD-11 cache once against the
running container rather than requiring live access on every claim:
```bash
docker compose up -d
python -m src.icd11.refresh_cache --codes-file data/processed/icd11_codes_used.txt \
    --cache-path artifacts/icd11_cache.json
docker compose down
```

## Testing

```bash
venv/bin/python -m pytest -q
```

Use `python -m pytest`, not bare `pytest`: there is no package config, and `python -m` is
what puts the repo root on `sys.path`.

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
data-flow diagram, and the remaining Sprint-by-Sprint checklist. Read this before
starting Sprint 1.

## Deployment

The Django application uses hosted Neon PostgreSQL and Upstash Redis (see Web application);
only the web process itself still needs a host (#32, #39). The older notes below were
written for the Streamlit dashboard.

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
