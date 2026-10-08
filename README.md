# ClaimGuard

LLM-Assisted Anomaly Detection for Post-Payment Fraud Audit: A Case Study of Kenya's
Social Health Authority — final-year project implementation (ICS Project II, Strathmore
University). Gichuru Victor Kimotho (167026), ICS 4B — Supervisor: Mr. Tiberius Tabulu.

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
├── data/              # ClaimsPreprocessor + fraud/text-noise injection (Table 3.1)
├── detection/         # IsolationForestDetector, AutoencoderDetector
├── explainability/     # SHAPExplainer
├── classification/     # SemanticConsistencyClassifier + LoRA/PEFT fine-tuning
├── icd11/               # ICD11APIClient + ICD11Cache (see docs/environment.md) + refresh_cache CLI
├── narrative/            # AuditNarrativeGenerator (LLM, explanation-only)
├── reporting/            # AuditReportGenerator, AuditReportRepository
├── feedback/              # FeedbackRepository (FR-12), ClassifierRetrainer (FR-13/FR-14)
├── auth/                  # User entity, authenticate() (NFR-03, PBKDF2-hashed passwords)
└── db/                    # SQLite schema (Figure 4.7) + seed script
dashboard/app.py            # Streamlit entrypoint (IR-03)
docker-compose.yml           # WHO ICD-11 API, self-hosted
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
   python -m src.db.seed --db-path /path/to/sha_fraud.sqlite
   ```
   Seeds the schema (Figure 4.7) plus the two user accounts (Auditor, Administrator) — the
   only two that will ever exist; in-app account management is out of scope (FR-11 scope
   note).

## Running

```bash
streamlit run dashboard/app.py
```

Every module has a real implementation (not just stubs) — data preprocessing, both
anomaly detectors, SHAP, the classifier/training loop, ICD-11 client + cache, the LLM
narrative layer, the SQLite repository, feedback capture, and the retraining
promotion gate. What's still missing before this is genuinely useful end-to-end: an
actual synthetic dataset (Sprint 1) and a trained classifier checkpoint (Sprint 4+) —
until then, `admin_view()` will correctly report "no trained checkpoint found" rather
than fail. See `tests/test_plan.md` and the Product Backlog (documentation repo, Section
3.4.1) for build order.

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
pytest tests/
```

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
