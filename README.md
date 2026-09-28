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
multilingual transformer classifier (transfer learning, not a prompted LLM) predicts
whether the flagged claim's clinical justification is consistent with its billed code; an
LLM, invoked only for claims the classifier flags inconsistent, converts the verdict into
a plain-language audit narrative grounded in the WHO ICD-11 API.

## Architecture

Each module below maps directly to a class in the system design (Chapter 4, Class
Diagram), so the OOAD-to-code mapping stays visible as the project develops:

```
src/
├── data/            # ClaimsPreprocessor + noise/fraud/code-switching injection (Table 3.1)
├── detection/        # IsolationForestDetector, AutoencoderDetector
├── explainability/    # SHAPExplainer
├── classification/    # SemanticConsistencyClassifier + LoRA/PEFT fine-tuning
├── icd11/             # ICD11APIClient
├── narrative/          # AuditNarrativeGenerator (LLM, explanation-only)
├── reporting/          # AuditReportGenerator, AuditReportRepository
├── feedback/            # FeedbackRepository (FR-12), ClassifierRetrainer (FR-13/FR-14)
├── auth/                # User entity, authenticate() (NFR-03)
└── db/                  # SQLite schema (Figure 4.7) + seed script
dashboard/app.py          # Streamlit entrypoint (IR-03)
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
   # fill in .env: HF_TOKEN / OPENAI_API_KEY / ICD11_CLIENT_ID / ICD11_CLIENT_SECRET
   ```
   `config.yaml` and `.env` are gitignored — never commit them. On Colab, prefer
   `google.colab.userdata` for secrets instead of `.env` (see `implementation_roadmap.md`).

3. **Initialise the database:**
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

Not yet functional end-to-end — see `tests/test_plan.md` and the Product Backlog
(documentation repo, Section 3.4.1) for build order. Health-check/demo procedure will be
documented here once the pipeline has something to demonstrate.

## Testing

```bash
pytest tests/
```

`tests/test_plan.md` tracks every functional/non-functional requirement (FR-XX/NFR-XX)
against its test case, expected result, and actual result + evidence — the same format
Chapter 5's Table 5.3 uses, kept live from the start rather than reconstructed later.

## Development process

Scrum for One (single-developer Agile variant, Section 3.3): 2-week sprints tracked via
GitHub Projects/Issues mapped 1:1 to the Product Backlog. See `implementation_roadmap.md`
in the documentation repo for full environment setup (Colab, Google Drive folder layout,
experiment logging) and `docs/ai_use_log.md` for the running AI-assistance disclosure log
required by the course guide.

## License

MIT — see `LICENSE`.
