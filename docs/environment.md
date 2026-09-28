# Implementation Environment

Evidence for Chapter 5 Table 5.1 (`research_writing.txt` lines 999–1016). Fill this in
with *actually tested* values the first time real training/inference runs — not guessed
specs. See `implementation_roadmap.md` Section 2a for the exact commands to capture this.

## Development environment (Colab)

| Item | Tested value | Evidence |
|---|---|---|
| Runtime | *(not yet run)* | `!nvidia-smi` output, first classifier training run |
| GPU | *(not yet run)* | `!nvidia-smi` |
| Python version | *(not yet run)* | `!python --version` |
| Key package versions | see `requirements.txt` (starting pins, not yet tested) | `pip freeze` snapshot, saved to `artifacts/environment_logs/` |

## Deployment / demonstration environment

*(Not yet decided.)* Distinguish this explicitly from the Colab dev environment once
decided — `research_writing.txt` line 994 requires it, and warns not to claim public
availability unless actually deployed for public access. Likely candidates: local run via
`streamlit run dashboard/app.py`, or a small private VM — record whichever is actually
used, with its own tested minimum spec.

## External services

| Service | Role | Notes |
|---|---|---|
| WHO ICD-11 API | Code definition/inclusion/exclusion lookup (IR-01) | Client ID/secret via env vars, never committed |
| Hugging Face / OpenAI | LLM narrative generation (IR-02) | Provider choice: see `config/config.yaml` `llm_provider` |
| Google Drive | Data/model/artifact persistence (DR-05) | See `implementation_roadmap.md` Section 3 for folder layout |
