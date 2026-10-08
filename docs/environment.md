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

## Local machine (Docker host, dashboard host)

Measured 2026-09-28, not guessed — see `implementation_roadmap.md` Section 2a for why
this matters (Table 5.1 wants tested specs, not vendor marketing numbers):

| Item | Tested value | Evidence |
|---|---|---|
| CPU | AMD Ryzen 5 5625U (6 cores / 12 threads) | `lscpu` |
| GPU | AMD Radeon integrated (Barcelo) — no discrete GPU, no CUDA/ROCm ML support | `lspci \| grep -i vga`; `nvidia-smi` absent |
| RAM | 14GB total; **~11GB already in use at time of check, ~3.1GB available** | `free -h` |
| Disk | 460GB total, 92% full, 37GB free | `df -h` |
| Docker | 29.7.2, Compose v5.4.0, working | `docker --version`, `docker info` |

**Implication for the WHO ICD-11 container**: `whoicd/icd-api` runs an internal
Elasticsearch instance that WHO's own reference config gives a 2GB JVM heap. Against
~3.1GB typically available on this machine, that's tight — expect noticeable slowdown
(this system's swappiness is 180, so it swaps hard rather than OOM-killing outright) if
run alongside a browser/IDE. `docker-compose.yml` sets `mem_limit: 2560m` to bound the
worst case, and `restart: "no"` so it isn't silently running between sessions. Given
this, the caching approach in the note below (pre-fetch definitions once, mostly don't
run the container) is the recommended path for this machine specifically, not just the
lowest-effort one.

**Implication for classifier fine-tuning**: no discrete GPU locally rules out local
fine-tuning as a serious option regardless of the ICD-11 container question — Colab's
free-tier GPU remains necessary (consistent with Section 1.7's compute-constraint
reasoning for choosing LoRA/PEFT in the first place).

## Deployment / demonstration environment

*(Not yet decided.)* Distinguish this explicitly from the Colab dev environment once
decided — `research_writing.txt` line 994 requires it, and warns not to claim public
availability unless actually deployed for public access. Likely candidates: local run via
`streamlit run dashboard/app.py`, or a small private VM — record whichever is actually
used, with its own tested minimum spec.

## External services

| Service | Role | Notes |
|---|---|---|
| WHO ICD-11 API | Code definition/inclusion/exclusion lookup (IR-01) | **Self-hosted**, not WHO's cloud API — `whoicd/icd-api` Docker image, run via `docker-compose.yml` at repo root. No OAuth2 credentials needed for local deployment (that's cloud-API-only). `acceptLicense=true` and `saveAnalytics` are container startup env vars, not per-request config. |
| Hugging Face / OpenAI | LLM narrative generation (IR-02) | Provider choice: see `config/config.yaml` `llm_provider` |
| Google Drive | Data/model/artifact persistence (DR-05) | See `implementation_roadmap.md` Section 3 for folder layout |

### Note: the ICD-11 container and Colab don't share a network

Because the ICD-11 API is a **local** Docker container, Colab notebooks (running on
Google's infrastructure) can't reach it directly — Colab has no route to `localhost` on
your own machine, and Colab doesn't give you a Docker daemon to run the container inside
the notebook's own VM either. This needs a decision before Sprint 3 (WHO ICD-11 API
integration): either (a) run classifier fine-tuning locally instead of on Colab, (b)
tunnel the local container to a reachable URL (e.g. `ngrok`/Cloudflare Tunnel) for the
duration of each session, or (c) since ICD-11 code definitions are static reference data,
pre-fetch and cache the definitions your synthetic dataset actually uses into a local
JSON/CSV file once, upload that to Drive, and have both training (Colab) and the live
dashboard read from the cache first, falling back to a live container call only when a
code is missing. (c) is the lowest-effort option, avoids re-solving this every session,
and — given this machine's tight RAM headroom above — also means the heavy container
only needs to run occasionally to build/refresh the cache rather than stay up during
every training or dashboard session. Still flagged as your decision, not decided
unilaterally, but recommended given the measured constraints.
