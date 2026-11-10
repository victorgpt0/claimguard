# Implementation Environment

Evidence for Chapter 5 Table 5.1 (`research_writing.txt` lines 999–1016). Fill this in
with *actually tested* values the first time real training/inference runs — not guessed
specs. See `implementation_roadmap.md` Section 2a for the exact commands to capture this.

## Development environment (Colab)

Recorded from the notebook outputs and the `pip freeze` snapshots that each notebook saves
to `MyDrive/claimguard/artifacts/environment_logs/`.

| Item | Tested value | Evidence |
|---|---|---|
| Runtime | Google Colab with compute units (none left on 19 Oct) | notebooks 01-03 |
| GPU | Tesla T4, 15,360 MiB, driver 580.82.07 | `nvidia-smi` in notebook 01 (18 Oct) |
| Python | 3.13.15 | notebook 01 output (18 Oct) |
| Key packages | torch 2.10.0, transformers 4.57.6, peft 0.21.1, vllm 0.19.0, scikit-learn 1.6.1, shap 0.52.0, pandas 2.2.3, numpy 2.1.3 | `requirements_frozen_*.txt` |
| Throughput | Note generation (Qwen2.5 7B AWQ, vLLM): about 63 s per 500 notes, 20,000 in about 42 min. Classifier fine-tuning (Bio_ClinicalBERT + LoRA, 10,800 examples): about 42 min per seed | notebook 01 output; run folder timestamps in `experiments/classifier/v0.6-20k/` |

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

**Deployed 9–10 Nov 2026 at https://claimsguard.tech** (runbook: `deploy/README.md`). One
Oracle Cloud Always Free VM runs the whole platform; Colab runs GPU jobs on demand.

| Item | Configured value | Evidence |
|---|---|---|
| Host | Oracle Cloud Always Free, `VM.Standard.A1.Flex` (Ampere ARM), 2 OCPU, 12 GB, 100 GB boot volume, region UK South (`uk-london-1`), reserved public IP | `deploy/README.md` step 1 |
| OS and runtime | Ubuntu 24.04 (aarch64), Docker with Compose; 4 GB swap; keys-only SSH with fail2ban | `deploy/scripts/01-bootstrap-vm.sh` |
| Services (`deploy/compose.yaml`) | Caddy 2.10 (TLS, Cloudflare addresses only) · web (Django, gunicorn) · PostgreSQL 18 · Redis 7.4 (append-only) · sha-cdp (simulated claims platform, internal only) · worker-ingest (claims feed and beat) · worker-cpu (stage 1, SHAP, classifier) | compose file, memory limits per service |
| Edge | Cloudflare (proxied DNS, Full (strict) TLS with a Cloudflare Origin Certificate, Always Use HTTPS, TLS ≥ 1.2) | `deploy/README.md` step 5 |
| GPU jobs | `notebooks/05_gpu_worker.ipynb` on Colab (T4), reaching PostgreSQL :5432 and Redis :6380 over TLS with the deployment's own CA | `deploy/README.md` step 8 |
| Model and data | Classifier, datasets and ICD-11 cache read from Google Drive, mounted at `/srv/gdrive` by rclone; the frozen classifier is also published on the Hugging Face Hub (`victorgpt0/claimguard-consistency-classifier`, tag `v1.0-evaluated`, TC-052) | `deploy/scripts/02-drive-mount.sh` |
| Releases | `main` is deployed: a merged pull request runs the tests, then deploys over SSH with a forced-command key | `.github/workflows/deploy.yml`, `deploy/scripts/deploy.sh` |
| Backups | Both databases nightly at 02:30 (Nairobi), 14 days on the VM and 30 on Drive | `deploy/scripts/06-backups.sh` |

**Measured on the deployed release:** OS, CPU, memory, disk and image versions, response
times, failure rates, restarts and dependency failures are recorded by
`deploy/scripts/08-ops-check.sh` (TC-053, TC-054). *(Values to be filled in from its report.)*

Constraints to state in Chapter 5: the Always Free shape limit (2 OCPU / 12 GB in total),
Oracle's idle-reclamation rule for Always Free VMs (CPU, network and memory under 20% for 7
days), and that GPU work depends on a Colab session being open. The synthetic data and the
absence of real users limit what the operational figures say about production load.

**Local machine:** reads Drive directly. The `rclone-gdrive` systemd user service mounts
`MyDrive` at `~/gdrive` (rclone 1.75.2, VFS cache), and the local `config/config.yaml`
points `data_dir`, `model_dir`, `artifacts_dir`, `experiments_dir` and `db_path` at
`~/gdrive/claimguard/...`. The remote uses rclone's shared Google client ID, which rclone
says is being retired during 2026; create an own client ID
(https://rclone.org/drive/#making-your-own-client-id) if the mount stops authenticating.

Measured CPU speed on the local machine (19 Oct): classifier fine-tuning about 0.58 s
per example (one epoch of 10,860 about 1.8 h), inference about 0.09 s per example; the
tuned pipeline without narratives audited 900 claims in 98 s (TC-031).

## External services

| Service | Role | Notes |
|---|---|---|
| WHO ICD-11 API | Code definition/inclusion/exclusion lookup (IR-01) | **Self-hosted**, not WHO's cloud API — `whoicd/icd-api` Docker image, run via `docker-compose.yml` at repo root. No OAuth2 credentials needed for local deployment (that's cloud-API-only). `acceptLicense=true` and `saveAnalytics` are container startup env vars, not per-request config. |
| Hugging Face / OpenAI | LLM narrative generation (IR-02) | Provider choice: see `config/config.yaml` `llm_provider` |
| Google Drive | Data/model/artifact persistence (DR-05) | See `implementation_roadmap.md` Section 3 for folder layout |
| PostgreSQL 18 and Redis 7.4 on the VM | Operational database (IR-04) and job queue (IR-08) | Moved from Neon and Upstash on 9 Nov (`deploy/scripts/05-migrate-from-neon.sh`): Neon's US East region answered in about 0.3 s per query from Kenya. Reached by the Colab worker over TLS only. |
| Cloudflare | DNS, TLS at the edge, the only path to the VM's port 443 | Free plan |
| Hugging Face Hub | Public release of the frozen classifier | TC-052 |
| Google (OpenID Connect) | Sign-in for linked accounts (IR-07) | OAuth client in Google Cloud Console; redirect URIs for each address the app is opened at. |

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
