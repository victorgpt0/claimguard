# Traceability record

Drafted 2 Nov 2026 for Course Sprint III ("objective or requirement → component → test or
evaluation → result → evidence", `docs/sprint_plan.md`). It is the source for Chapter 5's
objective-by-objective discussion (Section 5.6) and for the requirements status in Table 5.5.
It links the objectives (Chapter 1, Section 1.3.2) and requirements (Chapter 4, Table 4.1,
plus FR-15 to FR-18, IR-07 and IR-08 added on 25 Oct in `docs/system_features_plan.md`) to the
code, the test records in `tests/test_plan.md` and the result files. Every number below is
copied from those records. Unmet or partly met requirements are listed with the reason and
the issue that tracks them. Do not tick a row until its evidence exists.

Where the evidence is:
- **Repo**: code, `tests/test_plan.md`, `tests/test_*.py`, `web/*/tests/`, `compare/chapter5.txt` (the Chapter 5 draft, outside this repo at `../compare/`).
- **Drive** (`MyDrive/claimguard/`, mounted at `~/gdrive/claimguard/`): `experiments/` (results), `models/classifier_lora/evaluated_v1/` (the frozen classifier), `db/` (result databases), `artifacts/` (feedback dataset, ICD-11 cache, environment logs).

## Table 1: Objectives to evidence

| Objective | Requirements | Components | Test or evaluation | Result | Evidence | Status |
|---|---|---|---|---|---|---|
| i. Analyse fraud in national health insurance (SHA/NHIF typologies) | DR-01, DR-02, DR-03 (typologies injected) | Chapter 1 and Section 2.2.3; Table 3.1; `src/data/` injection rules | TC-016 to TC-020 (each typology injected as specified; tariffs match the gazette) | Six typologies injected at the planned rate (333 per typology in v0.6-20k); 14/14 surgical tariffs match the gazette scan | `tests/test_plan.md` TC-016–TC-020; `docs/data_generation.md`; tag `v0.6-data` | Met (literature objective; evidence is the documentation plus the traced injection rules) |
| ii. Analyse ML and language models for fraud detection and verification | — (literature) | Chapter 2; model choices of Section 3.2.3 | Choices tested empirically in TC-045/TC-046 (structured, rules, classifier, zero-shot LLM compared) | See objective v | Chapter 2; `compare/chapter5.txt` Tables 5.9, 5.10 | Met (literature objective) |
| iii. Review gaps in unstructured-text analysis and black-box explainability | — (literature) | Chapter 2, Section 2.4 | The gap is tested by Table 5.10: structured methods vs a learned text classifier on the text label | Classifier lane F1 0.890 vs zero-shot LLM 0.578 on the text task | `compare/chapter5.txt` Table 5.10; TC-046 | Met (literature objective) |
| iv-a. Unsupervised anomaly detection stage | FR-01, FR-02, FR-10, FR-15, NFR-01 | `src/detection/` (Isolation Forest, autoencoder, text screen, `tune_stage1`), `src/data/` ingestion, `web/audit/` upload validation | TC-001, TC-002, TC-010, TC-022, TC-030, TC-032, TC-039 | Stage-1 recall 0.81 ± 0.02 over 5 seeds on v0.6 validation (contamination 0.15, AE quantile 0.975, screen 0.005553). NFR-01: 1,000 claims in 1 min 59.7 s, 0.12 s per claim against a 5 s target | `experiments/stage1_tuning/v0.6-20k_20261019T084100Z/` (repo); `config/config.example.yaml`; TC-002 | Met |
| iv-b. Explainable attribution (SHAP) | FR-03, NFR-04 | `src/explainability/` | TC-003; SHAP coherence check of 100 flagged claims (#38) | Every flagged claim has a 3-feature attribution (TC-003). Coherence sheet marked 1 Nov: 75 of 100 coherent; ghost billing 4 of 15 | TC-003; Drive `experiments/shap_coherence/sheet.csv`, `sheet_summary.json` | Partly met: the coherence result is not yet in `tests/test_plan.md` or Chapter 5, and #38 is still open |
| iv-c. Fine-tuned semantic consistency classifier | FR-04, NFR-05 (recall ≥ 0.80), IR-01, DR-03 | `src/classification/` (Bio_ClinicalBERT + LoRA), `src/icd11/` client and cache | TC-004, TC-006, TC-024, TC-026, TC-027, TC-033 | Validation over 5 seeds: precision 0.960 ± 0.009, recall 0.990 ± 0.006, F1 0.975 ± 0.004; seed 7 frozen | Drive `models/classifier_lora/evaluated_v1/metrics.json` (F1 0.981, recall 0.989); `experiments/classifier/v0.6-20k/` | Met. Model release with a model card (brief Example 3) not yet done |
| iv-d. Consistent verdicts filtered out; LLM explains only | FR-05, FR-06, IR-02, NFR-02 | `src/pipeline.py`, `src/narrative/narrative_generator.py` (refuses any verdict except inconsistent) | TC-005, TC-006, TC-025, TC-028, TC-034 | Deployment pool run: 354 reports, 82 narratives, exactly on the text-inconsistent reports, 0 failures; failing LLM retried 3 times then logged | Drive `experiments/pipeline_runs/v0.6-20k_deployment_pool_20261018T202852Z.json`; `db/sha_fraud.sqlite` | Met |
| iv-e. Audit report stored and reviewed | FR-07, FR-08, FR-09, FR-16, NFR-03, NFR-07, IR-03, IR-04, IR-07, IR-08 | `src/reporting/`, `web/audit/`, `web/accounts/`, Celery jobs | TC-007, TC-038, TC-040, TC-041, TC-043, TC-044; TC-015 (usability) | Reports stored with SHAP, verdict, narrative and metadata; review with reviewer identity; CPU job queue passes | `web/audit/tests/`, `web/accounts/tests/` (60 web tests); TC-038–TC-044 | Partly met: TC-015 not run; the GPU queue (TC-040) not yet run end to end; not yet deployed over HTTPS (#39) |
| iv-f. Feedback loop | FR-12, FR-13, FR-14, IR-06, DR-08 | `src/feedback/` (repository, `run_retraining`), `web/audit/` feedback page | TC-012, TC-013, TC-014, TC-039 (held-out claims refused at upload) | 59 feedback examples (all simulated); retraining ran 25 Oct; candidate F1 0.962 / recall 0.972 vs active 0.981 / 0.983, so not promoted | Drive `experiments/retraining_history.jsonl`; `artifacts/feedback_dataset/examples.jsonl`; `models/classifier_lora/candidates/20261025T194428Z/` | Met, with the stated caveat that all decisions were simulated from ground truth |
| v-a. Validate against baselines with statistical metrics | NFR-05, Section 3.2.4 | `src/evaluation/`, `src/baselines/` (rules, Isolation Forest, structured-only, zero-shot LLM) | TC-036, TC-037, TC-045, TC-046 | Held-out test, `is_fraud`: A F1 0.561, C 0.826, rules 0.578, IF 0.462, structured-only 0.588 ± 0.021, zero-shot in classifier's place 0.476. Text label: text lane 0.890 vs zero-shot 0.578. McNemar p < 0.001 for every comparison against A | Drive `experiments/evaluation/v0.6-20k_test_is_fraud_20261027T215553Z/`, `.../v0.6-20k_test_text_20261027T215556Z/` (`results.json`, `tables.md`) | Met. Pipeline as designed (A) does not beat the rules on F1; C does. Structured-only used 3 seeds, not the planned 5 |
| v-b. Robustness and fairness | NFR-05 (noise gap; profiling) | as v-a | TC-004 (profiling), TC-027, TC-046 | Text lane F1 clean 0.891 vs noisy 0.886; classifier FPR 0–0.26% at every facility level. Full pipeline FPR rises with facility level (L3 3.9% to L6 50%) | TC-004, TC-046; Table 5.12 | Partly met: the classifier passes; the full pipeline fails the profiling check (structural lane), recorded not corrected. The planned bootstrap significance test of the noise gap is not recorded in TC-046 |
| v-c. Qualitative review of audit notes | NFR-04, Section 3.2.4 | `src/evaluation/faithfulness.py`, rubric sheet generator | TC-047; rubric scoring of 50 reports (#37) | Faithfulness: 141 of 146 test-split narratives pass (96.6%); 3 of 11 failures genuinely unfaithful. Rubric sheet marked 1 Nov: means 4.1 clinical accuracy, 4.46 inconsistency identification, 4.48 clarity, 2.0 actionability, 4.74 no hallucination | Drive `experiments/faithfulness/`; `experiments/rubric/sheet.csv`, `sheet_summary.json` | Partly met: rubric result not yet recorded in `tests/test_plan.md` or Chapter 5, and #37 is still open; single rater, so no Krippendorff's alpha |
| General objective: a deployed, usable system | NFR-07, #39 | Docker Compose, Django, Celery, Neon and Upstash | Deployment over HTTPS; failure scenarios on the deployed system; TC-015 | — | `docs/hosting_guide.md` | Not met yet: hosting not chosen (#32), not deployed (#39), usability test not run |

## Table 2: Requirements catalogue status

Status: **Met** (test passed and recorded), **Partly** (met with a stated gap), **Not met**, **Changed** (wording changed; see Chapter 5 Table 5.4).

| ID | Requirement (short) | Tests | Status | Note |
|---|---|---|---|---|
| FR-01 | Ingest a claims batch | TC-001, TC-034, TC-039 | Met | |
| FR-02 | Anomaly score and flag | TC-002, TC-022, TC-030, TC-032 | Met | Stage 1 extended with a text screen (#50) |
| FR-03 | SHAP attributions | TC-003, #38 | Partly | Coherence result (75/100) not yet recorded |
| FR-04 | Semantic consistency classifier | TC-004, TC-026, TC-033 | Met | |
| FR-05 | Consistent verdicts logged, not queued | TC-005 | Changed | Reworded 17 Oct (#46): two review lanes |
| FR-06 | LLM narrative for inconsistent claims | TC-006, TC-034 | Met | |
| FR-07 | Audit report assembled and stored | TC-007, TC-034 | Met | |
| FR-08 | Auditor review queue | TC-038 | Met | Re-run on Django (TC-008 superseded) |
| FR-09 | Approve / reject / escalate with identity | TC-038 | Met | 80 imported decisions predate sign-in and have no reviewer |
| FR-10 | Thresholds from configuration | TC-010 | Met | |
| FR-11 | Metrics and retraining history for the Administrator | TC-011 | Partly | Model page shows validation metrics, not the held-out test metrics that now exist (TC-045); history table works |
| FR-12 | Decisions become labelled feedback | TC-012 | Met | All 59 examples simulated |
| FR-13 | Retraining-due trigger | TC-013 | Met | |
| FR-14 | No promotion of a weaker candidate | TC-014 | Met | Candidate refused on real data, 25 Oct |
| FR-15 | Upload validation (added 25 Oct) | TC-039 | Met | |
| FR-16 | Batch summary report and downloads | TC-041 | Met | |
| FR-17 | User management | TC-042 | Met | |
| FR-18 | Activity log | TC-042 | Met | |
| NFR-01 | ≤ 5 s per claim, 1,000 claims | TC-002 | Met | 0.12 s per claim on the workstation CPU |
| NFR-02 | Retry then log on API failure | TC-006, TC-025 | Met | |
| NFR-03 | Authenticated access only | TC-038, TC-042, TC-044 | Met | Re-check on the deployed system pending (#39) |
| NFR-04 | SHAP and narrative in every report | TC-003, TC-047 | Partly | Narratives exist only on text-inconsistent reports by design (FR-05 change); structural-lane reports carry SHAP only |
| NFR-05 | Recall ≥ 0.80, no profiling, noise gap, no regression | TC-004, TC-014, TC-033, TC-046 | Partly | Classifier passes; full pipeline fails the profiling clause (TC-004); data has no regions, facility level used |
| NFR-06 | Classes unit-testable | full suite (259 tests, GitHub Actions) | Met | `.github/workflows/tests.yml`; first green run on GitHub to confirm after push |
| NFR-07 | Single-view review; usability | TC-015 | Not met | Usability test not yet run (researcher, with one participant) |
| DR-01 | Sources | TC-016, `docs/data_generation.md` | Changed | Chapter 4 names DE-SynPUF and Synthea; the data was instead generated from SHA's published tariffs with LLM-written notes (`docs/data_generation.md`). Chapter 4 wording to update |
| DR-02 | Synthetic labels | `docs/data_generation.md` | Met | Limitation stated |
| DR-03 | Stratified splits | TC-016, TC-019 | Met | |
| DR-04 | Quality rules and noise | TC-001, TC-021, TC-023 | Partly | TC-021 notes the full 7B note-generation check as still to do; 0.8% of v0.6 notes partly non-English (TC-033) |
| DR-05 | Artefacts separate from the operational database | TC-044 | Met | SQLite replaced by PostgreSQL (Table 5.4) |
| DR-06 | No real data | `docs/data_generation.md` | Met | |
| DR-07 | Provenance and seeds | git history, tags `v0.6-data` | Met | |
| DR-08 | Test set never becomes feedback | TC-019, TC-039 | Met | Held-out claim ids refused at upload |
| IR-01 | WHO ICD-11 API and cache | TC-006, TC-024 | Met | Self-hosted container, MMS 2026-01 |
| IR-02 | LLM for explanation only | TC-028, TC-034 | Met | Llama 3.1 8B on Colab (vLLM) instead of a hosted API |
| IR-03 | Role-based views | TC-038 | Changed | Django instead of Streamlit |
| IR-04 | Relational store | TC-044 | Changed | PostgreSQL instead of SQLite |
| IR-05 | Git and GitHub | commit history | Met | |
| IR-06 | Retraining trigger after decisions | TC-013 | Met | |
| IR-07 | Google sign-in, existing accounts only | TC-043 | Met | Self-service connect added 2 Nov |
| IR-08 | Background jobs on CPU and GPU queues | TC-040 | Partly | CPU queue passes; GPU queue (Colab notebook 05) not yet run |

## Open items this record depends on

- #39, #32: deployment over HTTPS, then failure scenarios and operational measurements on the deployed system.
- TC-015: usability test (researcher).
- #37, #38: record the marked rubric and coherence results in `tests/test_plan.md` and Chapter 5, then close.
- Model release of `evaluated_v1` with a model card.
- FR-11: show the held-out test metrics on the model page, or reword FR-11.
