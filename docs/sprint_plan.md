# Sprint Plan

One-week sprints (Monday to Sunday), nested inside the course's three build-phase
sprints from the lecturer's timetable. Each course sprint keeps its expectations; the
weekly sprints split that work into reviewable increments, following Scrum for One
(Section 3.3). Project end: first week of November 2026.

Sprints 1–3 are reconstructed from evidence that already exists (git commits and file
modification dates). They describe what was actually done in those weeks; nothing is
backdated.

| Course sprint | Weekly sprint | Dates |
| --- | --- | --- |
| I: Data Preparation and Environment Setup | 1, 2, 3 | 14 Sep – 4 Oct |
| II: Core Functionality, Integration and Initial Testing | 4, 5 | 5 Oct – 18 Oct (progress review in Sprint 5) |
| III: Approaching Completion and Hosting | 6, 7 | 19 Oct – 1 Nov |
| Final documentation and presentation | 8 | 2 Nov – 7 Nov |

---

## Course Sprint I: Data Preparation and Environment Setup

### Sprint 1 (14–20 Sep): Documentation corrections and design completion — Done
- Chapter 3 patches for the feedback loop and Chapter 4 design notes finalised.
- Documentation build scripts updated; conceptual framework figure redrawn.
- **Evidence:** `proposal_revision/` files dated 13, 14, 17 and 20 Sep.

### Sprint 2 (21–27 Sep): Repository and implementation plan — Done
- `claimguard` repository created on GitHub; licence and initial structure.
- `implementation_roadmap.md` written (environment, Drive layout, tracking).
- Chapter 3 rigour improvements and Chapter 4 merged; documentation rebuilt.
- **Evidence:** commits on 21 Sep; documentation build dated 24 Sep.

### Sprint 3 (28 Sep – 4 Oct): Environment and core modules — Done
- Development environment, configuration and secrets handling (`.env.example`, `config.example.yaml`).
- Self-hosted WHO ICD-11 API (Docker) and ICD-11 client with cache.
- All pipeline classes implemented: preprocessing, fraud injection, Isolation Forest,
  Autoencoder, SHAP, classifier and training loop, narrative generator, report
  repository, feedback repository and retrainer, authentication, dashboard.
- 58 unit tests, passing (74 after Sprint 4).
- **Evidence:** 7 commits on 28 Sep.
- **Carried over:** no dataset was prepared in Course Sprint I. It moves to Sprint 4.

## Course Sprint II: Core Functionality, Integration and Initial Testing

### Sprint 4 (5–11 Oct): Scope change and data preparation — In progress
- [x] English-only scope approved; code-switching replaced by text-noise injection; classifier switched to Bio_ClinicalBERT (60 tests passing).
- [x] `.gitignore` fixed so `src/data/` is version-controlled.
- [x] Chapters 1–4 condensed and converted to past tense.
- [x] Commit and push all of the above.
- [x] Base claims source decided (11 Oct): a custom generator over a catalogue of 37 ICD-11 codes priced from LN 56/2025, replacing DE-SynPUF/Synthea. Synthea's case mix is US-specific (no malaria), it codes in SNOMED, and neither source has ICD-11 codes or clinical free text. Chapter 3 Section 3.2.1 wording must be updated to match.
- [x] Dataset generation code (11 Oct): stages A-C (`src/data/generator.py`, `note_synthesis.py`, `build_dataset.py`) plus `notebooks/01_generate_dataset.ipynb`. All six Table 3.1 typologies, 10% prevalence, 30% text noise, DR-08 pool, then 60/20/20; `is_fraud` now produced. Tested end to end with a fake LLM (74 tests).
- [x] Run the notebook on Colab (11 Oct): 5,000 claims, Qwen2.5-7B notes (0 rejected), splits 2,700 / 900 / 900 / 500, ~10% fraud. Fixed a vLLM out-of-memory crash on the T4 along the way. Dataset on Drive, documented in `docs/data_generation.md`, tagged `v0.4-data`.
- [ ] Log the notes review in `docs/ai_use_log.md` (student's own entry).
- [x] Missing-value and numeric-perturbation noise for structured fields (12 Oct): 10% of claims, preprocessing repairs codes and imputes blanks. Dataset rebuilt as `v0.5-5k`; baseline F1 0.45 (v0.4: 0.48) over 5 seeds (TC-023).
- [x] Isolation Forest baseline (12 Oct): validation precision 0.47, recall 0.55, F1 0.51 (contamination 0.10). Catches conversion and fictitious dependents fully; misses ghost billing and fabrication, as designed (TC-022).

### Sprint 5 (12–18 Oct): Working core workflow — Progress review
- [x] Orchestration script: claims batch → preprocessing → Isolation Forest and Autoencoder → SHAP → classifier → ICD-11 lookup → LLM narrative → report saved to SQLite. (12 Oct: `src/pipeline.py` built and wiring-tested on the 500-claim pool with a stand-in classifier, 19 s on CPU, 44 reports the dashboard reads (TC-025). First real run 12 Oct on Colab: 44 reports, 1 narrative, end-to-end recall on text-inconsistent claims 1/25 because of the funnel (TC-028).)
- [x] Fill the ICD-11 cache for every code in the dataset (12 Oct): 38 codes in `artifacts/icd11_cache.json`; residual categories inherit their parent's definition (TC-024).
- [x] Choose the LLM provider and model; run the narrative layer on real flagged claims. (12 Oct: Hugging Face Inference returned 402, no account credit, so the narrative model runs on the Colab T4 with vLLM instead: 4-bit Llama 3.1 8B Instruct, `notebooks/03_run_pipeline.ipynb`. Gershon et al. (2025) turned out not to support a model choice; the choice rests on practical grounds and three passages citing it for ICD-11 need rewording in the documentation. Real run pending.)
- [x] Single-seed classifier fine-tune on Colab (12 Oct): seed 42, validation precision 0.917, recall 0.978, F1 0.946 (TC-026). Per-typology breakdown (TC-027, `src/classification/evaluate.py`): fabrication 15/15, upcoding 15/15, ghost 14/15; false positives are mostly genuine caesarean notes written as "CS", a word-matching shortcut to address in the next dataset version.
- [ ] Dashboard shows generated reports; auditor can record a decision.
- [ ] Test records in `tests/test_plan.md`: expected result, actual result, correction (valid input, invalid input and failure cases).
- [ ] Chapter 5 started: environment, implemented modules, initial results.
- [ ] Prepare the review: demo, test evidence, one AI suggestion checked (from `docs/ai_use_log.md`).

## Course Sprint III: Approaching Completion and Hosting

### Sprint 6 (19–25 Oct): Full training and feedback loop
- [ ] Full dataset (about 20,000 claims), regenerated if Sprint 4 used a smaller one.
- [ ] Autoencoder threshold and Isolation Forest contamination tuned on the validation split; values recorded.
- [ ] Classifier fine-tuned with all 5 seeds on Colab; evidence saved per run in Drive.
- [ ] Rule-based baseline and structured-only baseline implemented.
- [ ] Feedback loop demonstrated once: decisions → Feedback Dataset → retraining → promotion check.
- [ ] Deployment requirements identified; hosting option chosen from `docs/hosting_guide.md`.

### Sprint 7 (26 Oct – 1 Nov): Evaluation and hosting
- [ ] Held-out evaluation against all four baselines: McNemar's test, bootstrap confidence intervals, mean and standard deviation over 5 seeds.
- [ ] Stratified results: clean versus noisy text, common versus rare ICD-11 codes.
- [ ] Cost summary table (fraud value flagged, auditor-hours).
- [ ] Automated faithfulness check script: a narrative passes only if it names the billed code's official ICD-11 title and refers to specific content from the claim's clinical notes. Run on every report.
- [ ] Researcher scores 50 random reports on the five-criterion Likert rubric (single-rater; no Krippendorff's alpha).
- [ ] SHAP coherence check on 100 flagged claims.
- [ ] Project deployed online and the tested setup recorded in `docs/environment.md`.

## Final

### Sprint 8 (2–7 Nov): Documentation and presentation
- [ ] Chapter 5 completed with verified results; Chapter 6 written.
- [ ] Chapters 3 and 4 updated from planned to completed wording with actual numbers.
- [ ] Abstract revised last; Gantt chart appendix redrawn to this plan.
- [ ] AI-use declaration distilled from the log.
- [ ] Final release tagged (`v1.0`); presentation prepared.

---

## Later improvements (deferred, not in the current scope)

Recorded so they are not lost; the original funnel design stays for now.

- **Stage-1 text-mismatch screen** (12 Oct finding). In the funnel, the classifier only sees
  claims the anomaly detectors flag, and Isolation Forest flags 0-1 of 15 ghost-billing and
  record-fabrication claims (their fraud is only in the note). Prototype: flag a claim in
  stage 1 if the detectors flag it OR its note has zero TF-IDF word overlap with its code's
  reference text. Validation: ghost billing passed to the classifier 0% -> 80%, fabrication
  0% -> 100%, stage-1 recall 0.36 -> 0.67, 175 of 900 claims passed on (14% of clean
  claims). Limits: crude word overlap (12% of clean notes also score 0); a clinical
  sentence-embedding similarity would separate better. Needs Chapter 3/4 wording (a
  "text-consistency screen" box in stage 1).
- **Tune Isolation Forest for recall, not F1.** As the first stage of a funnel it should be
  generous; F1 tuning picks contamination 0.05, which lets inflated caesarean and most
  upcoding through. The autoencoder already uses a recall floor (NFR-05); apply the same.
- **Random audit sample** of unflagged claims (e.g. 2%) sent to the classifier anyway, to
  measure what the funnel misses.
- Until then, Chapter 5 must state the funnel's limitation with the numbers: text-only
  typologies are mostly lost at stage 1 (also noted in `docs/completion_plan.md`).

## Working rules
- Tag the end of each sprint that produces a working increment (`v0.4-data`, `v0.5-core`, …). Never backdate tags or commits.
- Reference the GitHub issue in commits (`closes #12`).
- Keep `tests/test_plan.md` and `docs/ai_use_log.md` updated as work happens.
