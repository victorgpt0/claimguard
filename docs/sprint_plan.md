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
| III: Project Completion, Deployment and Final Evaluation | 6, 7, 8 | 19 Oct – course completion review (dates per course announcements) |
| Final documentation and presentation | 8 | 2 Nov – 7 Nov (part of Course Sprint III since the brief was rewritten) |

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

### Sprint 4 (5–11 Oct): Scope change and data preparation — Done (one student task open)
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
- [x] Dashboard shows generated reports; auditor can record a decision. (17 Oct, `3f083c0`, TC-029.)
- [x] Test records in `tests/test_plan.md`: expected result, actual result, correction (valid input, invalid input and failure cases). TC-001 to TC-047 filled except TC-015 (usability, moved to Course Sprint III). Board item #24 still "In Progress".
- [x] Chapter 5 started: environment, implemented modules, initial results (`compare/chapter5.txt`, sections 5.1–5.4.9 drafted). Board item #25 still in the Sprint Backlog.
- [ ] Prepare the review: demo, test evidence, one AI suggestion checked (from `docs/ai_use_log.md`). Blocked on the student's own AI-use log entries (#54).

### Course Sprint II status (checked 2 Nov against `build_phase_sprints.txt`)
Substantively met: the core workflow runs from a claims batch to stored, reviewable audit
reports (pipeline and Django platform), the components are integrated (database, model,
ICD-11 cache, LLM, web interface, workers), initial tests cover valid, invalid and failure
cases, and the evaluation goes well beyond a baseline (held-out test results, TC-045/046).
Still open before it can be called closed:
- [x] **GitHub Actions:** `.github/workflows/tests.yml` runs the test suite on every push and pull request to `dev`; `deploy.yml` tests and deploys every merge into `main` (10 Nov: all runs green).
- [ ] AI-use log: one entry still has `[Student: fill in]` review fields, and the review needs "one AI suggestion checked" (#54, student's own).
- [~] Board: #25 closed 10 Nov. #24 stays open for TC-015 (usability); #26 (review preparation) and #54 stay open.
- [x] `dev` pushed and the interface work committed (#63).

## Course Sprint III: Project Completion, Deployment and Final Evaluation

### Rewritten course requirements (brief updated; mapped 2 Nov)
The brief now asks for a complete, usable, deployed and evaluated project, not only
"approaching completion". Each expectation below is mapped to existing work or a gap.
- [ ] **Traceability record:** objective or requirement → component → test or evaluation → result → evidence. `tests/test_plan.md` covers FR/NFR/DR; an objective-level table for Chapter 5, with unmet requirements explained, is missing.
- [~] **Coherent, task-based interface:** consistent design system, navigation, roles, dark mode and sign-in done 2 Nov; empty, error and permission-denied states covered on every page (#63). Still to check: keyboard use, visible focus, contrast, status never shown by colour alone; mobile layout.
- [x] **Honest uncertainty:** the report page showed the classifier's softmax output as "% confidence"; the brief says a raw score is not a calibrated probability. Relabelled "model score" with an explanation, CSV column `classifier_score` (2 Nov). Calibration not evaluated: Chapter 5 should say so. The narrative prompt still passes the score as "Classifier confidence" (frozen, evaluated pipeline); none of the 229 stored narratives repeats it.
- [ ] **Feedback as training data:** the brief asks that feedback be reviewed before it becomes training data. Auditor decisions are the review, and FR-14 gates promotion; Chapter 5 should state this argument explicitly.
- [x] **Operational release over HTTPS** with the real backend, database and model (#32, #39): Oracle Cloud Always Free VM (uk-london-1) at claimsguard.tech behind Cloudflare, PostgreSQL and Redis on the VM, deployed automatically from `main` (9–10 Nov; runbook `deploy/README.md`). Still to do for #39: record the tested setup in `docs/environment.md`, which still describes Neon and Upstash.
- [~] **Colab GPU worker against the deployed release:** `notebooks/05_gpu_worker.ipynb` runs and connects to the deployed queue (10 Nov); it has not yet processed a batch. One uploaded batch run end to end on the GPU queue still to record (TC-040).
- [x] **Claims ingestion on the live release (#64–#66):** `live-v1` generated and ingested into the deployed site (by 10 Nov). Record the live run in TC-049, which still describes the 3 Nov smoke run on the deployment pool.
- [ ] **Model release:** frozen classifier (`evaluated_v1/`) published with a model card and a pinned revision (brief Example 3; `docs/completion_plan.md` already plans the Hugging Face Hub hand-off).
- [ ] **Testing and evaluation:** usability test with a representative user (TC-015), failure scenarios on the deployed system (expired session, unavailable worker or database, duplicate submission), defects corrected and retested.
- [ ] **Operational evidence:** response times, failure rates, a restart and an unavailable-dependency test on the deployed release.
- [ ] **Report:** Chapters 5 and 6, corrections to Chapters 1–4 (#40, #41, #42), AI-use declaration (#43).
- [ ] **Reproducible handover:** release tag (#44), live URL or operating instructions, `.env.example`, migrations, model and data access instructions, user guide, known limitations, offer expiry and cost plan, secure assessor access.
- [ ] **Demonstration:** representative test inputs, backup recording and screenshots, clearly labelled.

### Decisions for Sprints 6–8 (17 Oct)
Code instructions in `docs/remaining_work_plan.md`.
- **Funnel (#50):** stage 1 = Isolation Forest tuned for recall + autoencoder + a TF-IDF
  text-consistency screen, with NFR-05's recall floor applied to the union (TC-030).
  Chapters 3/4 need a "text-consistency screen" box in stage 1.
- **Consistent verdicts (#46):** two review lanes, "text inconsistent" and "structural
  anomaly"; a screen-only flag the classifier clears is logged, not queued. FR-05 wording
  to update.
- **Explanation scoring (#37):** single rater (the researcher), no Krippendorff's alpha;
  Section 3.2.4 to be changed and the limitation stated.
- **Hosting (#32):** Cloudflare Tunnel from the researcher's machine recommended
  (`docs/hosting_guide.md` option A); to confirm.
- **Commits:** one per issue on `dev`, pushed after review.

### Sprint 6 (19–25 Oct): Full training and feedback loop
Started early on 17 Oct. Status on 19 Oct below.
- [x] Caesarean "CS" shortcut removed from the training noise before regenerating (#51).
- [x] Stage-1 tuning script and text screen; v0.5 values in `config.example.yaml` (#28, #50, TC-030). Re-run on v0.6.
- [x] Review lanes and flag sources in the pipeline and dashboard (#46); first validation run with tuned stage 1 (TC-031).
- [x] Code: seed summary (#29), rule-based baseline (#30), retraining script and history (#31, #47, #48, #49), evaluation harness (#33–#38).
- [x] **Open question from TC-031:** on validation the rule-based baseline has the higher F1 (0.60 vs 0.51): the structural-anomaly lane carries most of the pipeline's false positives, and the rules catch inflated caesarean, which the pipeline misses. Resolved 25 Oct (TC-036): configurations A and C are both reported; stage 1 + rules is future work only (10 Nov).
- [x] Full dataset (18 Oct): v0.6-20k, 20,000 claims, 240 providers, 333 per typology, tagged `v0.6-data` (`docs/data_generation.md`).
- [x] Stage 1 re-tuned on v0.6 (19 Oct, TC-032): contamination 0.15, autoencoder quantile 0.975, screen threshold 0.005553; recall 0.81 ± 0.02. In `config.example.yaml`.
- [x] Classifier fine-tuned with all 5 seeds on Colab (18 Oct, TC-033): validation F1 0.975 ± 0.004. Best seed 7 copied to Drive `models/classifier_lora/active/` and frozen as `evaluated_v1/` (adapter files directly inside each folder, not in a run subfolder). "CS" shortcut largely gone; new defect: 0.8% of notes partly in Chinese (validator fixed for future data).
- [x] Pipeline run on the v0.6 Deployment/Demonstration Pool with narratives (18 Oct, notebook 03, TC-034): 354 reports, 82 narratives, 0 failures. Stage-1 settings as in TC-032 (screen threshold 0.005553); classifier seed 7.
- [x] Rule-based baseline run on v0.6 validation (222 of 3,600 flagged). Structured-only and zero-shot LLM baselines written and run on 25 Oct (#30, #52, TC-037).
- [x] Feedback loop: retraining run on 25 Oct; the candidate was refused by the FR-14 gate (#31, TC-013, TC-014). Before that, 80 simulated decisions were recorded on 19 Oct (36 approved, 44 rejected; 59 examples, all tagged simulated; 44 old v0.5 reports unmatched and left pending). **Do not rerun `--simulate-auditor`.** Remaining: the retraining run (`python -m src.feedback.run_retraining --config config/config.example.yaml --classifier-data /content/classifier_data` on Colab after re-exporting), which waits for a GPU: no Colab compute units on 19 Oct; on the workstation CPU one epoch takes about 1.8 h.
- [x] Deployment requirements identified; hosting option chosen (#32): Oracle Cloud Always Free VM behind Cloudflare (9 Nov), replacing the 17 Oct Cloudflare Tunnel recommendation.

### Sprint 7 (26 Oct – 1 Nov): Evaluation and hosting
- [x] Validation comparison on v0.6 and choice of reported configurations (25 Oct, TC-036): configuration A (the pipeline as designed) and C (`hybrid_text_rules`: text lane OR any rule) are both reported on the test split. Phase 0 order in `docs/platform_plan.md`.
- [x] GPU baselines and the retraining run on validation (25 Oct, Colab session A; TC-013, TC-014, TC-037): structured-only F1 0.585 ± 0.023 over 3 seeds; zero-shot LLM F1 0.594 against the classifier's 0.952 on the text label (first 1,000 claims); the retrained candidate was refused by the FR-14 gate.
- [x] Held-out evaluation against all four baselines: McNemar's test, bootstrap confidence intervals, mean and standard deviation over the structured-only seeds (3, for GPU time) (27 Oct, Colab session B; TC-045, TC-046): A F1 0.561, C 0.826, rules 0.578, structured-only 0.588 ± 0.021; text lane 0.890 against the zero-shot LLM's 0.578 on the text label.
- [x] Stratified results: clean versus noisy text, common versus rare ICD-11 codes (TC-045, TC-046).
- [x] Cost summary table (fraud value flagged, auditor-hours) (TC-045).
- [x] Automated faithfulness check script: a narrative passes only if it names the billed code's official ICD-11 title and refers to specific content from the claim's clinical notes. Run on every report (28 Oct, TC-047): 141 of 146 test-split narratives pass; of all 11 failures across both databases, 3 are real (definition text presented as note content), 8 are the rule's false negatives.
- [x] Researcher scores 50 random reports on the five-criterion Likert rubric (single-rater; no Krippendorff's alpha) (1 Nov, #37, TC-050): means 4.10–4.74 on four criteria; actionability 2.00 is the weak one.
- [x] SHAP coherence check on 100 flagged claims (1 Nov, #38, TC-051): 75 of 100 coherent; every structural typology fully coherent, ghost billing 4/15 and record fabrication 3/14.
- [~] Project deployed online (9–10 Nov, claimsguard.tech); the tested setup still to record in `docs/environment.md` (#39).
- [x] Django platform, decided 25 Oct (`docs/platform_plan.md`; reverses the 12 Oct descoping of sign-in and roles), built on branch `django` alongside phase 0: accounts, roles and Google sign-in (#62, #61), review pages (#63), upload with validation and jobs on CPU and GPU queues (#57, #58), hosted Neon PostgreSQL and Upstash Redis (26 Oct), batch summary with Markdown and CSV downloads (#59), user management and activity log (#56). TC-038 to TC-044.
- [x] Documentation of the platform (#60, 26 Oct): Chapter 4 diagrams in `docs/diagrams/`, paste-ready Chapter 3 and 4 text in `compare/changes.txt` section D, Chapter 5 draft, test plan, README, hosting guide.
- [~] Colab GPU worker run once end to end (`notebooks/05_gpu_worker.ipynb`): works and connects (10 Nov), but has not yet processed a batch.
- [~] `django` merged into `dev` (1 Nov, `41101b9`); supervisor agreement to the change of stack still to confirm.
- [x] Claims ingestion (3 Nov): simulated SHA claims platform (#65), live feed data set (#64, `live-v1` generated), scheduled pull and Data pipeline page (#66). TC-048, TC-049.

## Final

### Sprint 8 (2–7 Nov): Documentation and presentation
- [ ] Chapter 5 completed with verified results; Chapter 6 written.
- [ ] Chapters 3 and 4 updated from planned to completed wording with actual numbers.
- [ ] Abstract revised last; Gantt chart appendix redrawn to this plan.
- [ ] AI-use declaration distilled from the log.
- [ ] Final release tagged (`v1.0`); presentation prepared.

### Remaining implementation (checked 10 Nov)
In order; tag `v1.0` (#44) only once the rest is done.
- [ ] One uploaded batch processed end to end by the Colab GPU worker against the deployed release (TC-040).
- [ ] TC-049 updated with the live `live-v1` ingestion (done; only the record is missing).
- [ ] Model release: `evaluated_v1/` on the Hugging Face Hub with a model card and a pinned revision.
- [ ] Usability test with a representative user (TC-015, #24).
- [ ] Failure scenarios on the deployed system: expired session, unavailable worker or database, duplicate submission; defects fixed and retested.
- [ ] Operational evidence: response times, failure rates, a restart and an unavailable-dependency test; tested setup in `docs/environment.md` (#39).
- [ ] Accessibility pass: keyboard use, visible focus, contrast, status not by colour alone, mobile layout.

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
