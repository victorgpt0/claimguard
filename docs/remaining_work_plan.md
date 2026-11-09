# Remaining Work Plan (Sprints 5–8)

Written 17 Oct 2026, the last day of Sprint 5. It turns the open items in
`docs/sprint_plan.md` into concrete code and run instructions. `sprint_plan.md` stays the
record of what was done. This file says *how* to do what is left. Every task names its
GitHub issue: the board is **ClaimGuard: Scrum for One**
(https://github.com/users/victorgpt0/projects/2).

Ground rules that apply to every task below:

- **The LLM is an explainer, never a detector.** The zero-shot LLM baseline (task 6.4)
  is a comparison system. It lives in `src/baselines/` and `src/pipeline.py` never
  imports it.
- **The test split is read once**, at the final evaluation (task 7.1), behind a `--final`
  flag, the way `src/classification/evaluate.py` already works. Tuning, seed selection
  and threshold choice all use the validation split.
- **Every run leaves evidence** in `experiments/<area>/<run_id>/` (config, metrics,
  predictions) and a row in `tests/test_plan.md`. Chapter 5 quotes these numbers and
  nothing else.
- **Tag each working increment:** `v0.6-data` (20k dataset), `v0.6-core` (tuned pipeline
  plus baselines), `v0.7-eval` (held-out results), `v1.0` (submission). Never backdate.

---

## 0. Status on 17 Oct

| Area | State |
| --- | --- |
| Dataset | `v0.5-5k` (5,000 claims, splits 2,700/900/900/500) on Drive |
| Stage 1 | Isolation Forest and autoencoder fitted inside `src/pipeline.py`; contamination hardcoded at 0.05 (`DEFAULT_CONTAMINATION`); no tuned values in `config.yaml` |
| Classifier | One seed (42), validation F1 0.946 (TC-026, TC-027) |
| ICD-11 cache | 38 codes, complete (TC-024) |
| Narrative | vLLM + 4-bit Llama 3.1 8B on Colab; 1 narrative produced so far (TC-028) |
| Pipeline | End to end, tagged `v0.5-core`; recall on text-inconsistent claims 1/25 because of the funnel |
| Dashboard | Review queue, decisions and feedback tabs committed (`3f083c0`, TC-029) |
| Feedback → retrainer | Contract fixed in `3f083c0`; no retraining script; `mark_consumed()` never called |
| Baselines | Isolation-Forest-only exists (`evaluate_baseline.py`); rule-based, structured-only and zero-shot LLM do not |
| Evaluation | Validation stratification exists (`evaluate.py`); no held-out harness, bootstrap, McNemar, cost or faithfulness code |
| Tests | 104 passing; TC-001–005 and TC-007–015 empty |

---

## 1. Decisions to make first

**Decided 17 Oct** (recorded in `docs/sprint_plan.md`): D1 recall tuning + text screen;
D2 two review lanes; D4 single rater. D3 and D5 as recommended unless changed. Code for
6.2, 6.4a/b, 6.5 and 7.1–7.6 is written; see the issues for what still needs running.

These change what gets built. Make each one, write it in `docs/sprint_plan.md`, then
start the code.

**D1. The stage-1 funnel (#50).** Text-only fraud (ghost billing, record fabrication)
almost never passes stage 1, so the classifier never sees it. Options:
(a) leave it and report the limitation; (b) tune Isolation Forest for recall with a
floor, like the autoencoder; (c) add a text-consistency screen to stage 1;
(d) a random audit sample of unflagged claims.
**Recommendation: (b) + (c).** (b) is a few lines and matches NFR-05. (c) is what makes
the classifier matter end to end: without it the headline pipeline result is close to
Isolation Forest alone, which undermines the research contribution. The prototype
already showed ghost billing 0% → 80% and fabrication 0% → 100% reaching the
classifier. Use a sentence-embedding similarity rather than TF-IDF overlap if time
allows (12% of clean notes have zero overlap). (c) needs a box in Chapter 3/4's
stage 1 (#41).

**D2. What a consistent verdict means (#46).** Conversion and anomalous-dependent fraud
have consistent notes, so the classifier calls them consistent, yet they are real fraud
that only stage 1 catches. Hiding consistent-verdict reports, as FR-05 currently says,
would remove them from the auditor's queue.
**Recommendation:** keep a report for every stage-1 flag, with two queue lanes:
"text inconsistent" (verdict inconsistent, has a narrative) and "structural anomaly"
(verdict consistent, SHAP only). Reword FR-05 to match. For evaluation, a **pipeline
positive** is then any claim that gets a report. Also report the "text inconsistent"
lane alone, because that is where the classifier's contribution shows.

**D3. Hosting (#32).** Recommendation: **Cloudflare Tunnel from your own machine**
(`docs/hosting_guide.md`, option A). Free-tier HF Spaces wipes the SQLite database and
the feedback file on every restart, which breaks FR-12 to FR-14. The narrative step
needs a GPU, so the hosted demo shows reports produced on Colab and records decisions.
It does not generate new narratives live. Say so in Chapter 5.

**D4. Explanation scoring (#37).** The documentation says three reviewers plus
Krippendorff's alpha; the sprint plan says one rater (you). Either find two reviewers
this week or change Section 3.2.4 to a single rater and name it as a limitation.
Recommendation: single rater, stated honestly. Recruiting a medical coder and an auditor
in two weeks is not realistic.

**D5. Which checkpoint Chapter 5 reports.** The feedback demonstration (task 6.5) may
promote a new checkpoint, and the FR-14 promotion check reads the held-out test set.
To keep the reported results clean, **freeze** the checkpoint chosen on validation as
`models/classifier_lora/evaluated_v1/` before the feedback demonstration, and report
that one in Chapter 5. The promotion check uses the test set only as a pass/fail gate,
and its metrics are not reported as results.

---

## 2. Sprint 5 wrap-up (17–18 Oct)

### 5.1 Test records TC-001 to TC-015 (#24)
Most of these requirements already have unit tests. Fill each row from the test that
covers it: run that test file, paste the command, and record what it checks. Do not
invent new results.

| TC | Requirement | Covered by | Notes |
| --- | --- | --- | --- |
| TC-001 | FR-01 ingest | `test_ingestion.py`, `test_preprocessing.py` | include an invalid-input case (bad code repaired, blank imputed) |
| TC-002 | FR-02 / NFR-01 | `test_isolation_forest.py`, `test_autoencoder.py`, TC-022 | NFR-01 throughput: quote TC-025's 500 claims in 19 s on CPU |
| TC-003 | FR-03 SHAP | `test_shap_explainer.py` | it mocks KernelSHAP; the real run is TC-025/TC-028 |
| TC-004 | FR-04 / NFR-05 | `test_classifier.py`, TC-026 | |
| TC-005 | FR-05 | `test_pipeline.py` | rewrite after D2 |
| TC-007 | FR-07 narrative | `test_narrative_generator.py` | include the `ValueError` gate for consistent verdicts |
| TC-008 | FR-08 / NFR-03 | `test_repository.py`, `test_dashboard.py` | NFR-03: auth was descoped; say so |
| TC-009 | FR-09 decisions | `test_dashboard.py`, `test_repository.py` | |
| TC-010 | FR-10 config thresholds | `test_config.py` | update after task 6.2 writes real values |
| TC-011 | FR-11 admin view | — | fill after #47/#49 |
| TC-012 | FR-12 feedback | `test_feedback_repository.py` | |
| TC-013 | FR-13 / IR-06 trigger | `test_retrainer.py` | complete after #48 |
| TC-014 | FR-14 promotion | `test_retrainer.py` | complete with the task 6.5 run |
| TC-015 | NFR-07 usability | — | a short walkthrough of the dashboard with a classmate; record what they could and couldn't do |

### 5.2 AI-use log (#54)
Yours to write. It must record your own prompts and how you checked each suggestion.
Add entries for: the dataset generation (11–12 Oct), the notes review, the pipeline and
dashboard work, and this plan. Pick one checked suggestion for the progress review
(the TC-006 ICD-11 client bugs are a good one: a suggestion that passed mocked tests and
failed live).

### 5.3 Start Chapter 5 (#25)
Write 5.1 (environment, from `docs/environment.md`) and 5.2 (implemented modules,
one paragraph per class, pointing at the file). Leave results for Sprint 7.

### 5.4 Progress review (#26)
Demo: open the dashboard on the TC-028 database, walk one claim from queue to decision
and show the feedback example appear. Show the test plan and the board.

---

## 3. Sprint 6 (19–25 Oct): full training, baselines, feedback loop

Order matters: 6.1 → 6.2 → 6.3 → 6.4 can overlap with 6.3 → 6.5 last.

### 6.1 Fix the "CS" shortcut, then generate the 20k dataset (#51, #27)

**Code (local, before Colab):**
1. `src/data/noise_injection.py`: add caesarean shorthand to `CLINICAL_ABBREVIATIONS`
   (`"caesarean section": "CS"`, `"lower segment caesarean section": "LSCS"`). Because
   noise is applied to every split regardless of label, genuine caesarean notes then
   appear abbreviated with the `consistent` label in training, which removes the
   shortcut. Keep the unseen test glossary (DR-03) different: put `"C/S"` only there.
2. `src/data/note_synthesis.py`: add a style instruction that describes findings
   without naming the diagnosis ("Describe findings and procedure only; do not state
   the diagnosis by name") to a share of requests (data_generation.md issue 2).
3. Scale providers with claims: `--n-providers 240` for 20,000 claims, so per-provider
   typologies (caesarean share, dependents) keep realistic volumes.
4. Test: extend `tests/test_noise_injection.py` with a case asserting "caesarean
   section" becomes "CS" at rate 1.0, and a dataset test that the share of abbreviated
   caesarean notes is similar across labels.

**Run (Colab T4, `notebooks/01_generate_dataset.ipynb`):** set `N_CLAIMS = 20000`,
`N_PROVIDERS = 240`, `DATASET_VERSION = "v0.6-20k"`. Note generation is resumable
(batches of 500), so a disconnect is fine. Expect about 4× the v0.4 runtime.

**Evidence:** `docs/data_generation.md` gets a v0.6 section (counts per split and
typology, prevalence, rejected notes). New test record. Tag `v0.6-data`.

**Done when:** splits about 10,800 / 3,600 / 3,600 / 2,000 (60/20/20 after a 2,000
pool), about 10% fraud, every typology above 300 claims.

### 6.2 Tune stage 1 and implement the funnel decision (#28, #50, #46)

**New file `src/detection/tune_stage1.py`** (CLI, validation split only):

```python
def tune(data_dir: Path, seeds: list[int], min_recall: float = 0.80) -> dict:
    """For each seed: fit preprocessing on train; Isolation Forest contamination chosen
    as the smallest value in [0.05, 0.40] whose validation recall >= min_recall (not max
    F1); autoencoder threshold via AutoencoderDetector.tune_threshold(min_recall).
    Report the chosen values per seed and the stage-1 recall per typology for IF alone,
    AE alone and their OR."""
```

- Add `IsolationForestDetector.tune_contamination_for_recall(...)` next to the existing
  F1 version (keep the F1 version: Chapter 5 can show why it was not used).
- Write the chosen values into `config/config.example.yaml` under `thresholds:`
  (`anomaly_score_threshold`, plus a new `isolation_forest_contamination`), and copy
  them to `config.yaml`. FR-10 says thresholds come from the config file.
- `src/pipeline.py`: read both values from config instead of `DEFAULT_CONTAMINATION`
  and the training quantile. Keep `--contamination` as an override for experiments.

**If D1 includes the text screen**, new file `src/detection/text_screen.py`:

```python
class TextConsistencyScreen:
    """Stage-1 screen: flags a claim whose notes are dissimilar to its code's ICD-11
    reference text. Not a detector of fraud on its own; it only decides which claims
    the classifier sees."""
    def __init__(self, cache: ICD11Cache, method: str = "tfidf"): ...
    def fit(self, train: pd.DataFrame) -> "TextConsistencyScreen": ...   # vectoriser on train notes + references
    def score(self, claims: pd.DataFrame) -> np.ndarray: ...            # cosine similarity, low = suspicious
    def tune(self, val: pd.DataFrame, max_pass_rate: float = 0.15) -> float: ...
```

Use TF-IDF cosine first (CPU, no new dependency). Only try a sentence-embedding model
if there's time. Tune the threshold on validation so the screen passes at most ~15% of
clean claims. In `AuditPipeline.run`, `flagged = forest | autoencoder | screen`, and
record which stage-1 component flagged each claim in the run log (`flag_sources`), so
Chapter 5 can show each component's contribution.

**D2 in code:** add a `lane` to each report (`"text_inconsistent"` or
`"structural_anomaly"`): a column on `AuditReport` in `src/db/schema.sql`, set in
`AuditPipeline._audit`, shown as a filter in `dashboard/app.py`'s review tab. Update
`test_pipeline.py` and `test_dashboard.py`.

**Evidence:** `experiments/stage1_tuning/<run_id>/results.json`; new test record with
per-typology stage-1 recall before and after. Run on the 20k validation split.

### 6.3 Classifier: five seeds (#29)

No new training code; `train.py` already runs all seeds.

1. Export the 20k dataset: `python -m src.classification.export --data data/processed/v0.6-20k --cache artifacts/icd11_cache.json --out data/classifier/v0.6-20k`.
2. Colab, `notebooks/02_train_classifier.ipynb`, `--seeds 42 123 2024 7 31415`.
   About 4× the single-seed time on 4× the data, so budget two Colab sessions. Run
   seeds in separate cells and save each to Drive as it finishes.
3. **New `src/classification/summarize_seeds.py`:** reads every
   `experiments/classifier/*/metrics.json` of a run, prints and saves mean ± std for
   precision, recall and F1 (validation), and names the best-validation-F1 seed.
4. Copy the best seed's adapter to `models/classifier_lora/active/` **and**
   `models/classifier_lora/evaluated_v1/` (D5).
5. Re-run `python -m src.classification.evaluate` on validation for the chosen seed
   (stratified breakdown, and check the "CS" false positives are gone).

**Evidence:** seed summary JSON, test record (TC for 5-seed results).

### 6.4 Baselines (#30, #52)

All baselines write the **shared prediction format**, so task 7.1 treats every system
the same:

```
experiments/predictions/<system>/<split>.csv
claim_id, flagged (0/1), score (float, higher = more suspicious; blank if none)
```

Put the writer in a new `src/evaluation/predictions.py`
(`write_predictions(system, split, frame)`, `read_predictions(system, split)`) and use
it everywhere, including in the pipeline (task 7.1).

**a. Rule-based, `src/baselines/rule_based.py`.** The five rules exactly as Section
3.2.4 states them, constants at the top of the file, fixed before evaluation, never
tuned:

| Rule | Implementation on our columns |
| --- | --- |
| (i) upcoding | `claim_amount > 1.5 × median(claim_amount)` over the same provider's claims for that `icd11_code` (provider history = all claims of that provider in the batch) |
| (ii) conversion | `claim_type == "inpatient"` and stay (`discharge_date − service_date`) ≤ 1 day and the code is in an outpatient-only list (catalogue entries with `payment: capitation`) |
| (iii) duplicate / ghost | another claim in the batch with the same `member_id`, `icd11_code`, `service_date` |
| (iv) caesarean rate | provider's caesarean share among deliveries > 15% + 10 points (WHO threshold + fixed margin) → all that provider's caesarean claims |
| (v) dependents | provider's distinct dependents per principal member above a fixed threshold (set from the catalogue's realistic household size, e.g. > 5) → that provider's dependent claims |

Flag = any rule (logical OR). Run it on validation and (at 7.1) on test. Several rules
may catch little on this data (e.g. upcoding bills a *different* code, so a per-code
median rarely moves). That is a legitimate finding about static rules, not a bug to
tune away. Test file `tests/test_rule_based.py` with one hand-built claim per rule.

**b. Isolation-Forest-only.** Refactor `src/detection/evaluate_baseline.py` to also
write predictions in the shared format, using the contamination from task 6.2.

**c. Structured-only, `src/baselines/structured_only.py`.** Section 3.2.4: same model
family and training procedure, no notes. Serialise the structured fields to a short
text and reuse `train.py` unchanged:

```python
def structured_text(claim) -> str:
    # "code JB20.0 | KES 0 | stay 0 days | outpatient | level 3 | dependent | age 27 F"
def export(data_dir, out_dir): ...  # same JSONL fields as export.py, notes replaced by structured_text,
                                    # icd11_definition kept, label = is_fraud mapped to inconsistent/consistent
```

Label is `is_fraud` (all typologies), not the text label, because this baseline
answers "can structured data alone detect the fraud?". Train 5 seeds on Colab alongside
task 6.3, then write predictions.

**d. Zero-shot LLM, `src/baselines/zero_shot_llm.py`** (Colab, vLLM, same Llama 3.1 8B):

```python
PROMPT = """You are a medical claims auditor. Billed ICD-11 code: {code} — {title}.
Official definition: {reference}
Clinical notes: {notes}
Are the notes consistent with the billed code? Answer with one word: CONSISTENT or INCONSISTENT."""
def predict(claims, cache, llm) -> pd.DataFrame  # parses the first word; unparseable → flagged=0, counted
```

Temperature 0. Save the prompt text and the unparseable count in the run folder. Like
the classifier, it scores the text label, so compare it with the classifier on the
text task, and as a stage-2 replacement inside the funnel (flagged = stage-1 flag AND
LLM says INCONSISTENT) for the end-to-end comparison.

### 6.5 Feedback loop demonstration (#31, #48, #47, #49)

**New `src/feedback/run_retraining.py`** (Colab, GPU):

```python
def run(config, data_dir, seed=42) -> dict:
    feedback = FeedbackRepository(...)
    examples = feedback.read_examples(since_consumed_marker=True)
    feedback.mark_consumed()                         # #48: at the start, so the trigger resets
    train = load_jsonl(train_split) + examples       # same format since 3f083c0
    candidate_dir = models/classifier_lora/candidates/<timestamp>
    train_one_seed(seed, train, val, ..., output_dir=candidate_dir)
    promoted = ClassifierRetrainer(...).evaluate_and_promote(candidate_dir, test_examples)
    append experiments/retraining_history.jsonl:     # #47
      {run_at, feedback_examples, candidate_metrics, active_metrics, promoted, candidate_path}
```

Dashboard: show the "retraining due" notice and the history table in the admin/model
tab, not after the auditor's decision (#49, #47).

**Getting enough decisions.** Run the pipeline on the 20k dataset's 2,000-claim pool
(about 4× more reports than TC-028). Review enough reports in the dashboard to pass
the trigger (`retrain_trigger_count: 50`). If real review is too slow, you may add a
`--simulate-auditor` option that decides from the ground-truth label. If you use it,
**Chapter 5 must say the decisions were simulated.** Prefer at least some real
decisions.

**Evidence:** the history row, before/after metrics, test records TC-013 and TC-014.
The demonstration succeeds whether or not the candidate is promoted: the point is that
the gate works.

### 6.6 Hosting decision and housekeeping (#32, #53)
- Record D3 in `docs/environment.md` ("Deployment / demonstration environment").
- #53: pin `huggingface_hub<2` in `requirements.txt`; `git rm index.html`.

### 6.7 Batch upload screen (#45), only if time is left
A dashboard upload that saves a CSV and calls `AuditPipeline.run` with no narrator on
CPU (narratives need the GPU). Lowest priority of the sprint; drop it to Product
Backlog if Sprint 6 runs late, and say so in Chapter 5.

**End of Sprint 6:** tag `v0.6-core`.

---

## 4. Sprint 7 (26 Oct – 1 Nov): evaluation and hosting

### 7.1 Held-out evaluation (#33)

1. **Pipeline predictions on test.** Run `python -m src.pipeline --batch test` with the
   `evaluated_v1` checkpoint and the tuned config (Colab, with narratives). Add a
   `--predictions` flag that writes `experiments/predictions/pipeline/test.csv`:
   flagged = report produced (D2), score = classifier probability for flagged claims,
   else 0. Also write `pipeline_text_lane` (verdict inconsistent only).
2. Run every baseline on test (6.4), 5 seeds where the system is trained.
3. **New `src/evaluation/heldout.py`** (`--final` required):

```python
def metrics(y, flagged, score) -> dict          # precision, recall, F1, FPR, AUC-ROC (if score)
def bootstrap_ci(y, flagged, score, n=1000, seed=0) -> dict   # 95% percentile CIs, same resamples for all systems
def mcnemar(y, pred_a, pred_b) -> dict          # exact binomial on discordant pairs (scipy.stats.binomtest)
def seed_summary(per_seed_metrics) -> dict      # mean ± std
def main(): writes experiments/evaluation/<run_id>/results.json and tables.md
```

Ground truth is `is_fraud` for end-to-end comparisons, and the text label for the
classifier-vs-zero-shot comparison. Write `tables.md` in the layout Chapter 5 needs,
so the tables paste straight in. Tests: `tests/test_heldout.py` with small hand-made
arrays (known McNemar result, a CI containing the point estimate).

### 7.2 Stratified results (#34)
Generalise `src/classification/evaluate.stratify` to take any system's predictions
plus the test claims (clean vs noisy text, common vs rare codes, per typology). Call it
from `heldout.py` for every system.

### 7.3 Cost summary (#35)
New `src/evaluation/cost.py`: amount-weighted recall (share of fraudulent KES flagged),
fraudulent KES missed, and auditor-hours = flagged claims × minutes per review. Make
the minutes a stated assumption (e.g. 20) and show the result at two values.
Capitation claims carry KES 0, so amount-weighted recall covers only case-rate and
per-diem fraud. Say this next to the table.

### 7.4 Faithfulness check (#36)
New `src/evaluation/faithfulness.py`, run on every narrative in the test run:

```python
def check(narrative, icd11_title, notes, shap_top3) -> dict:
    names_code   = official title (case-insensitive, allow minor punctuation differences) in narrative
    cites_notes  = at least one content word or 2-gram from the notes (stopwords removed) in narrative
    cites_shap   = at least one top-3 SHAP feature mentioned (reported, not required to pass)
    passed       = names_code and cites_notes
```

`cites_shap` is reported because Section 3.2.4 defines faithfulness with SHAP. Either
make it part of the pass rule or reword 3.2.4 (#41). Test with a passing and a
generic narrative.

### 7.5 Rubric scoring (#37)
New `src/evaluation/rubric.py`:
`sample` draws 50 random narrated reports (fixed seed) into
`experiments/rubric/sheet.csv` with columns for the five criteria;
`summarise` gives per-criterion mean ± std and the share of reports ≥ 4.
You score the sheet by hand. Score blind to the claim's ground-truth label.

### 7.6 SHAP coherence (#38)
Same pattern, `src/evaluation/shap_coherence.py`: a stratified sample of 100 flagged
claims across typologies with their top-3 features into a sheet; you mark each
coherent / not coherent with a short reason; `summarise` reports per typology.

### 7.7 Deploy (#39)
Cloudflare Tunnel (D3): copy the test-run SQLite database and the feedback file to
your machine, run Streamlit, start the tunnel. Record the exact versions, URL type and
what works offline in `docs/environment.md`.

**End of Sprint 7:** tag `v0.7-eval`.

---

## 5. Sprint 8 (2–7 Nov): documentation

These are yours to write. The code above produces the numbers.

- **Chapter 5 and 6 (#40):** results come only from `experiments/evaluation/` and
  `tests/test_plan.md`. Include the funnel limitation with numbers whatever D1 decides.
- **Chapters 3 and 4 (#41)**, places that no longer match the code:
  - 3.2.1: base data is a custom generator over a 37-code LN 56/2025 catalogue, not
    Synthea/DE-SynPUF; dataset size and splits from v0.6.
  - 3.2.4: bilingual stratification → clean vs noisy text; "multilingual transformer" →
    Bio_ClinicalBERT; three-reviewer panel → whatever D4 decides; faithfulness
    definition → task 7.4's rule.
  - The three passages citing Gershon et al. (2025) for the model choice.
  - Stage 1 text screen and the two queue lanes, if D1/D2 adopt them (FR-05 wording).
  - The zero-shot baseline's role (comparison only).
- **Abstract and Gantt chart (#42):** last, from final numbers and this plan's dates.
- **AI-use declaration (#43):** distilled from `docs/ai_use_log.md`.
- **v1.0 and presentation (#44).**

---

## 6. Critical path and Colab budget

```
6.1 dataset (Colab, ~1 day) ─▶ 6.2 tuning (CPU) ─▶ 6.3 five seeds (Colab, 2 sessions)
                                       │                 └─▶ freeze evaluated_v1
                                       └─▶ 6.4 baselines (rule/IF on CPU; structured-only + zero-shot on Colab)
                                                         └─▶ 6.5 feedback demo (Colab)
7.1 pipeline test run (Colab) ─▶ 7.1–7.3 metrics (CPU) ─▶ 7.4–7.6 narrative/SHAP checks ─▶ 7.7 deploy
```

Colab sessions needed: dataset notes (1–2), five classifier seeds (2), structured-only
seeds (1), zero-shot baseline (1), feedback retrain (1), pipeline test run with
narratives (1). Everything else runs on CPU locally. If Colab quota runs short, cut in
this order: structured-only to 3 seeds, zero-shot on a 1,000-claim test sample (state
it), the batch upload screen.
