# Phase 0: what remains (26 Oct)

Validation is finished and frozen. The test split has not been read. Start here in a new session.

## State

- Commits on `dev` up to `bf04426`, plus this file. The local tag `v0.9-frozen` marks the state
  before the test split is read. **Push both before Colab:** `git push origin dev v0.9-frozen`.
- `notebooks/04_baselines.ipynb` has uncommitted session A outputs. Commit them as evidence, or
  discard them.
- Validation records are in `tests/test_plan.md`:
  - TC-036: configurations A and C.
  - TC-037: structured-only and zero-shot.
  - TC-013 and TC-014: retraining. The candidate was not promoted.
- Evaluations are on Drive under `experiments/evaluation/`. Predictions are under
  `experiments/predictions/v0.6-20k/`.
- Systems reported on test:
  - pipeline (A)
  - `hybrid_text_rules` (C)
  - `rule_based`
  - `isolation_forest` (contamination 0.05)
  - `structured_only_seed{42,123,2024}`
  - `zero_shot_llm`, `zero_shot_llm_text_lane` and `zero_shot_llm_funnel`
  - `pipeline_text_lane`

## 1. Colab session B (GPU; a free-tier T4 is enough, session A ran on one)

Colab clones `dev` from GitHub, so push first. Close the dashboard and don't write the SQLite
database from the laptop at the same time.

1. Run `notebooks/03_run_pipeline.ipynb` with:
   - `BATCH = 'test'`
   - `FINAL = True`
   - `WRITE_PREDICTIONS = True`
   - `CLASSIFIER_RUN = '20261018T181622_seed7'`

   The notebook writes to `db/evaluation_v0.6-20k_test.sqlite`, never `sha_fraud.sqlite`.
2. Run `notebooks/04_baselines.ipynb` with:
   - `SPLIT = 'test'`
   - `FINAL = True`
   - `TRAIN_STRUCTURED = False`
   - `RUN_ZERO_SHOT = True`
   - `ZERO_SHOT_LIMIT = None` (all 3,600 claims; about 25 min)
   - `RUN_RETRAINING = False`

   `structured_only predict` still runs, using the three trained seeds. `combine` runs only if
   notebook 03 has already written `pipeline_stage1/test.csv`.

If GPU time runs short, `ZERO_SHOT_LIMIT = 1000` also works. Every system is then compared on
those claims with `--claims-from zero_shot_llm`, and Chapter 5 must state the limit.

## 2. Laptop, after session B

Run from `claimguard/`:

```bash
D=~/gdrive/claimguard/data/processed/v0.6-20k; P=~/gdrive/claimguard/experiments/predictions/v0.6-20k
# only if notebook 04 printed "pipeline_stage1/test.csv not found":
venv/bin/python -m src.baselines.zero_shot_llm combine --split test --final --predictions-root $P
venv/bin/python -m src.baselines.rule_based --data $D --split test --final --predictions-root $P
venv/bin/python -m src.detection.evaluate_baseline --data $D --split test --final --contamination 0.05 --predictions --predictions-root $P
venv/bin/python -m src.evaluation.hybrid --split test --final --predictions-root $P
venv/bin/python -m src.evaluation.heldout --data $D --split test --final --predictions-root $P --truth is_fraud \
  --systems pipeline hybrid_text_rules rule_based isolation_forest zero_shot_llm_funnel \
            structured_only_seed42 structured_only_seed123 structured_only_seed2024
venv/bin/python -m src.evaluation.heldout --data $D --split test --final --predictions-root $P --truth text \
  --reference pipeline_text_lane --systems pipeline_text_lane zero_shot_llm zero_shot_llm_text_lane
```

Add `--claims-from zero_shot_llm` to the heldout commands only if the zero-shot run was limited.
Run each test command once. Don't rerun them to tune anything.

## 3. Write-up

- Copy `experiments/evaluation/v0.6-20k_test_*` to Drive.
- Add test-plan rows for the test results: main metrics, McNemar, seed mean ± std, cost, strata.
- Tick the Sprint 7 evaluation lines in `docs/sprint_plan.md`.
- Draft the Chapter 5 tables in `compare/chapter5.txt`. Caveats to state:
  - C was chosen on validation among three options (A, B: pipeline OR caesarean rule, and C).
  - The rules encode the injected typologies, so their share of C's result is partly by
    construction.
  - C misses anomalous dependents (0.28 against A's 1.00).
  - Structured-only used 3 seeds, not 5.
  - The feedback decisions were simulated.
  - The FR-14 gate read test labels for pass/fail only.
- Then move #30, #31, #33, #34, #35 and #52 to Review.

## Never

- Rerun `--simulate-auditor`.
- Overwrite `evaluated_v1`.
- Read the test split without `--final`.
- Commit `config/config.yaml`, `.env` or `.streamlit/secrets.toml`.

The Django work (`docs/platform_plan.md`) starts only after these results exist and the
supervisor agrees to the change of stack.
