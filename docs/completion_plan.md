# Completion Plan: What's Left, What Trains Where, What's Stored Where

This is the single place that answers "is the implementation done, and if not, what
exactly remains and how does it all fit together." `implementation_roadmap.md` (in the
documentation repo) covers *environment setup* (already done); this file covers *finishing
the implementation* — training, storage, and runtime architecture end to end.

## 1. Current state (2026-09-28)

| Layer | Code | Trained/run on real data? |
|---|---|---|
| Data ingestion / preprocessing / noise injection (`src/data/`) | Done, unit-tested (41 tests, synthetic 4-row fixture) | **No** — no real dataset exists yet |
| Isolation Forest / Autoencoder (`src/detection/`) | Done, unit-tested | **No** — never fit on the real dataset |
| SHAP explainer (`src/explainability/`) | Done, unit-tested | **No** — nothing to explain without a trained detector |
| Semantic classifier + LoRA training loop (`src/classification/`) | Done, unit-tested (mocked) | **No** — no fine-tuning run has happened |
| WHO ICD-11 client + cache (`src/icd11/`) | Done, unit-tested | Cache is empty — never run against the real container |
| Narrative generator (`src/narrative/`) | Done, unit-tested (mocked HF/OpenAI calls) | Never called against a real HF/OpenAI endpoint |
| SQLite repository + schema (`src/reporting/`, `src/db/`) | Done, unit-tested | DB never seeded/run against real claims |
| Feedback repository + retrainer (`src/feedback/`) | Done, unit-tested (mocked) | Never exercised with real auditor decisions |
| Dashboard (`dashboard/app.py`) | Written, review-only (see gap below) | Never run end-to-end against real data/models |
| Hosting (`docs/hosting_guide.md`) | Guide written, verified against provider docs | Not deployed |

**Bottom line**: every module has a real (non-stub) implementation and a passing test
suite (58 tests, all environments — pandas/sklearn/torch/transformers/peft/shap/
streamlit all actually installed and importable, not just mocked out at the dependency
level) that checks *this project's logic* with mocks standing in for GPUs, model
downloads, and live external APIs. None of it has been run against real data or produced
a real trained artifact yet. That's the entire remaining scope, and it's exactly Sprints
1–8 of the Product Backlog (Section 3.4.1) — nothing here requires new code design, only
running the pipeline and capturing evidence as it happens.

**Two real gaps found and handled while verifying this:**
1. **Fixed** — the classifier's default base model in `SemanticConsistencyClassifier.__init__`
   and `train.py`'s CLI did not match the primary model declared in
   `config/config.example.yaml`, so a run without an explicit `--base-model` would have
   silently fine-tuned the fallback. All three now name `emilyalsentzer/Bio_ClinicalBERT`
   (changed from AfroXLM-R on 2026-10-08, when English-Kiswahili code-switching was
   dropped from scope). There's still no automatic fallback-on-error, so pass
   `--base-model bert-base-uncased` explicitly if Bio_ClinicalBERT ever fails to load.
2. **Documented, not yet fixed** — `dashboard/app.py` never actually calls
   `SemanticConsistencyClassifier.predict()`. It's a review-only UI: it reads
   already-created `AuditReport` rows and calls the classifier's static `load_metrics()`
   only. The orchestration script that actually runs a claim through
   ingestion → detection → classifier → SHAP → narrative → `AuditReportGenerator.
   assemble_and_save()` doesn't exist yet as a module. This is squarely Sprint 7 work
   (§5 below), not a bug in what exists — but worth knowing the dashboard alone won't
   produce a single report until that orchestration script is written.

## 2. What actually gets trained, and where

| Component | Trainable? | Compute need | Where it runs | Why |
|---|---|---|---|---|
| Isolation Forest | Yes (`.fit()`) | CPU only, seconds | Colab **or** locally | sklearn, trivial cost either way |
| Autoencoder | Yes (`.fit()`) | CPU-feasible (small tabular net), faster with a GPU | Colab **or** locally | Small feed-forward net; no discrete GPU locally (`docs/environment.md`), but not GPU-*required* like the classifier is |
| Semantic Consistency Classifier (LoRA/PEFT on Bio_ClinicalBERT) | Yes, 5 seeded fine-tuning runs | GPU required for practical runtime | **Colab only** (free T4) | This machine has no discrete GPU/CUDA/ROCm — confirmed in `docs/environment.md`; exactly the compute constraint Section 1.7 cites for choosing LoRA over full fine-tuning in the first place |
| LLM narrative generator | **Not trained at all** | N/A | Wherever the dashboard runs (calls a hosted HF or OpenAI endpoint) | Explanation-only, prompted, not fine-tuned — this is the architecture pivot documented in Chapter 3/4 |
| ClassifierRetrainer (feedback loop, FR-13/14) | Re-runs the classifier training step on accumulated feedback | Same as classifier | Wherever it's triggered from (see §4) | Reuses `src/classification/train.py`, just on the Feedback Dataset instead of the original split |

**Practical recommendation**: run Isolation Forest and Autoencoder training in the *same*
Colab notebook as the classifier, right after data prep, even though they don't strictly
need a GPU. One notebook per pipeline stage means one place where `experiments/<run_id>/`
gets written and one `docs/environment.md` capture per session — simpler evidence trail
than splitting local vs. Colab runs. This is a preference, not a requirement; running the
two detectors locally is equally valid and slightly faster to iterate on.

## 3. Where things are stored

Three distinct storage locations, each with a different lifetime and a different reason
for existing — don't conflate them:

### 3a. Google Drive — the durable store during development (DR-01–DR-08)

Everything written by training, in the layout `implementation_roadmap.md` Section 3
already specifies:

```
MyDrive/SHA_Fraud_Audit/
├── data/{raw,processed}/        # DR-03 — synthetic dataset + splits
├── models/{isolation_forest,autoencoder,classifier_lora/{active,candidates}}/
├── artifacts/{feedback_dataset,environment_logs}/
├── experiments/<run_id>/{config.json,metrics.json}
└── db/sha_fraud.sqlite          # IR-04, the one genuinely "live" file here
```

This is the source of truth while the project is in development. `config.yaml`'s
`paths:` block already points here (`config/config.example.yaml`) — nothing to change,
just populate it by actually running training.

### 3b. Hugging Face Hub — the hand-off point to deployment

Drive isn't reachable from wherever the dashboard is actually deployed (a Cloudflare
Tunnel target machine is your own machine, so *technically* it could read Drive via
`rclone`/similar — but an HF Space definitely can't mount Drive). Rather than build two
different loading paths, both deployment options load the classifier from a single
private HF Hub model repo:

```
Colab (train) → Drive (durable store) → upload_folder() the *active* LoRA adapter → 
                                          private HF Hub repo → deployed dashboard 
                                          pulls from HF Hub via HF_TOKEN
```

Already documented with the exact `upload_folder()` call in `docs/hosting_guide.md`
("How training, storage, and deployment fit together"). Run this upload step once after
each `ClassifierRetrainer.evaluate_and_promote()` call that actually promotes a new
checkpoint (FR-14) — not after every training run, only after a promotion.

The ICD-11 cache JSON (`artifacts/icd11_cache.json`) travels differently: it's small
enough to commit directly into whatever repo the deployed app runs from, rather than
routing it through HF Hub.

### 3c. The deployed dashboard's own storage — the part that needs a decision

This is the one gap the hosting guide doesn't resolve, and it matters because of the
feedback loop (FR-12/13/14): `AuditReportRepository`/`FeedbackRepository` write to a
**local SQLite file and a local JSONL file** wherever the dashboard process runs. Whether
that data survives depends entirely on which hosting option is chosen:

| Deployment option | Where SQLite/feedback files live | Survives a restart? |
|---|---|---|
| **Cloudflare Tunnel** (your own machine) | Your machine's normal disk | Yes — ordinary local disk, no special handling needed |
| **HF Spaces, free tier** | The Space's container filesystem | **No** — confirmed current (2026) HF docs: free-tier Spaces get 50GB of *ephemeral* disk that resets on every restart, sleep, or rebuild; persistence requires the paid Persistent Storage add-on ($5/mo+) or an external store (Supabase/Neon/S3/a private HF Dataset repo) |
| **HF Spaces, with Persistent Storage add-on** | `/data` mount | Yes |

**This is a real architectural decision, not a detail to gloss over**: if Option B (HF
Spaces + Cloudflare Worker) is used on the free tier, every audit decision and every
feedback example is wiped the next time the Space restarts or sleeps from inactivity —
which directly breaks FR-12/13/14's entire premise (feedback accumulating toward a
retrain trigger). Three ways to resolve it, in order of how little they change:
1. Use **Option A (Cloudflare Tunnel)** for anything beyond a short demo — sidesteps the
   problem entirely since it's your own machine's disk.
2. If Option B's always-on property is worth it, pay for the **Persistent Storage
   add-on** and mount the DB/feedback files under `/data`.
3. Point `config.yaml`'s `db_path` at an external database instead of local SQLite (a
   bigger change — `AuditReportRepository` would need a different DB driver — genuinely
   not worth it for a thesis-scale proof of concept).

Record whichever is chosen in `docs/environment.md`'s "Deployment / demonstration
environment" section (still marked "not yet decided" there) — that's also what
`research_writing.txt` line 994 requires Chapter 5 to state accurately.

Sources verified 2026-09-28: [Disk usage on Spaces](https://huggingface.co/docs/hub/en/spaces-storage), [Storage limits](https://huggingface.co/docs/hub/en/storage-limits).

## 4. How the whole thing runs, end to end

```
                     ┌─────────────── DEVELOPMENT (Colab) ───────────────┐
                     │                                                    │
  data/raw (base)  ──┼─▶ noise_injection.py ─▶ preprocessing.py ─▶ data/processed/
                     │      (Table 3.1 rules)     (encode+split)   (train/val/test/
                     │                                               deployment pool)
                     │                                                    │
                     │   ┌────────────────────────────────────────────────┘
                     │   ▼
                     │  IsolationForestDetector.fit() ─┐
                     │  AutoencoderDetector.fit()      ├─▶ models/ (Drive)
                     │  SHAPExplainer (post-hoc, no training) 
                     │                                 │
                     │  train_all_seeds() (5 seeds) ───┘─▶ experiments/<run_id>/
                     │       │                              {config,metrics}.json
                     │       ▼                              (Drive)
                     │  best/active LoRA adapter ──▶ models/classifier_lora/active/ (Drive)
                     │                                       │
                     └───────────────────────────────────────┼──────────────────────┘
                                                              │ upload_folder()
                                                              ▼
                                              Private Hugging Face Hub model repo
                                                              │
                     ┌────────────────────────────────────────┼──────────────────────┐
                     │              DEPLOYMENT (wherever the dashboard runs)         │
                     │                                        ▼                      │
                     │   dashboard/app.py at startup:                                │
                     │     - loads config.yaml                                       │
                     │     - SemanticConsistencyClassifier pulls adapter from HF Hub │
                     │     - ICD11Cache loads artifacts/icd11_cache.json (bundled    │
                     │       with the deployed app's own repo, not from Drive/HF)    │
                     │     - AuditReportRepository opens local SQLite (see §3c for   │
                     │       the persistence caveat)                                 │
                     │                                                              │
                     │   Runtime flow per claim:                                    │
                     │     ingestion → preprocessing → IsolationForest/Autoencoder  │
                     │     score → (if anomalous) SemanticConsistencyClassifier     │
                     │     verdict → (if inconsistent) SHAP explain +               │
                     │     AuditNarrativeGenerator (calls HF Inference/OpenAI live) │
                     │     → AuditReportGenerator.assemble_and_save() → SQLite      │
                     │                                                              │
                     │   Auditor reviews in dashboard → decision recorded →         │
                     │   FeedbackRepository.materialize_example() (FR-12) →         │
                     │   at retrain_trigger_count (FR-13), ClassifierRetrainer runs │
                     │   → evaluate_and_promote() (FR-14) → if promoted, re-upload  │
                     │   the new active adapter to HF Hub (loop closes back to      │
                     │   step above — this retraining step itself needs a GPU, so   │
                     │   it either runs back on Colab manually, or (if ever          │
                     │   automated) on whatever compute triggers it — CPU-only      │
                     │   LoRA retraining on a live server is not realistic at this  │
                     │   project's scale, so treat retraining as a manual,          │
                     │   Colab-triggered step for the thesis timeline, not a fully  │
                     │   automatic production pipeline)                             │
                     └──────────────────────────────────────────────────────────────┘
```

The one honest simplification here: FR-13/14 describe retraining as config-triggered,
but *where* that retraining compute comes from isn't automatic — the deployed dashboard
itself has no GPU. For this project's scope, the realistic loop is: dashboard flags
"retrain threshold reached," you notice it (or check periodically), pull the
accumulated feedback JSONL back into a Colab notebook, run `ClassifierRetrainer` there,
and push the promoted checkpoint back to HF Hub. Document it this way in Chapter 5 rather
than implying a fully unattended production retraining pipeline — that would overstate
what's actually running, which is exactly the kind of claim `research_writing.txt` line
994 warns against.

## 5. Remaining work, in order (maps to Product Backlog Sprints, Section 3.4.2)

> **Superseded for scheduling (2026-10-08):** dates and sprint numbers now live in
> `docs/sprint_plan.md`, which fits the work into weekly sprints ending in the first week of
> November. The task descriptions below still apply; their "Sprint N" labels do not.

- [ ] **Sprint 1 — Data preparation.** Pick a public base claims dataset (not yet chosen),
      write an orchestration script that runs it through `noise_injection.py`'s Table 3.1
      functions at documented rates, produces the ~20,000-claim set, and calls
      `ClaimsPreprocessor.split_dataset()` for the 60/20/20 split (+ DR-08's disjoint
      Deployment/Demonstration Pool). Persist under Drive's `data/processed/`.
- [ ] **Sprint 2 — Baseline anomaly detection.** Fit Isolation Forest and Autoencoder on
      the real training split; run `tune_contamination()` / `tune_threshold()` against
      the validation split; record chosen `anomaly_score_threshold` into `config.yaml`
      (not committed — record the *value* in `docs/environment.md` or a run log instead).
- [ ] **Sprint 3 — WHO ICD-11 integration.** Bring up `docker compose up -d` once, run
      `src/icd11/refresh_cache.py` against every ICD-11 code the real dataset uses, commit
      the resulting `artifacts/icd11_cache.json` into the app repo, tear the container
      back down.
- [ ] **Sprint 4 — Classifier fine-tuning.** On Colab (GPU), run `train_all_seeds()` (5
      seeds per `config.yaml`), capture `experiments/<run_id>/` evidence, pick the active
      checkpoint, upload it to the private HF Hub repo.
- [ ] **Sprint 5 — SHAP integration.** Run `SHAPExplainer` against the *trained* Isolation
      Forest/Autoencoder (this is currently only tested against a mock/stub estimator).
- [ ] **Sprint 6 — LLM narrative layer.** Pick and confirm the real HF or OpenAI
      provider/model (`config.yaml`'s `llm_provider`/`llm_model_name`, currently `null`),
      run `AuditNarrativeGenerator.generate()` against a handful of real
      inconsistent-verdict claims, sanity-check output quality manually before automating.
- [ ] **Sprint 7 — Orchestration + dashboard + persistence end-to-end.** Write the
      still-missing orchestration script (§1's second flagged gap) that actually runs a
      claim through ingestion → preprocessing → IsolationForest/Autoencoder →
      SemanticConsistencyClassifier.predict() → SHAPExplainer → (if inconsistent)
      AuditNarrativeGenerator → `AuditReportGenerator.assemble_and_save()` — this is what
      populates the `AuditReport` rows the dashboard reviews; nothing produces them today.
      Then seed the real SQLite DB (`src/db/seed.py`), run
      `streamlit run dashboard/app.py` against the trained models and real ICD-11 cache,
      walk through the auditor/admin flows manually, confirm
      `FeedbackRepository`/`ClassifierRetrainer` fire correctly on real decisions.
- [ ] **Sprint 8 — Quantitative + qualitative evaluation.** Run the rule-based baseline +
      structured-only ablation, McNemar's test/bootstrap CIs, 5-seed mean±std, clean-vs-noisy-text
      and rare-ICD-11-code stratified results, single-rater rubric scoring + automated faithfulness
      check on narratives, cost-weighted economic framing — all per
      `chapter3_rigor_improvements_addition.txt`'s already-merged methodology.
- [ ] **Deployment.** Pick Option A or B from `docs/hosting_guide.md`, resolve the §3c
      persistence decision above, deploy, record the real tested setup in
      `docs/environment.md`.
- [ ] **Throughout**: keep `tests/test_plan.md` (Table 5.3) and `docs/ai_use_log.md`
      current as each of the above actually happens — not reconstructed afterward.

## 6. Questions to anticipate from your supervisor

Grouped by theme, with where the answer already lives (or where it's a genuine open gap
you should have a real answer ready for, not just point at a document):

**Modeling choices**
- *Why LoRA/PEFT instead of full fine-tuning?* — Section 1.7's compute-constraint
  argument: no discrete GPU locally, free Colab T4 tier only. Already documented.
- *Why Bio_ClinicalBERT, not a general-purpose encoder or a larger LLM?* — Sections
  2.2.6 and 3.2.3: it is pre-trained on clinical notes (Alsentzer et al., 2019), the study
  is delimited to English-language clinical text, and its size keeps LoRA fine-tuning
  within the free Colab T4 budget. Earlier drafts used AfroXLM-R for English-Kiswahili
  code-switching; that was dropped with supervisor approval on 2026-10-08.
- *How do you know the classifier learns semantic consistency, not a shortcut (e.g.
  claim amount, provider ID)?* — this is exactly what the ablation baselines and the
  faithfulness-check paragraph (Chapter 3, rigor patch) exist to answer. Be ready to
  explain the structured-only ablation's *purpose*, not just that it exists.
- *If there's no real SHA data, how is "consistent/inconsistent" ground truth defined?*
  — by construction: a claim is labeled `inconsistent` exactly when a Table 3.1 injection
  function altered it (upcoding, ghost billing, etc.); everything else is `consistent`.
  Say this plainly — it's a real limitation (labels come from your own injection rules,
  not independent expert judgment), not something to gloss over.
- *Why filter with unsupervised anomaly detection first instead of running the
  classifier on every claim?* — cost/throughput funnel: the classifier (and especially
  the LLM) is far more expensive per claim than Isolation Forest/Autoencoder scoring, and
  the architecture assumes SHA's real volume makes that funnel necessary. If asked for a
  number, you don't have a real-volume benchmark yet — say so rather than inventing one.

**Known architectural gap, be ready for this one specifically**
- *What happens to a fraudulent claim the anomaly detectors don't flag?* — currently
  nothing: the classifier only ever sees claims already flagged anomalous, so a false
  negative at the anomaly-detection stage never reaches the classifier at all. There's no
  second path. This is a real, currently-undocumented limitation worth naming yourself
  before it's asked — better than being caught not having considered it.

**Feedback loop / retraining**
- *How do you stop the feedback loop from reinforcing an auditor's own bias, or drifting
  the model somewhere worse?* — FR-14's promotion gate only helps with the *silent
  regression* case (never promotes a checkpoint scoring worse on the fixed held-out test
  set). It does **not** address an auditor being systematically biased in one direction —
  that's a genuine open limitation, not something the current design solves. Say that
  directly if asked, rather than overclaiming what the promotion gate covers.
- *Is retraining actually automatic, or does a person do it?* — be honest per §4 above:
  a person manually triggers it on Colab once the feedback count crosses the threshold;
  there's no unattended production pipeline. Framing it as fully automatic would be an
  overclaim `research_writing.txt` line 994 explicitly warns against.

**Validity / generalization**
- *How confident are you the synthetic fraud patterns generalize to real SHA fraud?* —
  Section 3.9 (Threats to Validity) already names this as an external-validity limitation;
  have the specific documented-typology sources (Kilei 2025, Kizito 2026) ready, since
  that's the traceability chain the synthetic injection is grounded in.
- *Why can't you just use real SHA data?* — have the specific access/availability
  constraint stated plainly (data isn't available to you as a student researcher), not a
  vague "for privacy reasons."

**Novelty**
- *Why not just have a large LLM do detection and explanation both?* — the core Chapter
  1/2 novelty argument: a fine-tuned classifier is auditable, cheaper per claim, and
  faster at SHA's claimed volume than an LLM-as-detector; the LLM is deliberately
  demoted to explanation-only. Have this contrast ready in one sentence, not a full
  re-derivation.

**Practical/demo**
- *What's your fallback if the HF Hub or LLM API is unreachable during your defense
  demo?* — not currently documented anywhere. Worth deciding and writing down before
  the defense, not improvising live: e.g., a cached example set of pre-generated
  narratives for the walkthrough if the live call fails.
- *Given no dataset exists yet, what fraction of your remaining timeline is dataset
  generation versus everything else?* — a real scheduling risk, not just a
  documentation gap: Sprints 2–8 in §5 above all depend on Sprint 1 finishing first.
  Have an honest estimate ready.
