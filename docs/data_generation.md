# Synthetic Claims Dataset: How It Is Generated

How ClaimGuard's training and evaluation data is produced (Section 3.2.1), what each
column means, and what the current version (`v0.4-5k`) contains. Real SHA claims cannot be
used, so the dataset is synthetic: realistic Kenyan claims built from SHA's published
tariffs, with clinical notes written by an open LLM and the six fraud typologies of
Table 3.1 injected under known labels.

## 1. Overview

```
 sha_catalogue.yaml ──► Stage A: generator.py ──► base_claims.parquet, providers.parquet,
 (37 ICD-11 codes,       structured claims +        note_requests.jsonl
  SHA tariffs)           provider-level fraud              │
                                                           ▼
                        Stage B: note_synthesis.py ──► notes.jsonl (one note per claim)
                        Qwen2.5-7B on a Colab T4        boilerplate_notes.jsonl (300)
                                                           │
                                                           ▼
                        Stage C: build_dataset.py ──► train / validation / test /
                        claim-level fraud, labels,      deployment_pool .parquet
                        splits, text noise              + manifest.json
```

| Stage | Module | Runs on | Time (5,000 claims) |
|---|---|---|---|
| A. Structured claims | `src/data/generator.py` | CPU | seconds |
| B. Clinical notes | `src/data/note_synthesis.py` | GPU (Colab T4) | ~11 min, plus ~6 min to load the model |
| C. Fraud, labels, splits | `src/data/build_dataset.py` | CPU | seconds |

All three run from `notebooks/01_generate_dataset.ipynb`. Each stage reads only the
previous stage's files, so any stage can be rerun on its own.

## 2. The catalogue: what can be claimed and what it pays

`src/data/sha_catalogue.yaml` lists the 37 conditions the generator draws from. Each entry
gives:

- **ICD-11 code and title**, checked against the WHO ICD-11 API (MMS release 2026-01).
  Only stem codes are used, never post-coordinated clusters.
- **Benefit package and payment mechanism**, taken from Legal Notice No. 56 of 2025
  (*Tariffs for Healthcare Services*, Social Health Insurance Act 2023):

| Package | Payment | Amount per claim | Codes |
|---|---|---|---|
| Outpatient (Primary Healthcare Fund) | capitation | KES 0: paid as KES 900 per person per year, not per visit | 10 |
| Medical inpatient (SHIF) | per diem | L3 2,240 · L4 3,360 · L5 3,920 · L6 4,480 per day | 11 |
| Maternity | case rate | normal delivery 10,000 · caesarean section 30,000 | 3 |
| Surgical (Annex 3) | case rate | 14,000 to 280,000 | 14 |

  Surgical tariffs cite their Annex 3 row number. The gazette text was OCR'd, so every
  row used was checked against the scanned page.
- **Who gets it:** facility levels that offer it, length-of-stay range, age range, sex,
  and a relative frequency weight.
- **Fraud hooks:** `outpatient_only` marks conditions that never justify admission;
  `upcode_to` names a higher-tariff code in the same clinical area.

## 3. Stage A: structured claims

`python -m src.data.generator --n-claims 5000 --seed 42 --out <raw dir>`

1. **Providers:** 60 facilities across Kenya's levels 2 to 6 (30% L2, 30% L3, 25% L4,
   12% L5, 3% L6), with at least two at every level. Higher levels see more claims (an L6
   hospital sees 8× a dispensary's volume).
2. **Claims:** each claim draws a condition by catalogue weight, then a facility that
   offers it, a patient within the condition's age/sex range, a length of stay and a
   service date between 1 April and 30 September 2025. The amount follows the payment
   rule above. Members come from a shared pool of households, so the same member appears
   on several claims, as in real data. Patients under 18, and about 35% of adults, are
   dependents.
3. **Provider-level fraud** (Table 3.1 rows 4 and 6). These typologies only show up as
   patterns across many claims, so they are built here rather than injected later:
   - **Inflated caesarean rate:** some facilities able to perform caesareans get extra
     caesarean claims until their caesarean share sits between 50% and 70% (honest
     facilities in v0.4 peak at about 40%). A facility that can't reach that range
     without exceeding the claim budget is skipped, so no flagged provider looks normal.
   - **Anomalous dependents:** a facility attaches 10 to 30 fictitious dependents to one
     real household it already bills for (realistic maximum: 6) and bills one claim per
     dependent.

   Only the excess claims are labelled fraud. The household's real claims, and the
   facility's genuine caesareans, stay unlabelled.
4. Writes `note_requests.jsonl`: one request per claim (diagnosis title, procedure,
   package, age, sex, facility type, length of stay). The ICD-11 code is never included.

## 4. Stage B: clinical notes

`python -m src.data.note_synthesis --backend vllm --seed 42 --requests ... --out ... --boilerplate-out ...`

**The LLM only writes data here. It is not the detector.** Detection is the fine-tuned
classifier's job; this stage gives that classifier realistic text to learn from.

- **Model:** Qwen2.5-7B-Instruct (AWQ 4-bit) served by vLLM 0.19.0 on a Colab T4, in
  float16 since the T4 has no bfloat16. Fallback: `--backend transformers` with
  Qwen2.5-3B-Instruct.
- **Prompt:** the model plays a clinician at a Kenyan facility writing a claim
  justification for SHA. Each claim gets, by a seeded draw, one of five styles (terse
  shorthand, narrative, SOAP, discharge summary, two or three sentences) and one of four
  detail focuses (exam findings, an investigation result, the main treatment, the
  presenting complaint). The prompt states the setting: seen as an outpatient, admitted
  for *n* days, admitted for delivery, or admitted for a procedure. The style mix is what
  gives the corpus varied registers.
- **Checks:** a note is rejected and regenerated (up to 3 attempts) if it has fewer than
  5 or more than 150 words, or contains an ICD-11 code. A note that still fails keeps a
  `rejected_reason` and is dropped in Stage C.
- **Boilerplate notes:** 300 generic notes for the record-fabrication typology ("Patient
  reviewed and managed as per protocol. Stable for discharge."). A strict filter rejects
  any that mention a symptom, body part, finding, test, diagnosis, drug or procedure, any
  digits or markup, or that fall outside 5 to 40 words.
- **Reproducible and resumable:** each claim's sampling seed is derived from its claim
  ID. Notes are appended in batches of 500, so a disconnected runtime resumes where it
  stopped. `notes.jsonl.meta.json` records the model and settings.

## 5. Stage C: fraud injection, labels, splits, noise

`python -m src.data.build_dataset --raw <raw dir> --out <processed dir> --seed 42`

### 5.1 Claim-level fraud (Table 3.1 rows 1, 2, 3, 5)

Each typology targets the same number of claims: 5,000 × 10% ÷ 6 ≈ 83. Each claim gets at
most one typology.

| Typology | What changes | Eligible claims |
|---|---|---|
| Upcoding | Billed code and amount switched to the `upcode_to` code; the note still describes the original. `original_icd11_code` keeps the truth. | Codes with an upcode target, at facilities that offer the target |
| Outpatient → inpatient conversion | An outpatient-only visit becomes an admission billed per diem × stay; the note still describes an outpatient visit. | Outpatient-only codes at L3 or above |
| Ghost billing | The note is replaced by another patient's note from a different ICD-11 chapter, so it describes unrelated care. | Any claim |
| Record fabrication | The note is replaced by a boilerplate note. | Any claim |

The upcoding pairs are:

| Original code | Upcoded to |
|---|---|
| Normal delivery (KES 10,000) | Emergency caesarean section (KES 30,000) |
| Anal abscess (14,000) | Anal fistula (44,800) |
| Uterine leiomyoma (89,600) | Cervical carcinoma (246,400) |
| Prostatic hyperplasia (168,000) | Prostate adenocarcinoma (280,000) |

### 5.2 Labels

| Column | Meaning |
|---|---|
| `is_fraud` | 1 if any typology applies |
| `fraud_typology` | `none` or one of the six typology names |
| `label` | Classifier target: `inconsistent` when the note no longer matches the billed code (upcoding, ghost billing, fabrication), `consistent` otherwise |
| `is_upcoded`, `is_converted_admission`, `is_ghost_billed`, `is_inflated_csection`, `is_fabricated_record`, `is_anomalous_dependent` | One flag per typology |

Conversion and the two provider-level typologies are fraud with **consistent** notes. The
note matches the code; the fraud is in the admission or in claim volumes, so the anomaly
detector has to catch them, not the text classifier. This is the hybrid design's
rationale: each half covers fraud the other cannot see.

### 5.3 Splits

1. A 10% **Deployment/Demonstration Pool** (DR-08) is carved off first. It feeds the
   dashboard and the feedback loop and never touches reported metrics.
2. The rest is split 60/20/20 into train, validation and test.
3. Both steps are stratified on `fraud_typology`, so every typology appears in every
   split.

### 5.4 Unseen inputs in the test split (DR-03)

The test split and pool must contain inputs the model never saw in training:

- **Fabrication notes:** the boilerplate pool is split in half. Train and validation use
  one half; test and pool are redrawn from the other.
- **Shorthand:** text noise in test and pool uses a different glossary (`O/E`, `C/O`,
  `Hx.`, `D/C'd`…) from the one used in training (`pt`, `c/o`, `hx`, `dx`…).

### 5.5 Text noise

A random 30% of claims in every split get abbreviations, single-letter typos (swapped,
dropped or doubled letters, words of four or more letters), and inconsistent casing and
spacing, each applied with probability 0.2 per word or phrase. `has_text_noise` marks
them, so evaluation can report clean and noisy claims separately (NFR-05).

### 5.6 Structured-field noise (from v0.5)

A further random 10% of claims in every split, chosen independently of the text noise,
get one data-quality defect each, marked by `has_structured_noise`:

| Defect | Example |
|---|---|
| Missing value | `patient_age`, `patient_sex` or `discharge_date` left blank |
| Malformed code | `ca40.z`, `CA40Z`, ` CA40.Z ` |
| Numeric perturbation | patient age off by 1 or 2 years |

`claim_amount` and the fraud labels are never changed. SHA pays fixed tariffs, so a
perturbed amount would look like overbilling and turn a clean claim into a fraud-looking
one.

The pipeline handles these defects, as Section 3.2.2's first two steps require:
- `ClaimsPreprocessor.align_schema` repairs malformed codes, since each has one correct
  reading.
- `validate_quality` flags missing fields (`quality_flag_missing_field`) and codes that
  are still invalid (`quality_flag_invalid_code`).
- `encode_features` fills blanks from the training split (median age, the code's median
  stay) and counts them in the `has_missing_field` feature, so a defective claim is
  still scored.

Text noise is deliberately **not** cleaned: robustness to messy notes is what the
classifier is evaluated on. A cleaner would have to hard-code the generator's own
abbreviation list, which is circular.

Anything that reads `icd11_code` straight from the parquet files (for example the
classifier export) must pass it through `repair_icd11_code` first.

## 6. Outputs and where they live

Generated data is not in git (`/data/` is gitignored). The notebook writes to Google Drive:

```
MyDrive/claimguard/data/
├── raw/<version>/        base_claims.parquet, providers.parquet, note_requests.jsonl,
│                         generator_config.json, notes.jsonl (+ .meta.json),
│                         boilerplate_notes.jsonl
└── processed/<version>/  train.parquet, validation.parquet, test.parquet,
                          deployment_pool.parquet, manifest.json
```

`manifest.json` records the git commit, the seed, the SHA-256 hash of every input file,
rejection counts, and per-split counts. Together with the commit and seed it identifies
the exact dataset. The notebook also saves `pip freeze` to
`MyDrive/claimguard/artifacts/environment_logs/`.

## 7. Version v0.4-5k (generated 2026-10-11)

Generated with seed 42 on a Tesla T4 (15 GB), Python 3.13. The exact commit is in
`manifest.json` (`git_commit`, `colab_commit`).

- 5,000 notes written, **0 rejected** after regeneration. Boilerplate: 300 generated,
  1 rejected.

| Split | Claims | Fraud | Inconsistent | With text noise |
|---|---|---|---|---|
| train | 2,700 | 268 (9.9%) | 134 | 810 |
| validation | 900 | 89 | 45 | 270 |
| test | 900 | 90 | 45 | 270 |
| deployment pool | 500 | 49 | 25 | 150 |

Every typology has 44–45 claims in train, 14–15 in validation and test, and 8–9 in the
pool.

### v0.5-5k (built 2026-10-12)

Stage C rerun locally from v0.4's raw files at commit `1420fb1`, adding the structured-field
noise (§5.6). The claims, labels, amounts and notes are identical to v0.4; the only change
is 270 / 90 / 90 / 50 claims with a structured defect in train / validation / test / pool.
All results up to 17 Oct (TC-022 to TC-031) are on v0.5-5k.

### v0.6-20k (generated 2026-10-18)

The full-size dataset planned in Section 3.2.1 (#27). All three stages were run on Colab
at commit `e565b00` (`manifest.json`: `git_commit`, `colab_commit`), seed 42, on a Tesla
T4 (15 GB), Python 3.13.15. The parameters that changed from v0.4/v0.5 are
`N_CLAIMS = 20000` and `N_PROVIDERS = 240`, so the provider-level typologies keep the same
volume per provider. The generator, catalogue and noise settings are otherwise unchanged,
apart from the two #51 changes below.

- 20,000 notes written by Qwen2.5 7B Instruct (AWQ) on vLLM, **0 rejected**; about 42
  minutes (40 batches of 500 at about 63 s each). Boilerplate: 300 generated, 1 rejected;
  149 for train/validation, 150 for test/pool.
- 333 claims per typology (exactly the per-typology target), 10.0% fraud in every split,
  splits disjoint, 38 distinct codes after `repair_icd11_code`.

| Split | Claims | Fraud | Inconsistent | With text noise | With a structured defect |
|---|---|---|---|---|---|
| train | 10,800 | 1,079 (10.0%) | 540 | 3,240 | 1,080 |
| validation | 3,600 | 359 | 179 | 1,080 | 360 |
| test | 3,600 | 360 | 180 | 1,080 | 360 |
| deployment pool | 2,000 | 200 | 100 | 600 | 200 |

Each typology has 179–180 claims in train, 59–60 in validation and test, and 33–34 in the
pool (v0.5: 44–45 and 14–15).

**What changed for #51, and what it did** (checked on the built splits, 18 Oct):

- Caesarean shorthand is now training noise. Of the claims billed as caesarean
  (JB22.0/JB22.1) whose notes are consistent, 5% in train and 6% in validation contain
  "CS" (LLM-written or injected); "C/S" appears in 6% of the test split's and 2% of the
  pool's, from the unseen glossary, against 0–1% in train and validation, where only the
  LLM writes it. "LSCS" does not occur: the notes never spell out "lower segment caesarean
  section".
- About one prompt in six (16% of the first 4,000, re-derived from `note_requests.jsonl`)
  uses the findings-only style, which does not name the diagnosis.
- The word itself is still a strong cue: 70% of consistent caesarean-billed training
  notes contain "caesarean", and none of the upcoded ones do (an upcoded note describes
  the vaginal delivery that took place, so this is genuine content, not an artefact).
  Whether the classifier still keys on the word can only be seen in its false positives
  on genuine caesarean notes without it. Check this in the v0.6 classifier run (#29) and
  record it as a test result.

**v0.6-20k is the current dataset.** The stage-1 thresholds in `config.example.yaml`
were tuned on v0.5 and must be re-tuned on v0.6 (`python -m src.detection.tune_stage1`).

## 8. Known limitations (to address in a later version)

1. **The boilerplate notes are near-duplicates.** Most are variations on "Patient
   reviewed and managed as per protocol. Stable for discharge." The classifier may learn
   to spot this one template rather than to recognise an unjustified claim, and the
   held-out half barely differs from the training half. Fix: vary length, setting and
   style in the prompt, and reject near-duplicates, not just exact copies.
2. **Notes often name the diagnosis outright** ("underwent a single spontaneous vertex
   delivery"). The classifier may succeed by matching diagnosis words against the code
   title rather than by clinical reasoning. Real notes do this too, but it makes the task
   easier than it is in practice. To be stated under Section 3.9 (construct validity), or
   reduced by having some notes describe findings only. **Evidence (TC-027):** the first
   classifier caught every delivery-to-caesarean upcode, and 3 of its 4 false positives were
   genuine caesarean notes written as "CS" or "C/S", so it appears to key on the word
   "caesarean". Upcoding in validation is 14 of 15 delivery pairs, so the surgical pairs
   are barely tested.
   **Fixed for v0.6 (#51):** training noise now abbreviates "caesarean section" to "CS"
   and "lower segment caesarean section" to "LSCS" in every split's notes, consistent and
   upcoded alike; the unseen test/pool glossary uses "C/S" instead (DR-03). About one
   prompt in six now asks for findings and procedure only, without naming the diagnosis.
3. **Upcoding is mostly deliveries** (about 83% normal delivery → caesarean), because
   deliveries dominate the eligible claims. This is realistic for SHA's case mix, but
   per-pair results will be thin for the surgical pairs.
4. **Text noise is heavy:** at 0.2 per word, a 15-word note gets 3–4 typos. Halving it
   would let shorthand dominate over typos.
5. **Size:** resolved in v0.6-20k (20,000 claims, 333 per typology). Validation and test
   still have only 59–60 claims per typology, so per-typology results keep wide intervals.
6. **SOAP slips:** some SOAP notes put treatment under "A:" (Assessment). This is minor
   and also happens in real notes.
7. **Some notes switch language (found 19 Oct, TC-033).** Qwen2.5 occasionally continues a
   note in Chinese: 156 of v0.6's 20,000 notes (0.8%; train 82, validation 30, test 24,
   pool 20) contain non-Latin script, and v0.4/v0.5 had 37 of 5,000. `validate_note` did
   not check language, so they were accepted. They are mostly clean claims, yet 3 of the
   v0.6 classifier's 5 validation false positives are among them. `validate_note` now
   rejects any non-Latin letter, so the next generation regenerates them; v0.6 is left
   as built, because its classifier is already trained and regenerating needs a GPU.
   Report it as a data-quality limitation, or, if v0.6 is ever rebuilt, rerun stage B
   for these claims only (the stage resumes per claim) and then stage C.

## 9. Rerunning

- **Full rebuild:** run the notebook top to bottom with a T4 runtime. Put
  `GITHUB_TOKEN=...` (a fine-grained, read-only token) in `MyDrive/claimguard/.env`.
- **After changing only Stage C** (noise, splits): rerun just the Stage C cell. The notes
  are reused.
- **New version:** bump `DATASET_VERSION` in the notebook's first cell whenever the
  generator, catalogue or parameters change, so earlier versions stay intact.
- **Tests:** `venv/bin/python -m pytest -q tests/test_dataset_pipeline.py` runs all three
  stages at small scale with a fake LLM (TC-016 to TC-021 in `tests/test_plan.md`).

## Live claims feed (`live-v1`, #64)

The dashboard's data pipeline (`docs/ingestion_plan.md`) needs claims that arrive over time
from a separate claims platform. They come from a fresh set generated with the same three
stages (`notebooks/06_generate_live_feed.ipynb`), not from the training data:

| Choice | Value | Why |
|---|---|---|
| Seed | 2026 | a new draw, not a re-sample of v0.6 |
| Claims | 2,400 | about 400 a month, so each monthly audit batch has enough claims for the provider-level features |
| Service dates | 2025-10-01 to 2026-03-31 | the six months after the training window: the feed is "future" data |
| Providers | v0.6's 240 (`--providers`) | SHA's provider population does not change between months; the provider-level features compare a provider's new claims with its history |
| Ids | `--id-prefix L`: `LCLM000001`, `LMBR000001` | no collision with v0.6 claims or members, so member histories stay separate |
| Build | `build_dataset --live` → `live.parquet` | no split; every claim is unseen input (all boilerplate notes are new; text noise uses the held-out glossary) |

The labels stay in `live.parquet` for offline evaluation. The claims platform strips them
before it serves a claim (`sha_cdp/seed.py`), so nothing in the audit system sees them.

A provider flagged for inflated caesareans in the live set bills its extra caesareans
against a history of normal deliveries from training, so its share is diluted compared
with a provider whose whole history is fraudulent. That is the realistic case (a provider
that turns fraudulent) and is a recorded limitation if the provider-level lane misses it.
