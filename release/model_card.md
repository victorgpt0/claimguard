---
license: mit
base_model: emilyalsentzer/Bio_ClinicalBERT
library_name: peft
language:
- en
pipeline_tag: text-classification
tags:
- lora
- peft
- clinical-text
- health-insurance
- fraud-audit
- icd-11
- synthetic-data
---

<!-- Sources for every number are given in HTML comments next to it. Paths starting with
     "Drive:" are under MyDrive/claimguard/ (the project's results store, not in git). -->

# ClaimGuard semantic consistency classifier (`evaluated_v1`)

A LoRA adapter for [Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT)
that reads a claim's free-text clinical justification together with the definition of the
ICD-11 code it was billed under, and says whether the two are **consistent** or
**inconsistent**. It is the text-checking stage of ClaimGuard, a post-payment audit
prototype built for an undergraduate project on medical insurance fraud at Kenya's Social
Health Authority (SHA).

This is the frozen checkpoint every reported result was produced with. Pin the revision
(see "How to load") rather than loading the latest commit.

- **Developed by:** Gichuru Victor Kimotho, Strathmore University (ICS Project II), supervised by Mr. Tiberius Tabulu
- **Model type:** sequence-pair classifier, LoRA adapter (PEFT) with a trained classification head
- **Language:** English
- **Licence:** MIT (the base model is also MIT)
- **Code:** the ClaimGuard repository (`src/classification/`) <!-- TODO(student): add the public repo URL if the repository is made public; it is private now. -->

## Intended use

Decision support for **post-payment audit**: it helps decide which already-paid claims a
human auditor should look at first, and points at the text evidence. In ClaimGuard it runs
only on claims a statistical first stage has already flagged, and an auditor reviews every
report it produces.

**An "inconsistent" verdict is not a finding of fraud.** Notes can disagree with a billed
code because of coding errors, poor documentation or an unusual but genuine case. No claim,
provider or member should be penalised, rejected or reported on the basis of this model's
output without a human review of the underlying records.

Out of scope: pre-payment claim adjudication, any automated decision about a person or
facility, languages other than English (including English-Swahili code-switched notes), and
coding systems other than ICD-11.

## Inputs and outputs

- **Input:** a text pair, `(clinical_notes, icd11_definition)`, tokenised together, truncated to 512 tokens. <!-- src/classification/classifier.py predict(); config.json hyperparameters.max_length -->
- **Output:** two logits for the labels `0 = consistent`, `1 = inconsistent`. <!-- src/classification/classifier.py LABELS -->

### The score is not a calibrated probability

ClaimGuard reports the softmax value of the predicted label as a "model score". No
calibration step (temperature scaling, isotonic regression or similar) was applied, and
the model was trained with a class-weighted loss, which shifts the scores. A score of 0.95
does **not** mean a 95% chance that the notes are inconsistent. Use the score to rank
claims, not as a probability. <!-- config.json hyperparameters.weighted_loss = true; no calibration code exists in src/ -->

## Training data

**Synthetic only. No real SHA, patient or provider data was used.** Real claims data could
not be obtained, so the dataset (`v0.6-20k`) was generated:

- 20,000 synthetic claims with structured billing fields. A 10% deployment/demonstration pool was set aside first and never used for reported metrics; the rest was split 60/20/20 into train, validation and test, stratified on fraud typology. <!-- docs/data_generation.md §5.3; split sizes in config.json: train_examples 10800, validation_examples 3600 -->
- Clinical notes written by Qwen2.5-7B-Instruct (AWQ 4-bit) on a Colab T4. <!-- docs/data_generation.md stage B -->
- Fraud injected according to six typologies modelled on documented SHA fraud patterns: anomalous dependants, ghost billing, inflated caesarean rates, outpatient-to-inpatient conversion, record fabrication and upcoding. Only some of these leave a trace in the notes (ghost billing, record fabrication, upcoding); the rest are visible only in the structured fields and are left to the anomaly detectors. <!-- Drive: experiments/evaluation/v0.6-20k_test_text_20261027T215556Z/tables.md, typology table (only these three have non-zero recall under the text label) -->
- Label: "inconsistent" when the notes do not support the billed code. Validation had 179 inconsistent pairs out of 3,600. <!-- Drive: models/classifier_lora/evaluated_v1/config.json metrics.positives, validation_examples -->
- Robustness noise: 30% of claims in every split get abbreviations, typos or spacing errors in the notes; the test split uses a different shorthand glossary and held-out fabrication boilerplate so it contains text the model never saw. <!-- docs/data_generation.md §5.4, §5.5 -->
- ICD-11 definitions come from the WHO ICD-11 API, cached. <!-- artifacts/icd11_cache.json; README export command -->

The dataset is not published with this model. <!-- TODO(student): decide whether to release the synthetic dataset (e.g. as an HF dataset repo) and link it here. -->

## Training procedure

| Setting | Value |
|---|---|
| Base model | `emilyalsentzer/Bio_ClinicalBERT` |
| Method | LoRA, rank 8, alpha 16, dropout 0.1, on `query` and `value`; classification head fully trained |
| Epochs | up to 5, early stopping with patience 2 (this checkpoint: epoch 5) |
| Batch size / learning rate | 16 / 2e-4 |
| Max sequence length | 512 |
| Loss | class-weighted cross-entropy |
| Seeds | 42, 123, 2024, 7, 31415; this checkpoint is **seed 7**, the best validation F1 |
| Hardware | Google Colab, Tesla T4 |
| Software | PEFT 0.21.1 |

<!-- Drive: models/classifier_lora/evaluated_v1/config.json (lora_config, hyperparameters, seed, run_id 20261018T181622_seed7, colab_commit 5c6fb3df24787e287a87ff606420d59e041cfab2), adapter_config.json (modules_to_save, peft_version); seeds and hardware: tests/test_plan.md TC-033 -->

The adapter was copied unchanged to `models/classifier_lora/active/` and frozen as
`evaluated_v1/` on 18–19 October 2026, before the test split was read. <!-- docs/sprint_plan.md; byte-identical, checked 2026-11-02 -->

## Evaluation

All figures are on synthetic data and say nothing directly about performance on real
claims.

### Validation (3,600 pairs, used for seed selection)

| | Precision | Recall | F1 |
|---|---|---|---|
| Five seeds, mean ± std | 0.960 ± 0.009 | 0.990 ± 0.006 | 0.975 ± 0.004 |
| Seed 7 (this checkpoint) | 0.973 | 0.989 | 0.981 |

<!-- mean ± std: Drive: experiments/classifier/v0.6-20k/seed_summary.json (n = 5, sample std); seed 7: Drive: models/classifier_lora/evaluated_v1/metrics.json -->

Seed 7 was chosen because it had the best validation F1, so its validation figures are
slightly optimistic; the five-seed mean is the fairer estimate. On validation it made 5
false positives out of 3,421 consistent pairs and 2 false negatives out of 179 inconsistent
ones. F1 was 0.985 on clean notes and 0.969 on noisy notes, and 0.988 on rare ICD-11 codes
and 0.979 on common ones. <!-- tests/test_plan.md TC-033 -->

### Held-out test (3,600 claims, read once)

The classifier was not scored on its own on the test split. It was scored as ClaimGuard's
**text lane**: a claim counts as flagged when the stage-1 text screen passes it to the
classifier and the classifier calls it inconsistent. A claim the screen did not pass is
counted as a miss, so the recall below is a lower bound for the classifier on its own.
Truth is the text label (180 inconsistent claims). Intervals are 95% percentile bootstrap,
1,000 resamples.

| System | Precision | Recall | F1 | FPR |
|---|---|---|---|---|
| Text lane with this classifier | 0.993 [0.977, 1.000] | 0.806 [0.745, 0.863] | 0.890 [0.853, 0.925] | 0.000 [0.000, 0.001] |
| Zero-shot LLM in the classifier's place | 0.916 [0.851, 0.970] | 0.422 [0.351, 0.497] | 0.578 [0.504, 0.648] | 0.002 [0.001, 0.004] |

AUC-ROC of the text lane: 0.885 [0.849, 0.920]. McNemar's exact test against the zero-shot
LLM: b = 100, c = 25, p = 8.1e-12.

<!-- Drive: experiments/evaluation/v0.6-20k_test_text_20261027T215556Z/tables.md ("Main metrics", "McNemar") and results.json; run recorded as TC-045 in tests/test_plan.md -->

| Stratum (text lane, F1 / recall / FPR) | | |
|---|---|---|
| Notes: clean vs noisy | 0.891 / 0.809 / 0.000 | 0.886 / 0.796 / 0.000 |
| Structured fields: clean vs noisy | 0.884 / 0.798 / 0.000 | 0.957 / 0.917 / 0.000 |
| ICD-11 codes: common vs rare (< 1% of training claims, 18 codes) | 0.870 / 0.770 / 0.000 | 0.969 / 0.969 / 0.003 |

By typology (recall): record fabrication 0.983, ghost billing 0.750, upcoding 0.683.

<!-- Drive: experiments/evaluation/v0.6-20k_test_text_20261027T215556Z/tables.md (strata and typology tables) -->

### Within the whole system

Against overall fraud labels (360 fraudulent of 3,600 test claims), the full ClaimGuard
pipeline scored precision 0.439, recall 0.775, F1 0.561 [0.524, 0.598]. Most of its false
positives come from the anomaly detectors, not from this classifier. A hybrid of this
text lane with five billing rules scored 0.844 / 0.808 / 0.826 [0.795, 0.857]; the
rule-based baseline alone scored 0.760 / 0.467 / 0.578. The hybrid was chosen among three
options on validation, and the rules encode the injected typologies, so part of its result
is by construction.

<!-- Drive: experiments/evaluation/v0.6-20k_test_is_fraud_20261027T215553Z/tables.md; caveats from tests/test_plan.md TC-045. The text lane's FPR is 0.000 under the text label (above), while the pipeline's is 0.110, so the false positives come from elsewhere in the pipeline. -->

## Limitations and risks

- **Synthetic data.** The notes were written by one LLM from templates of the injected typologies. Real notes are shorter, messier, multilingual and written by many people; expect much lower performance on real claims until the model is evaluated and retrained on them.
- **Easy task by construction.** Inconsistent notes were generated to disagree with the code, so the classes are better separated than they would be in practice. The high validation F1 reflects that.
- **Generator artefacts.** 0.8% of the dataset's notes were partly written in Chinese by the note model; 3 of the 5 validation false positives were such notes. The checkpoint was trained on this data as is. <!-- tests/test_plan.md TC-033 -->
- **Narrow fraud coverage.** It only sees fraud that shows up in the text. It cannot detect phantom dependants, inflated caesarean rates or most admission conversions.
- **English only.** English-Swahili code-switched notes were dropped from scope and were not tested.
- **Uncalibrated scores** (see above).
- **Bias.** Providers whose documentation style differs from the synthetic notes (for example small facilities, or clinicians writing in shorthand) may be flagged more often. This was not measured, because the data has no real provider characteristics.
- **Feedback loop.** ClaimGuard can retrain the classifier from auditor decisions; any retrained checkpoint is a different model and needs its own evaluation and card. This card describes `evaluated_v1` only.

## How to load

```python
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO = "<repo-id>"          # TODO(student): the Hub repo, e.g. <user>/claimguard-consistency-classifier
REVISION = "<commit-sha>"   # the commit printed by src/release/publish_model.py (tag v1.0-evaluated)

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
base = AutoModelForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", num_labels=2)
model = PeftModel.from_pretrained(base, REPO, revision=REVISION).eval()

inputs = tokenizer(clinical_notes, icd11_definition, truncation=True, max_length=512, return_tensors="pt")
with torch.no_grad():
    scores = torch.softmax(model(**inputs).logits, dim=-1)[0]
verdict = ["consistent", "inconsistent"][int(scores.argmax())]
```

`training_config.json` in this repository holds the full training configuration and
`metrics.json` the validation metrics of this checkpoint.

## Citation

<!-- TODO(student): add the final project report's citation once submitted. -->
Gichuru, V. K. (2026). *LLM-Assisted Anomaly Detection for Post-Payment Fraud Audit: A
Case Study of Kenya's Social Health Authority*. ICS Project II, Strathmore University.

Base model: Alsentzer, E. et al. (2019). Publicly Available Clinical BERT Embeddings.
*Proceedings of the 2nd Clinical Natural Language Processing Workshop*.
