# Test Plan

Table 5.3 format (`research_writing.txt` line 1098), maintained from the start of
implementation rather than reconstructed at write-up time. Add a row as each requirement
is implemented and tested — actual test code lives alongside this as `test_*.py` files;
this table is the traceable summary Chapter 5 Section 5.4.2 draws from directly.

| Test ID | Traceability | Scenario, data and conditions | Expected result | Actual result, evidence and verdict |
|---|---|---|---|---|
| TC-001 | FR-01 | *(not yet implemented)* | | |
| TC-002 | FR-02 / NFR-01 | | | |
| TC-003 | FR-03 | | | |
| TC-004 | FR-04 / NFR-05 | | | |
| TC-005 | FR-05 | | | |
| TC-006 | FR-06 / NFR-02 | Live lookup of `1F40.0`, `CA40`, `JB20` against the local whoicd/icd-api container (release 2026-01), 2026-10-11. | Official title, definition, inclusion and exclusion text returned from the local container. | **Fail, corrected.** (1) Every lookup failed with 401: the API returns `stemId` as an `https://id.who.int/...` URI, and the client followed it to WHO's cloud API. Fixed by re-pointing the URI path at `base_url`. (2) Inclusions came back empty and exclusions as URIs: items are `{"label": {...}}`. Fixed to read `label`. Unit test fixture rewritten from a captured live response; it had the wrong shape, which is why both bugs passed unit testing. Re-run: titles and criteria correct (e.g. CA40 Pneumonia, exclusion "Pneumonitis"). `tests/test_icd11_client.py`. **Pass.** |
| TC-007 | FR-07 | | | |
| TC-008 | FR-08 / NFR-03 | | | |
| TC-009 | FR-09 | | | |
| TC-010 | FR-10 | | | |
| TC-011 | FR-11 | | | |
| TC-012 | FR-12 | | | |
| TC-013 | FR-13 / IR-06 | | | |
| TC-014 | FR-14 / NFR-05 | | | |
| TC-015 | NFR-07 (usability) | | | |
| TC-016 | DR-03 / Table 3.1 | Generate 1,200 claims (stages A-C, fake LLM for notes); `tests/test_dataset_pipeline.py`. | Fraud prevalence 9-11%, all six typologies present, at most one typology per claim, `is_fraud` equals "any typology". | 2026-10-11: as expected. At 5,000 claims, 81-83 claims per typology, prevalence 9.92%. **Pass.** |
| TC-017 | Table 3.1 row 2 | Same run: conversion count at 5,000 claims. | ~83 converted admissions (the per-typology target). | **Fail, corrected.** Only 47: the rate was computed against all untouched claims, not the eligible outpatient pool at Level 3+. Eligibility mask added in `build_dataset.py`. Re-run: 83. **Pass.** |
| TC-018 | Table 3.1 row 4 | Caesarean share per provider, 5,000 claims. | Flagged providers clearly above the WHO 15% threshold and above honest peers. | **Fail, corrected.** One flagged provider sat at 14% (it got a partial top-up), and honest providers reached 45%. Providers that can't reach the inflated share are now skipped, and the range was raised to 50-70%. Re-run: flagged 54-69%, honest max 40%. **Pass.** |
| TC-019 | DR-03 / DR-08 | Split disjointness and unseen test inputs; `tests/test_dataset_pipeline.py`. | Train/validation/test/pool disjoint and covering every claim; unseen shorthand glossary and held-out fabrication notes only in test/pool. | 2026-10-11: as expected. **Pass.** |
| TC-020 | Section 3.2.1 (catalogue) | Every claim's facility level and amount checked against `sha_catalogue.yaml`; catalogue tariffs checked against scanned LN 56/2025 Annex 3 pages. | Capitation claims KES 0; case-rate claims equal the gazetted tariff; no service billed at a level that can't provide it. | 2026-10-11: as expected. All 14 surgical tariffs match the gazette scan. **Pass.** |
| TC-021 | Section 3.2.1 (notes) | Stage B with `--backend transformers`, Qwen2.5-0.5B on CPU, 3 claims + 5 boilerplate (smoke test of the code path). | Notes written, resumable file and metadata saved; boilerplate generic. | Clinical notes plausible. **Fail, corrected:** boilerplate notes named specific conditions (URTI, haematoma, MRI findings). Added `validate_boilerplate` (rejects clinical vocabulary, digits, markup, >40 words) with regeneration, and split the accepted pool between train and test. The full 7B run on Colab is still to be done and checked. |
| TC-022 | FR-02, Section 3.2.3 (Isolation Forest) | `python -m src.detection.evaluate_baseline --data data/processed/v0.4-5k`: fit on train (2,700), contamination tuned on validation (900) over [0.05, 0.40]. Test split not read. | Catches the structural typologies (conversion, anomalous dependents, inflated caesarean); misses ghost billing and record fabrication, which only show in the note text. | 2026-10-12: contamination 0.10; precision 0.47, recall 0.55, F1 0.51, ROC-AUC 0.79; 7% of clean claims flagged. Recall: conversion 15/15, dependents 15/15, upcoding 9/15, caesarean 8/14, ghost 1/15, fabrication 1/15. **Fail, corrected first:** the original `encode_features` one-hot encoded providers per split (train and validation got different columns) and computed code frequency within each split. Replaced with a fixed 14-feature set fit on train. Without the history features (member and provider aggregates) F1 was 0.25 with dependents and caesarean at 0%; with claim-level features only, 0.18. |
| TC-023 | Section 3.2.2 (structured noise), NFR-05 | Stage C rebuilt locally from the Colab raw files as `v0.5-5k` (10% of claims get one structured defect); Isolation Forest baseline rerun on v0.4 and v0.5 with seeds 1-5, 15 features, validation split only. | Same claims, labels and notes as v0.4; every defective claim still scored; no invalid codes after repair; F1 drops slightly with noise, ROC-AUC roughly unchanged. | 2026-10-12: as expected. Local rebuild identical to the Colab build (input hashes, claims, labels, amounts, all notes). Validation: 90 claims with structured noise, 31 missing-field flags, 0 invalid codes after repair. Seeds 1-5, mean (range): v0.4 precision 0.69 (0.60-0.76), recall 0.36, F1 0.48 (0.45-0.49), ROC-AUC 0.77; v0.5 precision 0.60 (0.54-0.65), recall 0.36, F1 0.45 (0.42-0.46), ROC-AUC 0.77. Noise costs about 0.03 F1, all in precision. Contamination tuned to 0.05 on every seed; at 0.05 the inflated-caesarean claims are no longer flagged (8/14 at 0.10). **Pass.** Note: TC-022's single run (F1 0.51) sits at the top of this seed range. |

Rows above are placeholders for every FR/NFR from `chapter4_system_analysis_design.txt`
Section 4.2 — delete rows that end up not needing a dedicated test, add rows for anything
this list missed. Positive, negative, boundary, and exception cases per
`research_writing.txt` 5.4.1 — don't limit each row to only the happy path.
