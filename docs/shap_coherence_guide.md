# SHAP coherence marking guide (#38, Section 3.2.4)

The researcher marks 100 reported claims from the held-out test run, sampled as equally as
possible across the six fraud typologies and clean claims (`none`), on one question: do the
claim's top-3 SHAP features point to something actually unusual about this claim that an
auditor could check? This tests FR-03: the attribution shown beside each report should
help the auditor, not just exist.

## Procedure

1. Generate the sheet once:
   `python -m src.evaluation.shap_coherence sample --log <Drive>/experiments/pipeline_runs/v0.6-20k_test_20261027T211724Z.json --data <Drive>/data/processed/v0.6-20k --seed 0`.
   It reads the test split for the typology and the claim fields only; the evaluation is
   already final (TC-045), and nothing is tuned on these marks.
2. For each row, read the typology, the claim fields and the three features with their
   SHAP values (positive = pushed the claim towards anomalous). Mark `coherent` y or n and
   write a few words in `reason`.
3. The typology is shown on purpose: coherence is judged against why the claim is unusual.
4. Mark in one sitting if possible; don't change marks after running `summarise`.

Then: `python -m src.evaluation.shap_coherence summarise --sheet experiments/shap_coherence/sheet.csv`.

## Marking rule

Mark **y** when the top feature, or two of the three, refer to what is actually unusual
about the claim, judged from its typology and its fields, and no top feature contradicts the
fields. Mark **n** when the features have nothing to do with anything visible in the claim,
or contradict it (for example, `stay_days` on top for an outpatient claim with no stay).

What a coherent attribution looks like for each typology:

| Typology | Features that explain it |
|---|---|
| anomalous_dependents | `member_dependent_claims`, `member_claim_count`, `is_dependent` |
| outpatient_to_inpatient | `is_inpatient`, `stay_days`, `unusual_admission`, `claim_amount_per_day` |
| upcoding | `log_claim_amount`, `amount_vs_code_median`, `code_frequency`, `stay_days` |
| inflated_caesarean | `provider_caesarean_share` |
| ghost_billing, record_fabrication | The fraud is in the notes, so structured features rarely explain it. Mark y only if a feature points to something genuinely unusual in the fields; write "text fraud" in `reason` when marking n for this reason |
| none (clean claim flagged) | Mark y if the features explain why a clean claim looked unusual (a high amount at a level 5–6 hospital, a missing field, a rare code); this is coherent even though the flag was wrong |

**Flag sources.** The SHAP values always explain the Isolation Forest score. When
`flag_sources` is `text_screen` alone, Isolation Forest did not flag the claim, so the
attribution explains a detector that did not raise the flag: mark it on the same rule, and
write "screen only" in `reason`. `summarise` reports by typology; the screen-only rows can be
separated afterwards from the `reason` column or `flag_sources`.

## Reporting

Report the coherent share overall and per typology, as a single-rater judgement, with the
caveat that the rater knows the typology.
