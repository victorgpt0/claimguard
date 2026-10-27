# Rubric scoring guide (#37, Section 3.2.4)

The researcher scores 50 randomly drawn audit narratives on five criteria, each from 1 to 5.
This guide fixes what the scores mean before scoring starts, so that a single rater scores
the first and the fiftieth report the same way.

## Procedure

1. Generate the sheet once:
   `python -m src.evaluation.rubric sample --db <Drive>/db/evaluation_v0.6-20k_test.sqlite --seed 0`.
   It holds 50 of the 146 narratives from the held-out test run (TC-045), with the billed
   code, its official title and definition, and the claim's notes. It holds no label.
2. Score blind: do not look up the claims in the test data, the predictions, the
   faithfulness results (`experiments/faithfulness/`) or the test plan while scoring.
3. Read the notes and the definition first, then the narrative. Score every criterion for
   every report; score each criterion on its own (a clear but wrong narrative is clarity 5,
   clinical accuracy 1).
4. Use whole numbers. Use 2 and 4 when a narrative sits between two anchors.
5. Score in no more than two sittings and note the dates and the time taken in the issue.
6. Do not change a score after running `summarise`.

Then: `python -m src.evaluation.rubric summarise --sheet experiments/rubric/sheet.csv`.

## Criteria and anchors

**clinical_accuracy**: are the clinical statements in the narrative correct?
- 5: every clinical statement is correct, including any general medical knowledge it adds
  (typical symptoms, what a procedure implies).
- 3: the main point is correct, but one statement is doubtful or overstated (for example,
  "stable for discharge" taken to show the condition was never acute).
- 1: the main clinical reasoning is wrong.

**inconsistency_identification**: does the narrative state the actual mismatch between the
notes and the billed code?
- 5: names the billed condition and says precisely what in the notes contradicts it or is
  missing (for example, "the notes describe a spontaneous vertex delivery, not a caesarean").
  For a generic note with no clinical content, saying that the note documents nothing that
  supports the code counts as precise.
- 3: says the notes do not support the code, but vaguely, or gives a secondary reason
  ahead of the real one.
- 1: does not identify a mismatch, or identifies the wrong one.

**clarity**: could a claims auditor without clinical training follow it?
- 5: plain language, short, one clear line of argument; any clinical term is explained or
  obvious from context.
- 3: understandable but repetitive, padded, or reliant on unexplained terms.
- 1: confusing or contradictory.

**actionability**: does it tell the auditor what to check or do next?
- 5: points to something concrete to verify (the procedure record, the theatre log, a
  test result, the provider's other claims) or states plainly that the claim should be
  queried and why.
- 3: implies a next step ("further investigation is needed") without saying what.
- 1: gives the auditor nothing to act on.

**no_hallucination**: does everything the narrative says about this claim come from the
notes, the code title or the definition?
- 5: nothing invented. Statements attributed to the notes are in the notes (corrected
  spelling of a noisy note is fine); general knowledge is clearly presented as such.
- 3: one minor unsupported detail that does not change the conclusion.
- 1: presents something as being in the notes that is not there (for example, words from
  the ICD-11 definition quoted as the notes), or invents a test result, history or finding.

## Reporting

Report per criterion the mean ± standard deviation and the share of reports scored 4 or
more, as a single-rater assessment: one non-specialist rater establishes no inter-rater
reliability and no clinical validity (`compare/changes.txt` item 15). The planned
deployment target is a mean of 4.0 or more on every criterion.
