# System features plan: login, roles, batch input and output (25 Oct)

The supervisor asked for a complete system around the ML engine, not only the engine itself.
This reverses the 12 Oct descoping of sign-in and roles. Decisions taken with the student on
25 Oct:

- **Input:** the auditor uploads a claims batch in the dashboard; it is validated and stored,
  and **processed on Colab** (stage 1, classifier, SHAP and narratives) as now. No ML runs in
  the dashboard.
- **Output:** a per-batch summary report, shown in the dashboard and downloadable.
- **Account features:** sign-in, sign-out, change own password, Administrator user
  management, and an activity log.
- **Upload validation:** a clear, per-row error report before anything is stored.

## Principles that keep friction low

1. **The ML pipeline is not touched**, except one additive `--batch-file` option. Detection,
   the classifier, the explanation-only LLM gate, the evaluation scripts and the frozen
   `evaluated_v1` checkpoint stay as they are. No result in `tests/test_plan.md` or the
   Chapter 5 draft changes.
2. **No new dependencies.** Sign-in uses the existing `src/auth/user.py` (PBKDF2, standard
   library) and `st.session_state`, not streamlit-authenticator. The summary downloads as CSV
   and Markdown, so no PDF library is needed.
3. **Reuse what already exists:** the `User` table, `authenticate()`, `AuditReport.reviewed_by`,
   `record_decision(reviewer_id=...)`, `load_claims_batch()`, `REQUIRED_COLUMNS`, and the ICD-11
   cache for validating codes.
4. **Schema changes are additive only**, through the existing migration loop in
   `AuditReportRepository.__init__`. The TC-034 database on Drive keeps working, and its 398
   reports stay valid (old reports simply have no batch).
5. **Chapter 4 already describes most of this** (authenticated auditor, seeded accounts,
   role-based views in IR-03, "the Fraud Auditor submits claim batches"). The code is catching
   up with the document. Only user management and the activity log are new to the document.

## Roles and views

| Tab | Fraud Auditor | Administrator |
|---|---|---|
| Overview | yes | yes |
| Upload batch | yes | no |
| Batches (status + summary report) | yes | yes |
| Review queue (decisions) | yes | no, read-only at most |
| Model performance | no | yes |
| Feedback loop | no | yes |
| Users | no | yes |
| Activity log | no | yes |
| Account (change password, sign out) | yes | yes |

Every page checks the session's role before reading data (NFR-03); there is no route to a
report without signing in.

## Work items (one issue and one commit each)

### A. Sign-in, roles and reviewer identity
- Sign-in form before any tab renders; `authenticate()` against `User`; the user is held in
  `st.session_state`. Same message for an unknown user and a wrong password (already the
  design of `authenticate()`).
- Sign-out button; a 30-minute inactivity timeout (configurable in `config.yaml`).
- The tab set is built from the role, as in the table above.
- `record_decision(..., reviewer_id=user.id)`: FR-09's missing reviewer identity is fixed.
- First-run bootstrap: `python -m src.db.seed` seeds **one Administrator** (password asked
  interactively). Auditors are then created in the app (item B). Seeding is idempotent.
- Tests: AppTest with no session sees only the sign-in form; wrong password refused; an
  Auditor cannot reach admin tabs; a decision stores the reviewer's id. Update the existing
  `tests/test_dashboard.py` fixtures to sign in first.

### B. User management, change password, activity log
- Schema: `User` gains `active INTEGER NOT NULL DEFAULT 1` and `created_at`; new table
  `ActivityLog(id, user_id, action, detail, at)`.
- Administrator, **Users** tab: create an Auditor or Administrator, deactivate or reactivate
  an account, reset a password. Accounts are deactivated, never deleted, so decisions stay
  attributable. The last active Administrator cannot be deactivated.
- **Account** tab: change own password (current password required, minimum length 10).
- `authenticate()` refuses deactivated accounts.
- Logged actions: sign-in success and failure, sign-out, upload accepted or rejected,
  decision, account created, deactivated or reset, password changed, summary downloaded.
  Never log passwords or note text.
- Administrator, **Activity log** tab: filter by user and action, and download it as CSV.
- Tests: creating, deactivating and resetting users; the last-admin guard; a deactivated
  user cannot sign in; every action listed above writes one row.

### C. Batch upload and validation (FR-01)
- Auditor, **Upload batch** tab: CSV upload, a downloadable template CSV (the
  `REQUIRED_COLUMNS` header plus 2 example rows), and a "this file contains synthetic data
  only" confirmation (DR-06).
- Validation, reporting every error by row and column before anything is stored:
  - required columns present; no empty `claim_id`, `provider_id`, `icd11_code` or notes;
  - `claim_amount` numeric and at least 0 (KES 0 is valid for capitation outpatient claims);
  - dates parse, and the discharge date is not before the service date;
  - `icd11_code` present in the ICD-11 cache (after `repair_icd11_code`);
  - notes of at least 5 words, Latin script only (reuse `validate_note`'s rule);
  - no duplicate `claim_id` within the file or against claims already in the database;
  - **no `claim_id` from the held-out test split** (DR-08: the test set can never reach the
    review queue or the Feedback Dataset);
  - a maximum of 2,000 rows per batch.
- Any error rejects the whole file and shows the error table; nothing is half-stored.
- On success the file is written to `<data_dir>/uploads/<batch_id>.csv` (on Drive through the
  rclone mount, so Colab sees it) and a `Batch` row is created:
  `Batch(id, filename, uploaded_by, uploaded_at, claims, status, processed_at, run_log)`, with
  status `uploaded`, then `processing`, then `processed` or `failed`. `Claim` gains `batch_id`.
- Tests: one test per validation rule (positive and negative), the test-split guard, the
  duplicate guard, and storage only on success.

### D. Processing uploaded batches on Colab
- `src/pipeline.py`: add `--batch-file PATH --batch-id N` as an alternative to `--batch
  <split>` (stage 1 is still fitted on the dataset's train split). The run stores `batch_id`
  on each claim, sets the batch's status, and records the run-log path. Failure sets status
  `failed` with the reason (NFR-02 style); it never stops other batches.
- `src/pipeline.py --pending-uploads`: processes every batch whose status is `uploaded`.
- Notebook 03: one new cell, "Process uploaded batches", which runs the line above with the
  usual classifier and LLM settings.
- Keep the existing rule: **don't write the SQLite database from Colab and the laptop at
  once.** Upload, then close the dashboard (or just don't make decisions) while Colab runs.
- Tests: a `--batch-file` run with mocked models stores `batch_id` and sets status
  `processed`; a failing batch is marked `failed` and the next one still runs.

### E. Batch summary report (output)
- **Batches** tab: one row per batch (file, uploader, uploaded, claims, status). Opening a
  processed batch shows its summary:
  - claims received; flagged at stage 1 by each detector; cleared by the classifier;
    queued per lane; narratives generated and failed;
  - amount under audit (KES); top providers and ICD-11 codes among queued claims;
  - review progress: pending, approved, rejected, escalated, and by whom;
  - the classifier version and thresholds used (from the run log).
- Download the summary as **Markdown** (a readable report) and the batch's reports as **CSV**
  (claim, lane, verdict, confidence, SHAP top 3, decision, reviewer, time).
- Built from SQLite plus the run log; nothing is recomputed.
- Tests: summary counts on a seeded database; both downloads produce the expected
  columns and sections.

### F. Documentation (see the next section)

Suggested order: A, B, C, D, E, F. A and C are the minimum for the "complete system" demo;
B and E complete it. Estimate: A 0.5 day, B 1 day, C 1 day, D 0.5 day, E 0.5 day, F 1 day.

**Schedule risk:** Sprint 7 (26 Oct to 1 Nov) already holds the held-out evaluation and
hosting. This work fits there only if the retraining run and baselines are not delayed
further. If time runs short, cut the activity-log viewer UI (keep the logging) and the
Markdown summary (keep the CSV) before cutting anything from the evaluation.

## Demo data for uploads

Every deployment-pool claim is already in the database (TC-034), so a demo upload of pool
claims would be rejected as duplicates. To be decided: either generate a fresh small batch
(about 200 claims) with `src.data.build_dataset` on Colab under a new seed, kept out of all
splits (recommended), or remove the TC-034 rows for a slice of the pool before the demo.
**Never use test-split claims**; the validator refuses them anyway.

## Documentation changes

**In this repo** (committed with the matching issue):
- `README.md`: seeding the first Administrator, signing in, uploading a batch, processing
  uploads on Colab.
- `docs/sprint_plan.md`: add items A to E under Sprint 7, and record the 25 Oct reversal of
  the 12 Oct descoping.
- `tests/test_plan.md`: TC-008 (NFR-03) and TC-009 (reviewer identity) re-run and marked; new
  TC rows for sign-in, roles, user management, the activity log, upload validation, batch
  processing and the summary report.
- `docs/hosting_guide.md`: the seeding step, and a note that the hosted copy must not ship a
  default password.
- `docs/ai_use_log.md`: an entry for this work (the student fills in the review fields).
- `AGENTS.md` and the root `CLAUDE.md`: remove "in-app user management cut" from the
  deliberate decisions and the descoping note.

**In the student's real document** (`compare/changes.txt` gets the exact text to paste):
- **FR-01:** "...ingest a batch of synthetic claims **uploaded by an authenticated Fraud
  Auditor as a CSV file, validated against the claim schema before storage**...".
- **New FR-15** (input validation), **FR-16** (batch summary report), **FR-17** (user
  account management by the Administrator) and **FR-18** (activity log); and add them to
  Table 4.1.
- **NFR-03:** replace "The two user accounts are seeded at setup" with "One Administrator
  account is seeded at setup; the Administrator creates and deactivates other accounts. All
  sign-ins and actions are recorded in an activity log."
- **IR-03:** list the views per role as in the table above.
- **Figure 4.1 (use case diagram):** add Sign in, Upload claims batch, View batch summary,
  Change password (both actors), and Manage user accounts and View activity log
  (Administrator). Delete the sentence "Account creation ... not shown as use cases".
- **Figure 4.4 (class diagram):** `User` gains `active`; add `UserRepository`,
  `ActivityLog`, `BatchUploadValidator` and `Batch`.
- **Figure 4.7 (database schema):** `User.active` and `created_at`, `Claim.batch_id`, and the
  new `Batch` and `ActivityLog` tables.
- **Sequence diagram:** add one for "upload a batch and process it on Colab".
- **Section 3.5.1** (Chapter 3's use case text): the same actor and use-case additions.
- **Chapter 5 draft** (`compare/chapter5.txt`): an implementation section for access
  control, upload and outputs; NFR-03 changes from "not met (descoped)" to met, with the test
  evidence.

The PlantUML sources for the changed diagrams are drafted in `docs/diagrams/` and rendered
to PNG for the student to paste in, since the real document can't be edited from here.
