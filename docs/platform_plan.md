# Platform plan: Django, PostgreSQL, Celery and a Colab GPU worker (25 Oct)

## Decision

Taken with the student on 25 Oct. **Supervisor agreement is still needed**, because Chapter 3
names Streamlit and SQLite and Chapter 4's IR-03, IR-04, architecture, class diagram and
schema follow from them; Chapter 5 must present the change as a deliberate deviation from
the approved design, with this reasoning.

1. **Results first.** The held-out results are the research contribution. Phase 0 finishes
   them on the current stack before any platform work starts.
2. **Then replace the Streamlit dashboard with a Django application.** The project now needs
   database migrations, roles, account management, uploads, background jobs with visible
   progress, and stored report files. Streamlit offers a sign-in (`st.login`), pages
   (`st.navigation`) and periodic refresh (`st.fragment`), but no data layer or migrations,
   no job queue, and a rerun-the-whole-script model on which sign-in and roles had to be
   built by hand (#55, #56, #61). Django provides those parts as one established,
   long-term-supported framework.
3. **GPU work runs on a Colab worker notebook**, not by running pipeline steps by hand. The
   dashboard queues jobs and shows their progress. A Colab session cannot be started from
   outside, Colab ends idle runtimes and caps free sessions at 12 hours, so a GPU job runs
   while the worker notebook is open, and the dashboard shows whether a GPU worker is online.
   Modal (serverless GPU) was considered and kept as the alternative if one-click GPU runs
   become necessary.

## Stack

| Need | Choice |
|---|---|
| Web framework | Django 5.2 LTS: auth, groups and permissions, sessions, PBKDF2 hashing, admin site |
| Database and migrations | PostgreSQL (hosted free tier, reachable from the laptop and Colab); Django migrations |
| Google sign-in | django-allauth; only emails an Administrator has linked |
| UI | Django templates, HTMX for live progress, Vega-Lite charts (existing Altair specs) |
| Background jobs | Celery with Redis (hosted free tier); a `cpu` queue (local worker) and a `gpu` queue (Colab worker) |
| Files | Report exports through Django file storage (local disk, or Google Cloud Storage via django-storages) |
| Models and datasets | Hugging Face Hub (private), tagged versions |
| Reproducibility | uv lockfile; Docker Compose (web, cpu worker, Postgres, Redis); Git tags per sprint |
| ML code | `src/` unchanged as a library; tasks call it, with a progress callback added to `AuditPipeline.run` |

What carries over from the Streamlit work: the rules (linked-email-only Google sign-in,
deactivate never delete, last-Administrator guard, activity log without passwords or note
text, upload validation rules, test-split guard) and the tests' scenarios. The Streamlit code
itself is replaced.

## Phases

| Phase | Work | Days | Issues |
|---|---|---|---|
| 0 | Structured-only and zero-shot baselines (code, then Colab runs); feedback retraining run; pipeline and baselines on the test split; held-out evaluation, strata, cost | 2–3, GPU-bound | #30, #52, #31, #33, #34, #35 |
| 1 | Django project: models and initial migration from the current schema, auth and roles, allauth Google, admin, activity log, Docker Compose, uv | 1.5 | new |
| 2 | Screens: overview charts, review queue with the six panels, decisions and feedback examples, model and feedback pages | 2 | new |
| 3 | Jobs: upload and validation, Celery tasks with progress, Colab worker notebook, CPU worker | 2 | #57, #58 (re-scoped) |
| 4 | Batch summary and report files; migrate the existing SQLite reports into PostgreSQL | 1 | #59 (re-scoped) |
| 5 | Documentation: Chapter 3 tools, Chapter 4 requirements and diagrams, Chapter 5 deviation | 1–1.5 | #60 (re-scoped) |

About 9–10 days against 13 left before 7 Nov. If phase 0 slips, phase 4's file storage and
the activity-log screen are cut before anything in the evaluation.

`docs/system_features_plan.md` (the Streamlit plan) is superseded from item C onward.
