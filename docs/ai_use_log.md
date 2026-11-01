# AI-Use Log

Running record of AI-assisted work on this implementation, per
`coursework_materials/research_writing.txt` Section 5.6's disclosure requirement. This is
the raw working log — append an entry every time AI assistance is used. Section 5.6 of
the eventual Chapter 5 write-up wants a short, distilled *declaration* drawn from this
log (tool/provider, what it assisted with, how output was checked, confirmation you
remained responsible) — write that declaration separately and shorter; don't submit this
raw log as the declaration.

Format per entry:

```
## YYYY-MM-DD — <short description of what was worked on>
- Tool/provider:
- What it assisted with:
- What was checked/verified before accepting:
- Anything rejected or corrected:
```

---

<!-- Start appending entries below as implementation work begins. -->

## 2026-10-08 — Removed English-Kiswahili code-switching; switched classifier to Bio_ClinicalBERT
- Tool/provider: Claude Code (Anthropic, Claude Opus 5.5)
- What it assisted with: after the supervisor approved limiting the study to English-language
  clinical text, replaced the code-switching generator in `src/data/noise_injection.py`
  with a seeded text-noise generator (`inject_text_noise`, `apply_text_noise`: clinical
  abbreviations, single-character typos, casing/spacing noise, plus a `has_text_noise`
  flag for clean-vs-noisy evaluation); changed the classifier's default base model from
  AfroXLM-R to `emilyalsentzer/Bio_ClinicalBERT` (fallback `bert-base-uncased`) in
  `classifier.py`, `train.py` and `config.example.yaml`; replaced the three code-switching
  tests with five text-noise tests; updated README and `docs/completion_plan.md`.
- What was checked/verified before accepting: full test suite run, 60 passed. Sample
  noisy output inspected by eye. [Student: add your own review of the changes here.]
- Anything rejected or corrected: [Student: fill in.]

## 2026-10-25 to 2026-10-26 — Django platform: rebuild, hosted services, batch summary, documentation
- Tool/provider: Claude Code (Anthropic, Claude Opus 5.5)
- What it assisted with: proposed and, after the student chose it, built the Django
  replacement for the Streamlit dashboard on branch `django` (`docs/platform_plan.md`):
  project, models and migrations, accounts, roles and Google sign-in, review pages, upload
  validation, Celery jobs with progress, the Colab GPU worker notebook, the SQLite import, the
  batch summary with Markdown and CSV downloads; connected Neon PostgreSQL and Upstash Redis
  (the student created both services and signed in to Neon; secrets were typed into `.env`
  by the student, never into the chat); drew the Chapter 4 diagrams in `docs/diagrams/` and
  drafted the paste-ready Chapter 3, 4 and 5 text (`compare/changes.txt` section D,
  `compare/chapter5.txt`).
- What was checked/verified before accepting: full test suite (285 passing on SQLite; the
  web tests also on local PostgreSQL and on Neon); live runs recorded as TC-038 to TC-044:
  an upload processed by a real worker locally and through Upstash and Neon, the real
  SQLite import into Neon with row counts compared, Google sign-in by the student, every
  diagram rendered and inspected, and each statement in the requirement text checked
  against the code (claims without a test were either given one, such as the session
  timeout and the read-only activity log, or removed). [Student: add your own review.]
- Anything rejected or corrected: defects found by the live runs and fixed (TC-039,
  TC-040, TC-043, TC-044): held-out guard too broad, worker unable to read the uploaded
  file, Google redirect URI not registered, per-row import timing out against Neon. The
  student stopped one import attempt before it ran, then asked for it later. [Student: add
  anything you rejected or changed.]
