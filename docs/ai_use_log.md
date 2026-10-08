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
