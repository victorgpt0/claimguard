"""Faithfulness check for audit narratives (Section 3.2.4, task 7.4).

A narrative is faithful when it refers to the real evidence behind the verdict, not
only when it reads plausibly. Rules, all on lower-cased text with punctuation replaced
by spaces:

- names_code: the official ICD-11 title appears in the narrative. Fallback: every main
  word of the title appears. Main words come from the title's head (before the first
  comma, cut at "without" / "not known" / "not specified" / "site"), minus stopwords and
  qualifiers such as "unspecified" or "other". So "Malaria due to Plasmodium falciparum,
  unspecified" is named by any narrative containing "malaria", "plasmodium" and
  "falciparum".
- cites_notes: the narrative shares at least one content word (length >= 4) or one
  2-gram of content words with the clinical notes. Content words exclude English
  stopwords, generic audit vocabulary ("patient", "diagnosis", "claim", ...) and words
  of the ICD-11 title, so naming the code does not also count as citing the notes.
- cites_shap: any of the top-3 SHAP features is mentioned, by name or a humanised form
  ("claim_amount_per_day" -> "amount per day"). Reported, not part of the pass rule.
- passed = names_code and cites_notes.

    python -m src.evaluation.faithfulness --db experiments/pipeline_runs/test.db \\
        --cache artifacts/icd11_cache.json
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Words any narrative may use without having read the notes.
GENERIC_WORDS = {
    "patient", "patients", "presented", "presents", "presenting", "history", "clinical",
    "notes", "note", "diagnosis", "diagnosed", "treatment", "treated", "claim", "claims",
    "code", "codes", "coded", "billed", "billing", "bill", "documented", "documentation",
    "record", "records", "consistent", "inconsistent", "inconsistency", "support",
    "supports", "supported", "review", "reviewed", "audit", "auditor", "evidence",
    "condition", "symptoms", "findings", "procedure", "report", "reported", "indicate",
    "indicates", "suggest", "suggests", "described", "describe", "mention", "mentions",
    "mentioned", "icd", "classifier", "verdict", "flagged", "further", "investigation",
    "provided", "given", "noted", "states", "stated", "shows", "showed", "however",
}
STOPWORDS = set(ENGLISH_STOP_WORDS) | GENERIC_WORDS
TITLE_QUALIFIERS = {"unspecified", "specified", "other", "nos", "site", "organism", "agent"}
TITLE_CUTS = (" without ", " not known ", " not specified", " site ")
MIN_WORD_LENGTH = 4

# Plain-language forms of the stage-1 features (src/data/preprocessing.FEATURE_COLUMNS).
HUMANISED = {
    "log_claim_amount": ["claim amount", "amount claimed", "billed amount"],
    "stay_days": ["length of stay", "stay", "days admitted"],
    "claim_amount_per_day": ["amount per day", "per day"],
    "is_inpatient": ["inpatient", "admission", "admitted"],
    "facility_level": ["facility level"],
    "patient_age": ["age", "aged"],
    "is_female": ["sex", "female", "male"],
    "is_dependent": ["dependent", "dependant"],
    "code_frequency": ["code frequency"],
    "amount_vs_code_median": ["median", "typical amount", "usual amount"],
    "unusual_admission": ["unusual admission"],
    "member_claim_count": ["number of claims", "claim count"],
    "member_dependent_claims": ["dependent claims", "dependants"],
    "provider_caesarean_share": ["caesarean share", "caesarean rate", "share of caesarean"],
    "has_missing_field": ["missing field", "missing fields"],
}


def normalise(text: str | None) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).split())


def _tokens(text: str) -> list[str]:
    return normalise(text).split()


def title_main_words(title: str) -> set[str]:
    head = " " + normalise(title.split(",")[0]) + " "
    for cut in TITLE_CUTS:
        head = head.split(cut)[0] + " "
    return {w for w in head.split() if w not in ENGLISH_STOP_WORDS and w not in TITLE_QUALIFIERS}


def names_code(narrative: str, title: str) -> bool:
    text, official = f" {normalise(narrative)} ", normalise(title)
    if not official:
        return False
    if f" {official} " in text:
        return True
    words = title_main_words(title)
    return bool(words) and words <= set(text.split())


def _content(tokens: list[str], exclude: set[str]) -> list[str]:
    return [t for t in tokens if t not in STOPWORDS and t not in exclude and not t.isdigit()]


def cites_notes(narrative: str, notes: str, title: str = "") -> bool:
    title_words = set(_tokens(title))
    note_content = _content(_tokens(notes), title_words)
    narrative_content = _content(_tokens(narrative), title_words)
    words = {t for t in note_content if len(t) >= MIN_WORD_LENGTH}
    if words & set(narrative_content):
        return True
    bigrams = set(zip(note_content, note_content[1:]))
    return bool(bigrams & set(zip(narrative_content, narrative_content[1:])))


def top_features(shap_top3) -> list[str]:
    """A list of names, or a {feature: shap value} dict (top 3 by absolute value)."""
    if isinstance(shap_top3, dict):
        return sorted(shap_top3, key=lambda k: -abs(float(shap_top3[k])))[:3]
    return list(shap_top3 or [])[:3]


def feature_phrases(feature: str) -> list[str]:
    spaced = feature.replace("_", " ")
    stripped = re.sub(r"^(log|is|has) ", "", spaced)
    return [normalise(p) for p in [feature, spaced, stripped, *HUMANISED.get(feature, [])]]


def cites_shap(narrative: str, shap_top3) -> bool:
    text = f" {normalise(narrative)} "
    raw = (narrative or "").lower()
    return any(feature in raw or f" {phrase} " in text
               for feature in top_features(shap_top3) for phrase in feature_phrases(feature))


def check(narrative: str, icd11_title: str, notes: str, shap_top3) -> dict:
    result = {"names_code": names_code(narrative, icd11_title),
              "cites_notes": cites_notes(narrative, notes, icd11_title),
              "cites_shap": cites_shap(narrative, shap_top3)}
    result["passed"] = result["names_code"] and result["cites_notes"]
    return result


def load_narrated_reports(db_path: str | Path) -> list[dict]:
    """Every report with a narrative, joined to its claim (src/db/schema.sql)."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """SELECT r.id AS report_id, c.icd11_code, c.clinical_notes, r.shap_attribution,
                      r.llm_narrative
               FROM AuditReport r JOIN Claim c ON c.id = r.claim_id
               WHERE r.llm_narrative IS NOT NULL ORDER BY r.id""").fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def titles_for(codes, cache_path: Path) -> dict[str, str]:
    from src.icd11.cache import ICD11Cache
    cache = ICD11Cache(cache_path)
    return {code: cache.get(code).title if code in cache else "" for code in set(codes)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, required=True, help="SQLite database of a pipeline run")
    parser.add_argument("--cache", type=Path, default=Path("artifacts/icd11_cache.json"))
    parser.add_argument("--out", type=Path, default=Path("experiments/faithfulness"))
    args = parser.parse_args()

    reports = load_narrated_reports(args.db)
    if not reports:
        parser.error(f"no narrated reports in {args.db}")
    titles = titles_for([r["icd11_code"] for r in reports], args.cache)
    rows = []
    for r in reports:
        shap = json.loads(r["shap_attribution"] or "{}")
        rows.append({"report_id": r["report_id"], "icd11_code": r["icd11_code"],
                     "title": titles[r["icd11_code"]], "shap_top3": ";".join(top_features(shap)),
                     **check(r["llm_narrative"], titles[r["icd11_code"]], r["clinical_notes"], shap)})
    frame = pd.DataFrame(rows)
    summary = {"db": str(args.db), "narratives": len(frame),
               "missing_titles": int((frame["title"] == "").sum()),
               "pass_rates": {k: round(float(frame[k].mean()), 4)
                              for k in ["names_code", "cites_notes", "cites_shap", "passed"]},
               "rule": "passed = names_code and cites_notes; cites_shap reported only"}

    out = args.out / f"{args.db.stem}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "results.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
