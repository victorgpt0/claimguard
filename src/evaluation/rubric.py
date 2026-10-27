"""Explanation-quality rubric (Section 3.2.4, task 7.5).

`sample` draws narrated reports (fixed seed) from a pipeline run's SQLite database into
a scoring sheet with a blank 1-5 column per criterion. The sheet carries the billed code's
ICD-11 definition, which the narrator was given, so invented content can be told from
content taken from the definition; it carries no ground-truth columns, so scoring is
blind to the claim's label. Anchors for the scores: docs/rubric_guide.md. `summarise` reports
per-criterion mean ± std and the share of reports scored 4 or higher.

    python -m src.evaluation.rubric sample --db test.db --cache artifacts/icd11_cache.json
    python -m src.evaluation.rubric summarise --sheet experiments/rubric/sheet.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.evaluation.faithfulness import load_narrated_reports, titles_for

CRITERIA = ["clinical_accuracy", "inconsistency_identification", "clarity", "actionability",
            "no_hallucination"]
SHEET_COLUMNS = ["report_id", "icd11_code", "title", "definition", "notes", "narrative"] + CRITERIA
GOOD_SCORE = 4


def definitions_for(codes, cache_path: Path) -> dict[str, str]:
    from src.icd11.cache import ICD11Cache
    cache = ICD11Cache(cache_path)
    return {code: (cache.get(code).definition or "") if code in cache else "" for code in set(codes)}


def sample_sheet(reports: list[dict], titles: dict[str, str], n: int = 50, seed: int = 0,
                 definitions: dict[str, str] | None = None) -> pd.DataFrame:
    frame = pd.DataFrame(reports)
    frame = frame.sample(n=min(n, len(frame)), random_state=seed).sort_values("report_id")
    sheet = pd.DataFrame({"report_id": frame["report_id"], "icd11_code": frame["icd11_code"],
                          "title": frame["icd11_code"].map(titles).fillna(""),
                          "definition": frame["icd11_code"].map(definitions or {}).fillna(""),
                          "notes": frame["clinical_notes"], "narrative": frame["llm_narrative"]})
    for criterion in CRITERIA:
        sheet[criterion] = ""
    return sheet.reset_index(drop=True)[SHEET_COLUMNS]


def summarise(sheet: pd.DataFrame) -> dict:
    """Blank cells are unscored and left out; scores must be 1-5."""
    out = {}
    for criterion in CRITERIA:
        scores = pd.to_numeric(sheet[criterion], errors="coerce").dropna()
        if ((scores < 1) | (scores > 5)).any():
            raise ValueError(f"{criterion}: scores must be between 1 and 5")
        out[criterion] = {"scored": len(scores),
                          "mean": round(float(scores.mean()), 3) if len(scores) else None,
                          "std": round(float(scores.std(ddof=1)), 3) if len(scores) > 1 else None,
                          "share_ge4": round(float((scores >= GOOD_SCORE).mean()), 3) if len(scores) else None}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--db", type=Path, required=True)
    s.add_argument("--cache", type=Path, default=Path("artifacts/icd11_cache.json"))
    s.add_argument("--n", type=int, default=50)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", type=Path, default=Path("experiments/rubric/sheet.csv"))
    s.add_argument("--force", action="store_true", help="overwrite an existing (possibly scored) sheet")
    m = sub.add_parser("summarise")
    m.add_argument("--sheet", type=Path, default=Path("experiments/rubric/sheet.csv"))
    args = parser.parse_args()

    if args.command == "sample":
        if args.out.exists() and not args.force:
            parser.error(f"{args.out} exists and may hold scores; pass --force to overwrite")
        reports = load_narrated_reports(args.db)
        if not reports:
            parser.error(f"no narrated reports in {args.db}")
        codes = [r["icd11_code"] for r in reports]
        sheet = sample_sheet(reports, titles_for(codes, args.cache), args.n, args.seed,
                             definitions_for(codes, args.cache))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        sheet.to_csv(args.out, index=False)
        print(f"{len(sheet)} reports -> {args.out}")
    else:
        result = {"sheet": str(args.sheet), "criteria": summarise(pd.read_csv(args.sheet))}
        out = args.sheet.with_name(args.sheet.stem + "_summary.json")
        out.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        print(f"-> {out}")


if __name__ == "__main__":
    main()
