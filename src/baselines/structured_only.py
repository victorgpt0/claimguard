"""Structured-only baseline (Section 3.2.4, #30): the same model family and training
procedure as the Semantic Consistency Classifier, but without the clinical notes. Each
claim's structured billing fields are written as a short text and paired with the billed
code's ICD-11 reference text, so `src.classification.train` runs unchanged.

It answers "can structured data alone detect the fraud?", so its label is `is_fraud`
(every typology), not the text label: fraud becomes "inconsistent", a clean claim
"consistent". Only fields a payer receives on a claim are used; identifiers, injection
flags, the notes and anything derived from the label are left out.

    python -m src.baselines.structured_only export --data <processed>/v0.6-20k \\
        --cache artifacts/icd11_cache.json --out data/classifier_structured/v0.6-20k
    python -m src.classification.train --train-file .../train.jsonl --val-file .../validation.jsonl \\
        --experiments-dir experiments/structured_only/v0.6-20k          # Colab, 5 seeds
    python -m src.baselines.structured_only predict --runs experiments/structured_only/v0.6-20k \\
        --data data/classifier_structured/v0.6-20k --split validation
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd

from src.classification.export import SPLITS, reference_text
from src.data.preprocessing import repair_icd11_code
from src.evaluation.predictions import PREDICTIONS_DIR, write_predictions
from src.icd11.cache import ICD11Cache

SYSTEM = "structured_only"
RUN_SEED = re.compile(r"_seed(\d+)$")


def stay_days(service_date, discharge_date) -> int | None:
    if discharge_date is None or (isinstance(discharge_date, float) and math.isnan(discharge_date)):
        return None
    days = (pd.Timestamp(discharge_date) - pd.Timestamp(service_date)).days
    return None if pd.isna(days) else int(days)


def structured_text(claim) -> str:
    """e.g. "code JB20.0 | KES 0 | outpatient | stay none | level 3 sub_county_hospital |
    dependent | age 27 | female". Every value comes from the submitted claim."""
    days = stay_days(claim.service_date, claim.discharge_date)
    age = "unknown" if pd.isna(claim.patient_age) else str(int(claim.patient_age))
    return " | ".join([
        f"code {repair_icd11_code(claim.icd11_code)}",
        f"KES {claim.claim_amount:,.0f}",
        str(claim.claim_type),
        f"stay {'none' if days is None else f'{days} days'}",
        f"level {claim.facility_level} {claim.facility_type}",
        str(claim.beneficiary_type),
        f"age {age}",
        str(claim.patient_sex),
    ])


def export_split(claims: pd.DataFrame, cache: ICD11Cache) -> list[dict]:
    examples = []
    for row in claims.itertuples(index=False):
        code = repair_icd11_code(row.icd11_code)
        examples.append({
            "claim_id": row.claim_id,
            "icd11_code": code,
            # train.py reads the first input of the pair from this field; here it holds the
            # structured fields, never the notes.
            "clinical_notes": structured_text(row),
            "icd11_definition": reference_text(cache.get(code)),
            "label": "inconsistent" if row.is_fraud else "consistent",
            "fraud_typology": row.fraud_typology,
            "has_text_noise": bool(row.has_text_noise),
            "has_structured_noise": bool(getattr(row, "has_structured_noise", False)),
        })
    return examples


def export(data_dir: Path, cache_path: Path, out_dir: Path) -> dict[str, int]:
    cache = ICD11Cache(cache_path)
    if not len(cache):
        raise FileNotFoundError(f"ICD-11 cache at {cache_path} is missing or empty")
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in SPLITS:
        examples = export_split(pd.read_parquet(data_dir / f"{split}.parquet"), cache)
        with open(out_dir / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
        counts[split] = len(examples)
    return counts


def seed_of(run_dir: Path) -> int:
    match = RUN_SEED.search(run_dir.name)
    if not match:
        raise ValueError(f"{run_dir.name}: not a train.py run folder (<timestamp>_seed<N>)")
    return int(match.group(1))


def predictions_frame(examples: list[dict], probabilities: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"claim_id": [e["claim_id"] for e in examples],
                         "flagged": [int(p > 0.5) for p in probabilities],
                         "score": probabilities})


def predict(runs: Path, data: Path, split: str, base_model: str, root: Path) -> list[Path]:
    """Scores `split` with every seed's checkpoint under `runs`, writing
    `structured_only_seed<N>` predictions in the shared format."""
    from src.classification.classifier import SemanticConsistencyClassifier
    from src.classification.train import default_device, predict_probabilities

    examples = [json.loads(line) for line in open(data / f"{split}.jsonl", encoding="utf-8")]
    device = default_device()
    paths = []
    for run_dir in sorted(p for p in runs.iterdir() if p.is_dir() and RUN_SEED.search(p.name)):
        classifier = SemanticConsistencyClassifier(str(run_dir), base_model, device)
        if classifier.model_version == "untrained-base":
            raise FileNotFoundError(f"no checkpoint in {run_dir}")
        probabilities = predict_probabilities(classifier.model, examples, classifier.tokenizer, device)
        system = f"{SYSTEM}_seed{seed_of(run_dir)}"
        paths.append(write_predictions(system, split, predictions_frame(examples, probabilities), root))
    if not paths:
        raise FileNotFoundError(f"no <timestamp>_seed<N> run folders in {runs}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    exp = commands.add_parser("export", help="write the structured-only JSONL splits")
    exp.add_argument("--data", type=Path, required=True, help="processed dataset directory")
    exp.add_argument("--cache", type=Path, required=True, help="ICD-11 cache JSON")
    exp.add_argument("--out", type=Path, required=True)
    pred = commands.add_parser("predict", help="score a split with every trained seed")
    pred.add_argument("--runs", type=Path, required=True, help="train.py --experiments-dir")
    pred.add_argument("--data", type=Path, required=True, help="the exported JSONL directory")
    pred.add_argument("--split", default="validation")
    pred.add_argument("--final", action="store_true", help="required to score the test split")
    pred.add_argument("--base-model", default="emilyalsentzer/Bio_ClinicalBERT")
    pred.add_argument("--predictions-root", type=Path, default=PREDICTIONS_DIR)
    args = parser.parse_args()
    if args.command == "export":
        for split, n in export(args.data, args.cache, args.out).items():
            print(f"{split}: {n} examples")
        return
    if args.split == "test" and not args.final:
        parser.error("the test split is reserved for the final evaluation; pass --final to score it")
    for path in predict(args.runs, args.data, args.split, args.base_model, args.predictions_root):
        print(f"-> {path}")


if __name__ == "__main__":
    main()
