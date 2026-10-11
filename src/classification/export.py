"""Exports the processed dataset splits as the classifier's JSONL input (Section 3.2.3).

Each line pairs a claim's clinical notes with the billed ICD-11 code's reference text and
the `label` target, plus the columns evaluation stratifies on (typology, text and
structured noise). The reference text comes from the ICD-11 cache, never a live API, so
this runs anywhere the cache file is available (Colab included).

    python -m src.classification.export --data data/processed/v0.5-5k \\
        --cache artifacts/icd11_cache.json --out data/classifier/v0.5-5k
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data.preprocessing import repair_icd11_code
from src.icd11.cache import ICD11Cache
from src.icd11.client import ICD11Definition

SPLITS = ("train", "validation", "test", "deployment_pool")


def reference_text(definition: ICD11Definition) -> str:
    """The second input of the sentence pair: the code's official title, then its
    definition when it has one (its own or inherited from a parent category). Shared by
    training and inference so the classifier sees the same format in both."""
    if definition.definition:
        return f"{definition.title}. {definition.definition}"
    return definition.title


def export_split(claims: pd.DataFrame, cache: ICD11Cache) -> list[dict]:
    examples = []
    for row in claims.itertuples(index=False):
        # The billed code, as the classifier must judge the claim as submitted. Malformed
        # codes (structured noise) are repaired first, as the pipeline does.
        code = repair_icd11_code(row.icd11_code)
        example = {
            "claim_id": row.claim_id,
            "icd11_code": code,
            "clinical_notes": row.clinical_notes,
            "icd11_definition": reference_text(cache.get(code)),
            "label": row.label,
            # For stratified evaluation (Section 3.2.4); v0.4 has no structured noise.
            "fraud_typology": row.fraud_typology,
            "has_text_noise": bool(row.has_text_noise),
            "has_structured_noise": bool(getattr(row, "has_structured_noise", False)),
        }
        examples.append(example)
    return examples


def export(data_dir: Path, cache_path: Path, out_dir: Path) -> dict[str, int]:
    cache = ICD11Cache(cache_path)
    if not len(cache):
        raise FileNotFoundError(f"ICD-11 cache at {cache_path} is missing or empty")
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in SPLITS:
        claims = pd.read_parquet(data_dir / f"{split}.parquet")
        examples = export_split(claims, cache)
        with open(out_dir / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
        counts[split] = len(examples)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True, help="processed dataset directory")
    parser.add_argument("--cache", type=Path, required=True, help="ICD-11 cache JSON")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for split, n in export(args.data, args.cache, args.out).items():
        print(f"{split}: {n} examples")


if __name__ == "__main__":
    main()
