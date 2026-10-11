"""Dataset assembly, stage C of dataset construction (Sections 3.2.1, 3.2.2, 3.2.4).

Takes stage A's structured claims and stage B's LLM notes and produces the labelled,
split dataset:
  1. Join notes to claims; drop claims whose note failed validation.
  2. Inject the four claim-level Table 3.1 typologies (upcoding, outpatient-to-inpatient
     conversion, ghost billing, record fabrication), each on claims no other typology has
     touched, so every fraudulent claim has exactly one typology.
  3. Derive labels: is_fraud (any typology), fraud_typology, and the classifier's
     consistent/inconsistent label.
  4. Carve off the Deployment/Demonstration Pool (DR-08), then split the rest 60/20/20.
  5. Keep test-time inputs unseen in training (DR-03): fabricated claims in the test split
     and pool get boilerplate notes from a held-out half of the boilerplate pool, and text
     noise there uses a different shorthand glossary from training/validation.
  6. Write one parquet per split and a manifest recording inputs, counts and code version.

Usage: python -m src.data.build_dataset --raw data/raw --out data/processed
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.generator import load_catalogue
from src.data.noise_injection import (
    apply_structured_noise,
    apply_text_noise,
    inject_ghost_billing,
    inject_outpatient_to_inpatient_conversion,
    inject_record_fabrication,
    inject_upcoding,
)
from src.data.preprocessing import ClaimsPreprocessor

TYPOLOGY_FLAGS = {
    "is_upcoded": "upcoding",
    "is_converted_admission": "outpatient_to_inpatient",
    "is_ghost_billed": "ghost_billing",
    "is_inflated_csection": "inflated_caesarean",
    "is_fabricated_record": "record_fabrication",
    "is_anomalous_dependent": "anomalous_dependents",
}
# Typologies whose notes no longer describe the billed code. Conversion keeps a correct
# code (the fraud is the admission), and the two provider-level typologies have consistent
# notes (the fraud is in claim volumes), so the classifier should call those consistent.
TEXT_INCONSISTENT = {"upcoding", "ghost_billing", "record_fabrication"}

# Held-out shorthand for the test split and pool (DR-03): none of these forms appear in
# noise_injection.CLINICAL_ABBREVIATIONS, which is what training/validation use.
UNSEEN_ABBREVIATIONS: dict[str, str] = {
    "on examination": "O/E",
    "respiratory rate": "RR",
    "heart rate": "HR",
    "follow up": "F/U",
    "patient": "pnt",
    "complains of": "C/O",
    "history": "Hx.",
    "diagnosis": "Imp:",
    "treatment": "Rx:",
    "abdomen": "abdo",
    "blood pressure": "B/P",
    "temperature": "T°",
    "admitted": "admn",
    "discharged": "D/C'd",
    "review": "r/v",
    "normal": "NAD",
}

POOL_FRACTION = 0.10
NOISE_FRACTION = 0.30
NOISE_RATE = 0.20
# Share of claims given one structured-field defect, independent of the text noise.
STRUCTURED_NOISE_FRACTION = 0.10


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True, cwd=Path(__file__).parent).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clean(claims: pd.DataFrame) -> pd.Series:
    """Claims no typology has touched yet."""
    flags = [c for c in TYPOLOGY_FLAGS if c in claims.columns]
    return ~claims[flags].fillna(False).any(axis=1)


def _inject(claims: pd.DataFrame, eligible: pd.Series, target: int, inject, flag: str,
            **kwargs) -> pd.DataFrame:
    """Applies one injection function to the eligible, untouched claims, flagging about
    `target` of them, and merges the result back."""
    pool = claims[eligible & _clean(claims)]
    rate = min(1.0, target / len(pool)) if len(pool) else 0.0
    injected = inject(pool, rate=rate, **kwargs)
    out = claims.copy()
    if flag not in out.columns:
        out[flag] = False
    for column in injected.columns:
        if column not in out.columns:
            out[column] = None
        out.loc[injected.index, column] = injected[column]
    out[flag] = out[flag].fillna(False).astype(bool)
    return out


def inject_claim_level_typologies(claims: pd.DataFrame, catalogue: dict, target: int,
                                  boilerplate: list[str], seed: int) -> pd.DataFrame:
    conditions = {c["code"]: c for c in catalogue["conditions"]}
    tariffs = {c["code"]: float(c["tariff_kes"]) for c in catalogue["conditions"] if "tariff_kes" in c}
    per_diem = {int(k): float(v) for k, v in catalogue["per_diem_rates_kes"].items()}
    out = claims.copy()
    out["original_icd11_code"] = out["icd11_code"]

    # Upcoding only where the facility could plausibly provide the higher-tariff service;
    # an upcode to a caesarean at a dispensary would be caught by any eligibility rule.
    upcodes = {code: c["upcode_to"] for code, c in conditions.items() if c.get("upcode_to")}
    can_upcode = out.apply(
        lambda r: r["icd11_code"] in upcodes and all(
            r["facility_level"] in conditions[t]["levels"] for t in upcodes[r["icd11_code"]]),
        axis=1)
    out = _inject(out, can_upcode, target, inject_upcoding, "is_upcoded",
                  tariff_lookup=tariffs, higher_tariff_codes=upcodes, seed=seed)

    outpatient_only = {code for code, c in conditions.items() if c.get("outpatient_only")}
    # Must match the function's own eligibility test, or the rate is computed against the
    # wrong denominator and too few claims convert.
    can_convert = ((out["claim_type"] == "outpatient") & out["icd11_code"].isin(outpatient_only)
                   & out["facility_level"].isin(per_diem))
    out = _inject(out, can_convert, target,
                  inject_outpatient_to_inpatient_conversion, "is_converted_admission",
                  outpatient_only_codes=outpatient_only, per_diem_rates=per_diem, seed=seed + 1)

    # Donor notes come from the original, untouched claims.
    donors = claims[["icd11_code", "clinical_notes"]]
    out = _inject(out, pd.Series(True, index=out.index), target, inject_ghost_billing,
                  "is_ghost_billed", donor_notes=donors, seed=seed + 2)

    out = _inject(out, pd.Series(True, index=out.index), target, inject_record_fabrication,
                  "is_fabricated_record", boilerplate_notes=boilerplate, seed=seed + 3)
    out["tariff"] = out["claim_amount"]
    return out


def derive_labels(claims: pd.DataFrame) -> pd.DataFrame:
    out = claims.copy()
    for flag in TYPOLOGY_FLAGS:
        out[flag] = out[flag].fillna(False).astype(bool) if flag in out else False
    flags = out[list(TYPOLOGY_FLAGS)]
    if (flags.sum(axis=1) > 1).any():
        raise ValueError("A claim carries more than one fraud typology")
    out["is_fraud"] = flags.any(axis=1).astype(int)
    out["fraud_typology"] = flags.idxmax(axis=1).map(TYPOLOGY_FLAGS).where(out["is_fraud"] == 1, "none")
    out["label"] = out["fraud_typology"].isin(TEXT_INCONSISTENT).map(
        {True: "inconsistent", False: "consistent"})
    return out


def split_with_pool(claims: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    """DR-08 pool first, then 60/20/20, all stratified on fraud_typology so every
    typology is represented in every partition."""
    rest, pool = train_test_split(claims, test_size=POOL_FRACTION,
                                  stratify=claims["fraud_typology"], random_state=seed)
    splits = ClaimsPreprocessor().split_dataset(rest, fraud_label_col="fraud_typology", seed=seed)
    splits["deployment_pool"] = pool
    return splits


def split_boilerplate(notes: list[str], seed: int) -> tuple[list[str], list[str]]:
    """Halves the boilerplate pool: one half for training/validation, the other reserved
    for the held-out test split and the pool, so the classifier cannot pass the test by
    having memorised fabrication notes."""
    shuffled = notes[:]
    random.Random(seed).shuffle(shuffled)
    half = len(shuffled) // 2
    return shuffled[:half], shuffled[half:]


def redraw_held_out_fabrications(splits: dict[str, pd.DataFrame], held_out: list[str],
                                 seed: int) -> dict[str, pd.DataFrame]:
    rng = random.Random(seed)
    out = dict(splits)
    for name in ("test", "deployment_pool"):
        df = out[name].copy()
        fabricated = df.index[df["is_fabricated_record"]]
        df.loc[fabricated, "clinical_notes"] = [rng.choice(held_out) for _ in fabricated]
        out[name] = df
    return out


def add_noise(splits: dict[str, pd.DataFrame], seed: int) -> dict[str, pd.DataFrame]:
    noisy = {}
    for i, (name, df) in enumerate(splits.items()):
        unseen = name in {"test", "deployment_pool"}
        noisy[name] = apply_text_noise(df, fraction=NOISE_FRACTION, rate=NOISE_RATE,
                                       seed=seed + 100 + i,
                                       abbreviations=UNSEEN_ABBREVIATIONS if unseen else None)
        noisy[name] = apply_structured_noise(noisy[name], fraction=STRUCTURED_NOISE_FRACTION,
                                             seed=seed + 200 + i)
    return noisy


def build(raw: Path, out: Path, seed: int = 42) -> dict:
    catalogue = load_catalogue()
    generator_config = json.loads((raw / "generator_config.json").read_text())
    target = round(generator_config["n_claims"] * generator_config["fraud_prevalence"]
                   / generator_config["n_typologies"])

    claims = pd.read_parquet(raw / "base_claims.parquet")
    notes = pd.read_json(raw / "notes.jsonl", lines=True)
    boilerplate_df = pd.read_json(raw / "boilerplate_notes.jsonl", lines=True)
    rejected = notes["rejected_reason"].notna()
    claims = claims.merge(notes.loc[~rejected, ["claim_id", "clinical_notes"]], on="claim_id")
    boilerplate = boilerplate_df.loc[boilerplate_df["rejected_reason"].isna(), "clinical_notes"].tolist()
    if len(boilerplate) < 20:
        raise ValueError(f"Only {len(boilerplate)} boilerplate notes passed validation; "
                         "regenerate stage B with a larger --n-boilerplate")
    seen_boilerplate, held_out_boilerplate = split_boilerplate(boilerplate, seed)

    claims = inject_claim_level_typologies(claims, catalogue, target, seen_boilerplate, seed)
    claims = derive_labels(claims)
    splits = split_with_pool(claims, seed)
    splits = redraw_held_out_fabrications(splits, held_out_boilerplate, seed)
    splits = add_noise(splits, seed)

    out.mkdir(parents=True, exist_ok=True)
    for name, df in splits.items():
        df.sort_values("claim_id").to_parquet(out / f"{name}.parquet", index=False)

    manifest = {
        "git_commit": _git_commit(),
        "seed": seed,
        "generator_config": generator_config,
        "inputs": {p.name: _sha256(p) for p in sorted(raw.iterdir()) if p.is_file()},
        "notes_rejected": int(rejected.sum()),
        "boilerplate_notes": {"accepted": len(boilerplate),
                              "rejected": int(boilerplate_df["rejected_reason"].notna().sum()),
                              "train_validation": len(seen_boilerplate),
                              "test_pool": len(held_out_boilerplate)},
        "per_typology_target": target,
        "pool_fraction": POOL_FRACTION,
        "noise": {"fraction": NOISE_FRACTION, "rate": NOISE_RATE,
                  "unseen_glossary_splits": ["test", "deployment_pool"],
                  "structured_fraction": STRUCTURED_NOISE_FRACTION},
        "splits": {
            name: {
                "claims": len(df),
                "fraud": int(df["is_fraud"].sum()),
                "inconsistent": int((df["label"] == "inconsistent").sum()),
                "text_noise": int(df["has_text_noise"].sum()),
                "structured_noise": int(df["has_structured_noise"].sum()),
                "typologies": df["fraud_typology"].value_counts().to_dict(),
            }
            for name, df in splits.items()
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = build(args.raw, args.out, args.seed)
    print(json.dumps(manifest["splits"], indent=2))


if __name__ == "__main__":
    main()
