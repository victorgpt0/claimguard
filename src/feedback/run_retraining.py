"""Feedback-driven retraining run (FR-12 to FR-14; task 6.5, GitHub #47, #48). Run
manually on Colab once the dashboard reports the retraining trigger:

  1. take the unconsumed Feedback Dataset examples and mark them consumed at the start,
     so the trigger resets even if training later fails (#48);
  2. append them to the exported training split (src.classification.export's JSONL, the
     format FeedbackRepository writes) and train one candidate with train_one_seed into
     <model_dir>/classifier_lora/candidates/<timestamp>/;
  3. let ClassifierRetrainer.evaluate_and_promote gate it on the held-out test split:
     promoted to active/ only if F1 and recall are not worse than the active checkpoint's;
  4. append the outcome to <experiments_dir>/retraining_history.jsonl (#47).

The test split is used only as the FR-14 pass/fail gate; Chapter 5 reports the frozen
checkpoint chosen on validation, not these metrics (remaining_work_plan.md D5).

--simulate-auditor N first decides N pending reports from the ground-truth label, for
when real review cannot reach the trigger in time. Every simulated example carries
"simulated": true, and Chapter 5 must say the decisions were simulated.

    python -m src.feedback.run_retraining --classifier-data data/classifier/v0.5-5k
    python -m src.feedback.run_retraining --classifier-data data/classifier/v0.5-5k \\
        --simulate-auditor 60 --data data/processed/v0.5-5k
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.feedback.feedback_repository import FeedbackRepository
from src.feedback.retrainer import ClassifierRetrainer

DEFAULT_BASE_MODEL = "emilyalsentzer/Bio_ClinicalBERT"


def feedback_path(config: dict) -> Path:
    """The same Feedback Dataset the dashboard writes (dashboard/app.py)."""
    return Path(config["paths"]["artifacts_dir"]) / "feedback_dataset" / "examples.jsonl"


def classifier_dir(config: dict) -> Path:
    return Path(config["paths"]["model_dir"]) / "classifier_lora"


def history_path(config: dict) -> Path:
    return Path(config["paths"].get("experiments_dir", "experiments")) / "retraining_history.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_history(path: Path) -> list[dict]:
    """Every retraining run so far, oldest first; empty before the first run."""
    path = Path(path)
    return load_jsonl(path) if path.exists() else []


def _append_history(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def run(config: dict, data_dir: Path, seed: int = 42, trainer=None, **train_kwargs) -> dict | None:
    """One retraining run. `data_dir` holds the exported classifier JSONL (train,
    validation, test). `trainer` defaults to src.classification.train.train_one_seed
    (imported lazily, so torch loads only when a run actually trains). Returns the
    history row, or None when there is no unconsumed feedback."""
    data_dir = Path(data_dir)
    feedback = FeedbackRepository(str(feedback_path(config)))
    examples = feedback.take_unconsumed()
    if not examples:
        return None

    if trainer is None:
        from src.classification.train import train_one_seed as trainer

    run_at = datetime.now(timezone.utc)
    base_model = config.get("models", {}).get("classifier_base_model", DEFAULT_BASE_MODEL)
    train = load_jsonl(data_dir / "train.jsonl") + examples
    validation = load_jsonl(data_dir / "validation.jsonl")
    test = load_jsonl(data_dir / "test.jsonl")

    models = classifier_dir(config)
    candidate = models / "candidates" / run_at.strftime("%Y%m%dT%H%M%SZ")
    candidate.mkdir(parents=True, exist_ok=True)
    validation_metrics = trainer(seed, train, validation, base_model, output_dir=candidate,
                                 **train_kwargs)
    # The same records train_all_seeds writes, so the dashboard's model tab can read a
    # promoted candidate like any other checkpoint.
    record = {"run_id": f"retrain_{candidate.name}", "seed": seed, "base_model": base_model,
              "train_examples": len(train), "validation_examples": len(validation),
              "feedback_examples": len(examples), "metrics": validation_metrics}
    (candidate / "config.json").write_text(json.dumps(record, indent=2))
    (candidate / "metrics.json").write_text(json.dumps(validation_metrics, indent=2))

    retrainer = ClassifierRetrainer(feedback, active_model_path=str(models / "active"),
                                    candidate_dir=str(models / "candidates"))
    promoted = retrainer.evaluate_and_promote(str(candidate), test)
    result = retrainer.last_result or {}
    row = {
        "run_at": run_at.isoformat(),
        "feedback_examples": len(examples),
        "simulated_examples": sum(1 for e in examples if e.get("simulated")),
        "candidate_path": str(candidate),
        "candidate_validation_metrics": validation_metrics,
        "candidate_metrics": result.get("candidate_metrics"),
        "active_metrics": result.get("active_metrics"),
        "promoted": bool(promoted),
    }
    _append_history(history_path(config), row)
    return row


def _truth_key(provider_id, member_id, code, service_date, notes) -> tuple:
    return (str(provider_id), str(member_id), str(code), str(service_date)[:10], str(notes))


def simulate_decisions(db_path: str, data_dir: Path, split: str, n: int,
                       feedback_dataset: Path, cache_path: Path) -> dict:
    """Decides up to `n` pending reports from the ground truth, as an auditor would:
    rejected if the claim is not fraud, approved if it is. Records each decision and its
    feedback example the way dashboard/app.py's record_decision does, tagging every
    example "simulated": true. Reports are matched to the split's claims by the stored
    claim_ref; databases written before claim_ref existed fall back to provider, member,
    billed code, service date and notes. A report that matches no claim, or claims with
    different labels, is left pending."""
    from src.classification.export import reference_text
    from src.data.preprocessing import repair_icd11_code
    from src.icd11.cache import ICD11Cache
    from src.reporting.repository import AuditReportRepository

    claims = pd.read_parquet(Path(data_dir) / f"{split}.parquet")
    truth: dict[tuple, set] = {}
    claim_ids: dict[tuple, str] = {}
    for claim in claims.itertuples(index=False):
        key = _truth_key(claim.provider_id, claim.member_id, repair_icd11_code(claim.icd11_code),
                         claim.service_date, claim.clinical_notes)
        truth.setdefault(key, set()).add(int(claim.is_fraud))
        claim_ids[key] = claim.claim_id
    truth_by_ref = {str(c): int(f) for c, f in zip(claims["claim_id"], claims["is_fraud"])}

    repository = AuditReportRepository(db_path)
    feedback = FeedbackRepository(str(feedback_dataset))
    cache = ICD11Cache(cache_path)
    counts = {"decided": 0, "approved": 0, "rejected": 0, "examples": 0, "unmatched": 0}
    for pending in repository.get_pending_reports():
        if counts["decided"] >= n:
            break
        report = repository.get_report(pending["id"])
        if report.get("claim_ref") is not None:
            ref = str(report["claim_ref"])
            labels = {truth_by_ref[ref]} if ref in truth_by_ref else set()
        else:
            key = _truth_key(report["provider_id"], report["patient_ref"], report["icd11_code"],
                             report["service_date"], report["clinical_notes"])
            labels = truth.get(key, set())
            ref = claim_ids.get(key)
        if len(labels) != 1:
            counts["unmatched"] += 1
            continue
        decision = "approved" if next(iter(labels)) == 1 else "rejected"
        repository.record_decision(report["id"], decision)
        written = feedback.materialize_example(
            claim={"id": report["claim_id"], "clinical_notes": report["clinical_notes"],
                   "icd11_code": report["icd11_code"]},
            verdict={"verdict": report.get("verdict")},
            auditor_decision=decision,
            icd11_definition=reference_text(cache.get(report["icd11_code"])),
            extra={"simulated": True, "claim_ref": ref},
        )
        counts["decided"] += 1
        counts[decision] += 1
        counts["examples"] += int(written)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--classifier-data", type=Path, required=True,
                        help="exported classifier JSONL directory (train/validation/test)")
    parser.add_argument("--config", type=Path, help="config.yaml (default: config/config.yaml)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--simulate-auditor", type=int, metavar="N",
                        help="first decide N pending reports from the ground-truth label")
    parser.add_argument("--data", type=Path, help="processed dataset directory (with --simulate-auditor)")
    parser.add_argument("--split", default="deployment_pool",
                        help="split the reports were audited from (with --simulate-auditor)")
    parser.add_argument("--cache", type=Path, help="ICD-11 cache JSON (default: <artifacts_dir>/icd11_cache.json)")
    parser.add_argument("--simulate-only", action="store_true",
                        help="record the simulated decisions without retraining")
    args = parser.parse_args()

    from src.config import load_config
    config = load_config(args.config)

    if args.simulate_auditor:
        if args.data is None:
            parser.error("--simulate-auditor needs --data")
        cache = args.cache or Path(config["paths"]["artifacts_dir"]) / "icd11_cache.json"
        counts = simulate_decisions(config["paths"]["db_path"], args.data, args.split,
                                    args.simulate_auditor, feedback_path(config), cache)
        print(f"Simulated decisions (ground truth, not an auditor): {counts}")
        if args.simulate_only:
            return

    row = run(config, args.classifier_data, seed=args.seed, epochs=args.epochs,
              batch_size=args.batch_size)
    if row is None:
        print("No unconsumed feedback examples; nothing to retrain.")
        return
    print(json.dumps(row, indent=2))
    print(f"-> {history_path(config)}")


if __name__ == "__main__":
    main()
