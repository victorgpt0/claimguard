"""LoRA/PEFT fine-tuning script for the Semantic Consistency Classifier (Section 3.2.3,
Section 3.8.6). Fine-tunes across 5 random seeds (config.yaml training.random_seeds) per
the rigor-improvements patch (Block K): a single training run cannot distinguish a
genuinely better classifier from a lucky seed (Dodge et al., 2020).

For each seed, writes experiments/<run_id>/config.json (hyperparameters, seed, base
model, LoRA config, validation metrics of the saved checkpoint) per the implementation
roadmap's Section 2a — this is what Chapter 5 (research_writing.txt 5.3.2) means by
"retain configuration files and experiment logs so the reported run can be reproduced."
Only the training and validation files are read: the held-out test set is reserved for
the final evaluation (Section 3.2.4).

Inconsistent claims are about 5% of the data, so the loss is weighted by inverse class
frequency by default; without it the model can score well by always answering
"consistent".
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from src.classification.classifier import LABELS

DEFAULT_LORA_CONFIG = dict(r=8, lora_alpha=16, lora_dropout=0.1,
                            target_modules=["query", "value"], task_type="SEQ_CLS")


class ClaimPairDataset(Dataset):
    """clinical_notes + icd11_definition -> consistent/inconsistent label."""

    def __init__(self, examples: list[dict], tokenizer, max_length: int = 512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        # Padding happens per batch (DataCollatorWithPadding): note + definition is
        # ~170 tokens typically, so padding everything to 512 would waste most compute.
        item = dict(self.tokenizer(example["clinical_notes"], example["icd11_definition"],
                                   truncation=True, max_length=self.max_length))
        item["labels"] = LABELS.index(example["label"])
        return item


def _loader(examples: list[dict], tokenizer, batch_size: int, shuffle: bool = False,
            max_length: int = 512) -> DataLoader:
    return DataLoader(ClaimPairDataset(examples, tokenizer, max_length), batch_size=batch_size,
                      shuffle=shuffle, collate_fn=DataCollatorWithPadding(tokenizer))


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def class_weights(examples: list[dict]) -> torch.Tensor:
    """Inverse-frequency ("balanced", as in scikit-learn) weights: n / (classes x count),
    so each class contributes equally to the loss."""
    counts = np.array([sum(e["label"] == label for e in examples) for label in LABELS], dtype=float)
    weights = counts.sum() / (len(LABELS) * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_seed(seed: int, train_examples: list[dict], val_examples: list[dict],
                    base_model_name: str, output_dir: Path, epochs: int = 5,
                    batch_size: int = 16, lr: float = 2e-4, patience: int = 2,
                    max_length: int = 512, weighted_loss: bool = True,
                    lora_config: dict | None = None, device: str | None = None) -> dict:
    """Fine-tune the classifier for a single seed, saving the checkpoint with the best
    validation F1 to `output_dir`. Returns that checkpoint's validation metrics (not the
    last epoch's), which the caller writes to experiments/<run_id>/config.json alongside
    the hyperparameters. `device` defaults to the GPU when one is available."""
    set_seed(seed)
    device = device or default_device()
    lora_config = lora_config or DEFAULT_LORA_CONFIG

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name, num_labels=len(LABELS)
    )
    model = get_peft_model(base_model, LoraConfig(**lora_config)).to(device)

    train_loader = _loader(train_examples, tokenizer, batch_size, shuffle=True, max_length=max_length)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr)
    weights = class_weights(train_examples).to(device) if weighted_loss else None
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    model.train()
    best, epochs_without_improvement = None, 0
    for epoch in range(1, epochs + 1):
        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            optimiser.zero_grad()
            loss = loss_fn(model(**batch).logits, labels)
            loss.backward()
            optimiser.step()

        val_metrics = evaluate(model, val_examples, tokenizer, device, max_length=max_length)
        print(f"seed {seed} epoch {epoch}: " + ", ".join(f"{k} {v:.4f}" for k, v in val_metrics.items()
                                                       if isinstance(v, float)))
        if best is None or val_metrics["f1"] > best["f1"]:
            best = {**val_metrics, "epoch": epoch}
            epochs_without_improvement = 0
            model.save_pretrained(output_dir)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break  # early stopping on validation F1-score (Section 3.2.3)

    # The metrics of the checkpoint actually saved, not of the final in-memory weights,
    # which may be a worse, later epoch.
    return best


def predict_probabilities(model, examples: list[dict], tokenizer, device: str,
                          max_length: int = 512) -> list[float]:
    """Probability of "inconsistent" for each example, in order."""
    model.eval()
    probabilities = []
    with torch.no_grad():
        for batch in _loader(examples, tokenizer, batch_size=32, max_length=max_length):
            batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            probabilities.extend(torch.softmax(logits, dim=-1)[:, LABELS.index("inconsistent")].cpu().tolist())
    model.train()
    return probabilities


def evaluate(model, examples: list[dict], tokenizer, device: str, max_length: int = 512) -> dict:
    """Precision, recall and F1 for the "inconsistent" class at the argmax decision."""
    probabilities = predict_probabilities(model, examples, tokenizer, device, max_length)
    preds = [int(p > 0.5) for p in probabilities]
    labels = [LABELS.index(e["label"]) for e in examples]
    return {
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "flagged": int(sum(preds)),
        "positives": int(sum(labels)),
    }


def train_all_seeds(train_examples: list[dict], val_examples: list[dict],
                     base_model_name: str, seeds: list[int], experiments_dir: Path,
                     **train_kwargs) -> list[dict]:
    """Runs train_one_seed for every seed; writes experiments/<run_id>/config.json per
    run so mean +/- std can be computed and reported (Section 3.2.4), and the reported
    run can be reproduced (research_writing.txt 5.3.2)."""
    results = []
    for seed in seeds:
        run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_seed{seed}"
        run_dir = experiments_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        metrics = train_one_seed(seed, train_examples, val_examples, base_model_name,
                                  output_dir=run_dir, **train_kwargs)
        run_record = {
            "run_id": run_id, "seed": seed, "base_model": base_model_name,
            "lora_config": train_kwargs.get("lora_config", DEFAULT_LORA_CONFIG),
            "hyperparameters": {k: v for k, v in train_kwargs.items() if k != "lora_config"},
            "train_examples": len(train_examples), "validation_examples": len(val_examples),
            "metrics": metrics,
        }
        (run_dir / "config.json").write_text(json.dumps(run_record, indent=2))
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        results.append(run_record)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", required=True, help="JSONL: clinical_notes, icd11_definition, label")
    parser.add_argument("--val-file", required=True)
    parser.add_argument("--base-model", default="emilyalsentzer/Bio_ClinicalBERT",
                         help="Primary choice per config.yaml. There is no automatic "
                              "fallback — pass --base-model bert-base-uncased explicitly "
                              "if Bio_ClinicalBERT fails to load.")
    parser.add_argument("--experiments-dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024, 7, 31415])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--no-weighted-loss", action="store_true",
                        help="plain cross-entropy instead of inverse-frequency class weights")
    args = parser.parse_args()

    train_ex = [json.loads(line) for line in open(args.train_file)]
    val_ex = [json.loads(line) for line in open(args.val_file)]
    print(f"device: {default_device()}; {len(train_ex)} train, {len(val_ex)} validation examples")
    results = train_all_seeds(train_ex, val_ex, args.base_model, args.seeds, Path(args.experiments_dir),
                              epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                              patience=args.patience, max_length=args.max_length,
                              weighted_loss=not args.no_weighted_loss)
    for result in results:
        m = result["metrics"]
        print(f"{result['run_id']}: epoch {m['epoch']}, precision {m['precision']:.4f}, "
              f"recall {m['recall']:.4f}, f1 {m['f1']:.4f}")
