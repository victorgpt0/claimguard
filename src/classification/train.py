"""LoRA/PEFT fine-tuning script for the Semantic Consistency Classifier (Section 3.2.3,
Section 3.8.6). Fine-tunes across 5 random seeds (config.yaml training.random_seeds) per
the rigor-improvements patch (Block K): a single training run cannot distinguish a
genuinely better classifier from a lucky seed (Dodge et al., 2020).

For each seed, writes experiments/<run_id>/config.json (hyperparameters, seed, base
model, LoRA config, final validation/test metrics) per the implementation roadmap's
Section 2a — this is what Chapter 5 (research_writing.txt 5.3.2) means by "retain
configuration files and experiment logs so the reported run can be reproduced."
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
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
        encoding = self.tokenizer(
            example["clinical_notes"], example["icd11_definition"],
            truncation=True, padding="max_length", max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(LABELS.index(example["label"]))
        return item


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_seed(seed: int, train_examples: list[dict], val_examples: list[dict],
                    base_model_name: str, output_dir: Path, epochs: int = 3,
                    batch_size: int = 16, lr: float = 2e-4,
                    lora_config: dict | None = None, device: str = "cpu") -> dict:
    """Fine-tune the classifier for a single seed. Returns the run's validation metrics,
    which the caller writes to experiments/<run_id>/config.json alongside the
    hyperparameters."""
    set_seed(seed)
    lora_config = lora_config or DEFAULT_LORA_CONFIG

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name, num_labels=len(LABELS)
    )
    model = get_peft_model(base_model, LoraConfig(**lora_config)).to(device)

    train_loader = DataLoader(ClaimPairDataset(train_examples, tokenizer),
                               batch_size=batch_size, shuffle=True)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    best_val_f1, epochs_without_improvement, patience = -1.0, 0, 1
    for epoch in range(epochs):
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimiser.zero_grad()
            outputs = model(**batch)
            outputs.loss.backward()
            optimiser.step()

        val_metrics = evaluate(model, val_examples, tokenizer, device)
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            epochs_without_improvement = 0
            model.save_pretrained(output_dir)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break  # early stopping on validation F1-score (Section 3.2.3)

    return evaluate(model, val_examples, tokenizer, device)


def evaluate(model, examples: list[dict], tokenizer, device: str) -> dict:
    model.eval()
    dataset = ClaimPairDataset(examples, tokenizer)
    loader = DataLoader(dataset, batch_size=32)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            all_preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            all_labels.extend(labels.tolist())
    model.train()
    return {
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
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
    args = parser.parse_args()

    train_ex = [json.loads(line) for line in open(args.train_file)]
    val_ex = [json.loads(line) for line in open(args.val_file)]
    train_all_seeds(train_ex, val_ex, args.base_model, args.seeds, Path(args.experiments_dir))
