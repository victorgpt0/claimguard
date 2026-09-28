"""SemanticConsistencyClassifier (FR-04; Section 3.6.1 Class Diagram). A fine-tuned
multilingual transformer encoder (AfroXLM-R, or XLM-RoBERTa-base fallback — Conneau et
al., 2020; Alabi et al., 2022) accepting the clinical notes string and billed ICD-11 code
definition, producing a binary consistency verdict with a confidence score, learned via
transfer learning rather than rule application.

This is the detector the supervisor's feedback asked for: "let AI detect through transfer
learning" — not a prompted LLM. See src/narrative/narrative_generator.py for the
downstream, explanation-only LLM layer this classifier gates.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABELS = ["consistent", "inconsistent"]


@dataclass
class ClassifierVerdict:
    claim_id: str
    verdict: str  # "consistent" | "inconsistent"
    confidence: float
    model_version: str


class SemanticConsistencyClassifier:
    def __init__(self, model_path: str, base_model_name: str = "xlm-roberta-base",
                 device: str = "cpu"):
        """`model_path` points at a LoRA adapter checkpoint directory (see train.py),
        loaded on top of `base_model_name`. If `model_path` has no adapter yet (e.g. a
        fresh checkout before any fine-tuning has run), falls back to the untrained base
        model — predictions will be meaningless until a real checkpoint exists, but the
        interface works end-to-end for wiring/testing purposes."""
        self.model_path = Path(model_path)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name, num_labels=len(LABELS)
        )
        if self.model_path.exists() and any(self.model_path.iterdir()):
            self.model = PeftModel.from_pretrained(base_model, str(self.model_path))
            self.model_version = self.model_path.name
        else:
            self.model = base_model
            self.model_version = "untrained-base"
        self.model.to(device).eval()

    def predict(self, claim_id: str, clinical_notes: str, icd11_definition: str) -> ClassifierVerdict:
        """Binary sequence-pair classification: (clinical justification, ICD-11 code
        definition) -> consistent/inconsistent + confidence. NFR-05: recall floor 0.80 on
        the held-out test set."""
        inputs = self.tokenizer(
            clinical_notes, icd11_definition,
            truncation=True, padding=True, max_length=512, return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1)[0]
        predicted_idx = int(torch.argmax(probabilities))
        return ClassifierVerdict(
            claim_id=claim_id,
            verdict=LABELS[predicted_idx],
            confidence=float(probabilities[predicted_idx]),
            model_version=self.model_version,
        )

    @staticmethod
    def load_metrics(model_path: str) -> dict | None:
        """Reads the metrics.json a training run writes alongside its checkpoint (see
        train.py), used by ClassifierRetrainer's promotion gate (FR-14)."""
        metrics_path = Path(model_path) / "metrics.json"
        if not metrics_path.exists():
            return None
        return json.loads(metrics_path.read_text(encoding="utf-8"))
