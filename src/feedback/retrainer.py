"""ClassifierRetrainer (FR-13, FR-14; NFR-05). The only class with write access to the
Semantic Consistency Classifier's deployed weights — no class in the Online Inference
Pipeline can update the classifier directly, keeping the promotion gate as the single
place a new model version enters production.

FR-13: triggers a retraining job once the Feedback Dataset accumulates
config.yaml's retrain_trigger_count new labeled examples since the last run.
FR-14: evaluates the resulting candidate checkpoint against the SAME fixed held-out test
set from Section 3.2.4, and promotes it to models/classifier_lora/active/ only if its
F1-score and recall are not worse than the currently active checkpoint's — a candidate
that scores worse is retained in models/classifier_lora/candidates/ for inspection but
never deployed. This is the "never silently make it worse" guarantee.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

from src.feedback.feedback_repository import FeedbackRepository

logger = logging.getLogger(__name__)


class ClassifierRetrainer:
    def __init__(self, feedback_repo: FeedbackRepository, active_model_path: str,
                 candidate_dir: str):
        self.feedback_repo = feedback_repo
        self.active_model_path = Path(active_model_path)
        self.candidate_dir = Path(candidate_dir)
        self.candidate_dir.mkdir(parents=True, exist_ok=True)
        # The metrics behind the last evaluate_and_promote decision, for the retraining
        # history (#47): {candidate_metrics, active_metrics (None if nothing was
        # deployed), promoted}.
        self.last_result: dict | None = None

    def maybe_trigger_retraining(self, trigger_count: int) -> bool:
        """IR-06: poll-on-event check appended to
        AuditReportRepository.record_decision — not a standing scheduler. Returns True
        if a retraining job was (would be) triggered. Actually launching the training
        job (a potentially long-running, GPU-bound process) is left to the caller —
        e.g. a Colab notebook invoking src.classification.train — rather than run
        synchronously inside a web request."""
        unconsumed = self.feedback_repo.count_unconsumed_examples()
        should_trigger = unconsumed >= trigger_count
        if should_trigger:
            logger.info("Retraining trigger reached: %d unconsumed feedback examples "
                        "(threshold %d).", unconsumed, trigger_count)
        return should_trigger

    def evaluate_and_promote(self, candidate_checkpoint_path: str,
                              held_out_test_examples: list[dict]) -> bool:
        """FR-14: promote only if candidate F1/recall >= active checkpoint's, both
        measured on the SAME fixed held-out test set. Returns True if promoted; the
        metrics compared are kept in `last_result`."""
        # Imported here so the dashboard, which only runs the trigger check, never loads
        # torch and transformers.
        from src.classification.classifier import SemanticConsistencyClassifier

        candidate = SemanticConsistencyClassifier(candidate_checkpoint_path)
        candidate_metrics = self._evaluate_on_test_set(candidate, held_out_test_examples)

        if self.active_model_path.exists() and any(self.active_model_path.iterdir()):
            active = SemanticConsistencyClassifier(str(self.active_model_path))
            active_metrics = self._evaluate_on_test_set(active, held_out_test_examples)
            not_worse = (candidate_metrics["f1"] >= active_metrics["f1"]
                         and candidate_metrics["recall"] >= active_metrics["recall"])
        else:
            active_metrics = None
            not_worse = True  # nothing deployed yet — promote unconditionally
        self.last_result = {"candidate_metrics": candidate_metrics,
                            "active_metrics": active_metrics, "promoted": not_worse}

        if not_worse:
            self._promote(candidate_checkpoint_path)
            logger.info("Promoted candidate %s (F1=%.3f, recall=%.3f) over active %s.",
                        candidate_checkpoint_path, candidate_metrics["f1"],
                        candidate_metrics["recall"], active_metrics or "(none deployed)")
        else:
            self._retain_for_inspection(candidate_checkpoint_path)
            logger.info("Candidate %s scored worse than active — retained for "
                        "inspection, NOT deployed.", candidate_checkpoint_path)
        return not_worse

    def _evaluate_on_test_set(self, classifier: SemanticConsistencyClassifier,
                               examples: list[dict]) -> dict:
        from sklearn.metrics import f1_score, recall_score

        predictions, labels = [], []
        for example in examples:
            verdict = classifier.predict(
                example.get("claim_id", ""), example["clinical_notes"], example["icd11_definition"]
            )
            predictions.append(verdict.verdict)
            labels.append(example["label"])
        return {
            "f1": float(f1_score(labels, predictions, pos_label="inconsistent", zero_division=0)),
            "recall": float(recall_score(labels, predictions, pos_label="inconsistent", zero_division=0)),
        }

    def _promote(self, candidate_checkpoint_path: str) -> None:
        if self.active_model_path.exists():
            shutil.rmtree(self.active_model_path)
        shutil.copytree(candidate_checkpoint_path, self.active_model_path)

    def _retain_for_inspection(self, candidate_checkpoint_path: str) -> None:
        destination = self.candidate_dir / Path(candidate_checkpoint_path).name
        if destination != Path(candidate_checkpoint_path):
            shutil.copytree(candidate_checkpoint_path, destination, dirs_exist_ok=True)
