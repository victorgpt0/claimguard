"""FeedbackRepository (FR-12; DR-08). Reads Claim and AuditReport rows from SQLite to
build each labeled feedback example (clinical notes, billed ICD-11 code, auditor's
approve/reject determination), and writes the result to the file-based Feedback Dataset
— NOT back into SQLite. Kept structurally distinct from the training/validation/held-out
test subsets (DR-03) to prevent the retraining loop from contaminating reported metrics.

Only claims from the Deployment/Demonstration Pool ever reach this repository, since only
those are routed through the live dashboard (Section 3.2.2's fourth data partition).

Stored as append-only JSONL, one labeled example per line, plus a sidecar ``.consumed``
marker file recording how many lines ClassifierRetrainer has already accounted for —
this is what lets count_unconsumed_examples() answer FR-13's trigger-threshold check
without re-reading (and re-counting) the whole dataset's history every time.
"""
from __future__ import annotations

import json
from pathlib import Path

DECISION_TO_LABEL = {"approved": "confirmed_fraud", "rejected": "confirmed_false_positive"}


class FeedbackRepository:
    def __init__(self, feedback_dataset_path: str):
        self.feedback_dataset_path = Path(feedback_dataset_path)
        self.feedback_dataset_path.parent.mkdir(parents=True, exist_ok=True)
        self._consumed_marker_path = self.feedback_dataset_path.with_suffix(".consumed")

    def materialize_example(self, claim: dict, verdict: dict, auditor_decision: str) -> None:
        """auditor_decision: 'approved' (confirmed fraud) | 'rejected' (confirmed false
        positive). Writes one labeled example to the file-based Feedback Dataset.
        Escalate produces no feedback example (FR-12 scope) — callers should not invoke
        this for escalations."""
        if auditor_decision not in DECISION_TO_LABEL:
            raise ValueError(
                f"No feedback example is materialised for decision {auditor_decision!r}; "
                f"only {sorted(DECISION_TO_LABEL)} produce one (FR-12 scope)."
            )
        example = {
            "claim_id": claim["id"] if "id" in claim else claim.get("claim_id"),
            "clinical_notes": claim["clinical_notes"],
            "icd11_code": claim["icd11_code"],
            "classifier_verdict": verdict.get("verdict"),
            "label": DECISION_TO_LABEL[auditor_decision],
        }
        with self.feedback_dataset_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(example) + "\n")

    def total_example_count(self) -> int:
        if not self.feedback_dataset_path.exists():
            return 0
        with self.feedback_dataset_path.open(encoding="utf-8") as f:
            return sum(1 for _ in f)

    def count_unconsumed_examples(self) -> int:
        """IR-06: used by ClassifierRetrainer to check FR-13's trigger threshold."""
        consumed = int(self._consumed_marker_path.read_text()) if self._consumed_marker_path.exists() else 0
        return self.total_example_count() - consumed

    def mark_consumed(self, up_to_count: int | None = None) -> None:
        """Called by ClassifierRetrainer once it has kicked off a retraining job for the
        currently-unconsumed examples, so the next check doesn't re-trigger on the same
        examples."""
        count = up_to_count if up_to_count is not None else self.total_example_count()
        self._consumed_marker_path.write_text(str(count))

    def read_examples(self, since_consumed_marker: bool = True) -> list[dict]:
        """Returns the examples a retraining run should actually train on — everything
        unconsumed by default."""
        if not self.feedback_dataset_path.exists():
            return []
        with self.feedback_dataset_path.open(encoding="utf-8") as f:
            all_examples = [json.loads(line) for line in f]
        if not since_consumed_marker:
            return all_examples
        consumed = int(self._consumed_marker_path.read_text()) if self._consumed_marker_path.exists() else 0
        return all_examples[consumed:]
