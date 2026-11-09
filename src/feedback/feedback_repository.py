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

FEEDBACK_DECISIONS = {"approved", "rejected"}


def feedback_label(auditor_decision: str, classifier_verdict: str | None) -> str | None:
    """The classifier's training label implied by an auditor decision, in the same
    consistent/inconsistent vocabulary train.py uses. A rejection means the claim is not
    fraudulent, so its notes are consistent with the billed code. An approval confirms
    a text inconsistency only when the classifier had called the claim inconsistent: an
    approved claim the classifier called consistent was confirmed as fraud for another
    reason (a stage-1 anomaly such as an anomalous dependent), which says nothing about
    its notes, so it yields no example (None)."""
    if auditor_decision == "rejected":
        return "consistent"
    if auditor_decision == "approved" and classifier_verdict == "inconsistent":
        return "inconsistent"
    return None


class FeedbackRepository:
    def __init__(self, feedback_dataset_path: str):
        self.feedback_dataset_path = Path(feedback_dataset_path)
        self.feedback_dataset_path.parent.mkdir(parents=True, exist_ok=True)
        self._consumed_marker_path = self.feedback_dataset_path.with_suffix(".consumed")

    def materialize_example(self, claim: dict, verdict: dict, auditor_decision: str,
                            icd11_definition: str, extra: dict | None = None) -> bool:
        """auditor_decision: 'approved' (confirmed fraud) | 'rejected' (confirmed false
        positive). Writes one labeled example in train.py's input format (the exported
        JSONL of src.classification.export), so feedback can be appended to a training
        split as it is. Returns False when the decision implies no label (see
        feedback_label). Escalate produces no feedback example (FR-12 scope) — callers
        should not invoke this for escalations. `extra` adds fields to the example, e.g.
        {"simulated": True} for decisions taken from ground truth rather than an auditor."""
        if auditor_decision not in FEEDBACK_DECISIONS:
            raise ValueError(
                f"No feedback example is materialised for decision {auditor_decision!r}; "
                f"only {sorted(FEEDBACK_DECISIONS)} produce one (FR-12 scope)."
            )
        label = feedback_label(auditor_decision, verdict.get("verdict"))
        if label is None:
            return False
        example = {
            "claim_id": claim["id"] if "id" in claim else claim.get("claim_id"),
            "icd11_code": claim["icd11_code"],
            "clinical_notes": claim["clinical_notes"],
            "icd11_definition": icd11_definition,
            "label": label,
            "classifier_verdict": verdict.get("verdict"),
            "auditor_decision": auditor_decision,
            **(extra or {}),
        }
        with self.feedback_dataset_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(example) + "\n")
        return True

    def total_example_count(self) -> int:
        if not self.feedback_dataset_path.exists():
            return 0
        with self.feedback_dataset_path.open(encoding="utf-8") as f:
            return sum(1 for _ in f)

    def consumed_count(self) -> int:
        return int(self._consumed_marker_path.read_text()) if self._consumed_marker_path.exists() else 0

    def count_unconsumed_examples(self) -> int:
        """IR-06: used by ClassifierRetrainer to check FR-13's trigger threshold."""
        return self.total_example_count() - self.consumed_count()

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
        return all_examples[self.consumed_count():]

    def take_unconsumed(self) -> list[dict]:
        """The unconsumed examples, marked consumed in the same step (#48: a retraining run
        resets the trigger when it starts). The marker moves only past the lines read here,
        so an example appended while the run trains is left for the next run."""
        if not self.feedback_dataset_path.exists():
            return []
        with self.feedback_dataset_path.open(encoding="utf-8") as f:
            all_examples = [json.loads(line) for line in f]
        examples = all_examples[self.consumed_count():]
        self.mark_consumed(up_to_count=len(all_examples))
        return examples
