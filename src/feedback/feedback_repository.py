"""FeedbackRepository (FR-12; DR-08). Reads Claim and AuditReport rows from SQLite to
build each labeled feedback example (clinical notes, billed ICD-11 code, auditor's
approve/reject determination), and writes the result to the file-based Feedback Dataset
— NOT back into SQLite. Kept structurally distinct from the training/validation/held-out
test subsets (DR-03) to prevent the retraining loop from contaminating reported metrics.

Only claims from the Deployment/Demonstration Pool ever reach this repository, since only
those are routed through the live dashboard (Section 3.2.2's fourth data partition).
"""


class FeedbackRepository:
    def __init__(self, feedback_dataset_path: str):
        self.feedback_dataset_path = feedback_dataset_path

    def materialize_example(self, claim, verdict, auditor_decision: str):
        """auditor_decision: 'approve' (confirmed fraud) | 'reject' (confirmed false
        positive). Writes one labeled example to the file-based Feedback Dataset."""
        raise NotImplementedError

    def count_unconsumed_examples(self) -> int:
        """IR-06: used by ClassifierRetrainer to check FR-13's trigger threshold."""
        raise NotImplementedError
