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


class ClassifierRetrainer:
    def __init__(self, feedback_repo, active_model_path: str, candidate_dir: str):
        self.feedback_repo = feedback_repo
        self.active_model_path = active_model_path
        self.candidate_dir = candidate_dir

    def maybe_trigger_retraining(self, trigger_count: int):
        """IR-06: poll-on-event check appended to
        AuditReportRepository.record_decision — not a standing scheduler."""
        raise NotImplementedError

    def evaluate_and_promote(self, candidate_checkpoint_path: str, held_out_test_set):
        """FR-14: promote only if candidate F1/recall >= active checkpoint's."""
        raise NotImplementedError
