"""SHAPExplainer (FR-03; Section 3.8.9). Accepts the anomaly detector's output and the
feature matrix, produces a SHAP attribution dictionary identifying which structured
billing attributes most contributed to a claim's anomaly score.
"""


class SHAPExplainer:
    def __init__(self, detector):
        self.detector = detector

    def explain(self, feature_matrix) -> dict:
        raise NotImplementedError
