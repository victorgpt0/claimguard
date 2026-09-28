"""IsolationForestDetector (FR-02; Section 3.2.3). Tree-ensemble anomaly detector trained
unsupervised on 80% of the structured data stream; contamination hyperparameter tuned
across [0.05, 0.40] via grid search on the validation split, selected by F1-score.
"""
from src.detection.base import BaseAnomalyDetector


class IsolationForestDetector(BaseAnomalyDetector):
    def __init__(self, contamination: float | None = None):
        self.contamination = contamination  # tuned via validation-split grid search
        self.model = None

    def fit(self, feature_matrix):
        raise NotImplementedError

    def score(self, feature_matrix):
        raise NotImplementedError
