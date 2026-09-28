"""IsolationForestDetector (FR-02; Section 3.2.3). Tree-ensemble anomaly detector trained
unsupervised on 80% of the structured data stream; contamination hyperparameter tuned
across [0.05, 0.40] via grid search on the validation split, selected by F1-score.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score

from src.detection.base import BaseAnomalyDetector


class IsolationForestDetector(BaseAnomalyDetector):
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(contamination=contamination, random_state=random_state)

    def fit(self, feature_matrix: np.ndarray) -> "IsolationForestDetector":
        self.model.fit(feature_matrix)
        return self

    def score(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Higher = more anomalous (sklearn's own score_samples is inverted: lower/more
        negative = more anomalous, so we flip sign here for a consistent convention
        across both detectors in this module)."""
        return -self.model.score_samples(feature_matrix)

    @staticmethod
    def tune_contamination(train_features: np.ndarray, val_features: np.ndarray,
                            val_labels: np.ndarray, candidates: list[float] | None = None,
                            random_state: int = 42) -> tuple[float, "IsolationForestDetector"]:
        """Grid search over the contamination hyperparameter (Section 3.2.3: range
        [0.05, 0.40]), selecting by F1-score on the validation split."""
        candidates = candidates or [round(0.05 * i, 2) for i in range(1, 9)]  # 0.05..0.40
        best_f1, best_contamination, best_model = -1.0, candidates[0], None
        for contamination in candidates:
            detector = IsolationForestDetector(contamination=contamination, random_state=random_state)
            detector.fit(train_features)
            predicted_anomaly = detector.model.predict(val_features) == -1  # sklearn: -1 anomalous
            f1 = f1_score(val_labels, predicted_anomaly)
            if f1 > best_f1:
                best_f1, best_contamination, best_model = f1, contamination, detector
        return best_contamination, best_model
