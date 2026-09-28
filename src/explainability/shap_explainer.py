"""SHAPExplainer (FR-03; Section 3.8.9). Accepts the anomaly detector's output and the
feature matrix, produces a SHAP attribution dictionary identifying which structured
billing attributes most contributed to a claim's anomaly score.
"""
from __future__ import annotations

import numpy as np
import shap


class SHAPExplainer:
    def __init__(self, detector, feature_names: list[str], background_data: np.ndarray):
        """`detector` must expose `.score(feature_matrix) -> np.ndarray` (both
        IsolationForestDetector and AutoencoderDetector do). `background_data` is a
        representative sample of training features used to estimate SHAP's baseline —
        KernelExplainer works model-agnostically against either detector's score()."""
        self.feature_names = feature_names
        self._explainer = shap.KernelExplainer(detector.score, background_data)

    def explain(self, feature_matrix: np.ndarray, top_k: int = 3) -> list[dict]:
        """Returns one attribution dict per claim: {feature_name: shap_value}, restricted
        to the top_k features by absolute contribution (Figure 4.3's "SHAP attribution
        per flagged claim")."""
        shap_values = self._explainer.shap_values(feature_matrix)
        results = []
        for row in shap_values:
            top_indices = np.argsort(-np.abs(row))[:top_k]
            results.append({self.feature_names[i]: float(row[i]) for i in top_indices})
        return results
