import numpy as np
from unittest.mock import MagicMock, patch

from src.explainability.shap_explainer import SHAPExplainer


def test_explain_returns_top_k_features_by_absolute_value():
    fake_detector = MagicMock()
    fake_detector.score.return_value = np.array([0.5])
    feature_names = ["claim_amount", "stay_days", "code_frequency", "claim_amount_per_day"]
    background = np.zeros((5, 4))

    with patch("src.explainability.shap_explainer.shap.KernelExplainer") as MockExplainer:
        instance = MockExplainer.return_value
        instance.shap_values.return_value = np.array([[0.1, -0.9, 0.05, 0.02]])

        explainer = SHAPExplainer(fake_detector, feature_names, background)
        result = explainer.explain(np.zeros((1, 4)), top_k=2)

    assert len(result) == 1
    assert set(result[0].keys()) == {"stay_days", "claim_amount"}
    assert result[0]["stay_days"] == -0.9
