import numpy as np

from src.detection.isolation_forest import IsolationForestDetector


def _synthetic_features(n_normal=100, n_anomalous=10, seed=0):
    rng = np.random.default_rng(seed)
    normal = rng.normal(loc=0, scale=1, size=(n_normal, 3))
    anomalous = rng.normal(loc=8, scale=1, size=(n_anomalous, 3))
    features = np.vstack([normal, anomalous])
    labels = np.array([0] * n_normal + [1] * n_anomalous)
    return features, labels


def test_fit_and_score_returns_higher_scores_for_anomalies():
    features, labels = _synthetic_features()
    detector = IsolationForestDetector(contamination=0.1)
    detector.fit(features)
    scores = detector.score(features)

    normal_mean = scores[labels == 0].mean()
    anomalous_mean = scores[labels == 1].mean()
    assert anomalous_mean > normal_mean


def test_tune_contamination_selects_reasonable_value():
    train_features, _ = _synthetic_features(seed=1)
    val_features, val_labels = _synthetic_features(seed=2)
    best_contamination, best_model = IsolationForestDetector.tune_contamination(
        train_features, val_features, val_labels, candidates=[0.05, 0.10, 0.20]
    )
    assert best_contamination in [0.05, 0.10, 0.20]
    assert best_model is not None


def _processed_split(n, seed):
    import pandas as pd
    rng = np.random.default_rng(seed)
    fraud = np.arange(n) < n // 10
    amount = np.where(fraud, rng.uniform(50_000, 90_000, n), rng.uniform(1_000, 3_000, n))
    return pd.DataFrame({
        "claim_id": [f"S{seed}C{i:03d}" for i in range(n)],
        "provider_id": rng.choice(["PRV0001", "PRV0002"], n),
        "member_id": [f"MBR{i:04d}" for i in range(n)],
        "beneficiary_type": rng.choice(["principal", "dependent"], n),
        "patient_age": rng.integers(1, 80, n).astype(float),
        "patient_sex": rng.choice(["female", "male"], n),
        "facility_level": rng.choice([3, 4], n),
        "icd11_code": "BD10",
        "claim_amount": amount,
        "service_date": "2025-05-01",
        "discharge_date": "2025-05-02",
        "clinical_notes": "notes",
        "is_fraud": fraud.astype(int),
        "fraud_typology": np.where(fraud, "upcoding", "none"),
    })


def test_evaluate_baseline_fixed_contamination_writes_predictions(tmp_path, monkeypatch):
    from src.detection.evaluate_baseline import evaluate
    from src.evaluation.predictions import read_predictions

    _processed_split(120, 1).to_parquet(tmp_path / "train.parquet", index=False)
    validation = _processed_split(60, 2)
    validation.to_parquet(tmp_path / "validation.parquet", index=False)
    monkeypatch.chdir(tmp_path)

    results = evaluate(tmp_path, contamination=0.1, predictions=True)
    assert results["contamination"] == 0.1
    assert results["contamination_tuned"] is False

    predictions = read_predictions("isolation_forest", "validation")
    assert predictions["claim_id"].tolist() == validation["claim_id"].tolist()
    assert predictions["flagged"].sum() == results["flagged"]
    assert predictions["score"].notna().all()


def test_evaluate_baseline_tunes_by_default_and_writes_no_predictions(tmp_path, monkeypatch):
    from src.detection.evaluate_baseline import evaluate

    _processed_split(120, 1).to_parquet(tmp_path / "train.parquet", index=False)
    _processed_split(60, 2).to_parquet(tmp_path / "validation.parquet", index=False)
    monkeypatch.chdir(tmp_path)

    results = evaluate(tmp_path)
    assert results["contamination_tuned"] is True
    assert not (tmp_path / "experiments").exists()
