import numpy as np

from src.detection.autoencoder import AutoencoderDetector


def _synthetic_features(n_normal=60, n_anomalous=10, seed=0):
    rng = np.random.default_rng(seed)
    normal = rng.normal(loc=0, scale=1, size=(n_normal, 4)).astype(np.float32)
    anomalous = rng.normal(loc=10, scale=1, size=(n_anomalous, 4)).astype(np.float32)
    features = np.vstack([normal, anomalous])
    labels = np.array([0] * n_normal + [1] * n_anomalous)
    return features, labels


def test_fit_reduces_reconstruction_error_on_normal_data():
    features, labels = _synthetic_features()
    detector = AutoencoderDetector(input_dim=4, latent_dim=2)

    scores_before = detector.score(features[labels == 0])
    detector.fit(features[labels == 0], epochs=30)
    scores_after = detector.score(features[labels == 0])

    assert scores_after.mean() < scores_before.mean()


def test_anomalous_scores_higher_than_normal_after_fit():
    features, labels = _synthetic_features()
    detector = AutoencoderDetector(input_dim=4, latent_dim=2)
    detector.fit(features[labels == 0], epochs=40)

    scores = detector.score(features)
    assert scores[labels == 1].mean() > scores[labels == 0].mean()


def test_tune_threshold_meets_minimum_recall():
    features, labels = _synthetic_features()
    detector = AutoencoderDetector(input_dim=4, latent_dim=2)
    detector.fit(features[labels == 0], epochs=40)

    threshold = detector.tune_threshold(features, labels, min_recall=0.80)
    scores = detector.score(features)
    predicted = scores >= threshold
    recall = (predicted & (labels == 1)).sum() / (labels == 1).sum()
    assert recall >= 0.80
