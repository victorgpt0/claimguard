"""AutoencoderDetector (FR-02; Section 3.2.3). Deep autoencoder learning a compressed
latent representation of legitimate claim patterns; reconstruction error above a
validation-tuned threshold (minimising false positives while keeping recall >= 0.80,
NFR-05) marks a claim anomalous.
"""
from src.detection.base import BaseAnomalyDetector


class AutoencoderDetector(BaseAnomalyDetector):
    def __init__(self, reconstruction_threshold: float | None = None):
        self.reconstruction_threshold = reconstruction_threshold  # tuned on validation split
        self.model = None

    def fit(self, feature_matrix):
        raise NotImplementedError

    def score(self, feature_matrix):
        raise NotImplementedError
