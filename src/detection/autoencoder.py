"""AutoencoderDetector (FR-02; Section 3.2.3). Deep autoencoder learning a compressed
latent representation of legitimate claim patterns; reconstruction error above a
validation-tuned threshold (minimising false positives while keeping recall >= 0.80,
NFR-05) marks a claim anomalous.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from src.detection.base import BaseAnomalyDetector


class _AutoencoderNet(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 8):
        super().__init__()
        hidden_dim = max(latent_dim * 2, input_dim // 2)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class AutoencoderDetector(BaseAnomalyDetector):
    def __init__(self, input_dim: int, latent_dim: int = 8,
                 reconstruction_threshold: float | None = None, device: str = "cpu"):
        self.device = device
        self.model = _AutoencoderNet(input_dim, latent_dim).to(device)
        self.reconstruction_threshold = reconstruction_threshold  # tuned on validation split

    def fit(self, feature_matrix: np.ndarray, epochs: int = 50, lr: float = 1e-3,
            batch_size: int = 64) -> "AutoencoderDetector":
        x = torch.tensor(feature_matrix, dtype=torch.float32, device=self.device)
        dataset = torch.utils.data.TensorDataset(x)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimiser = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        self.model.train()
        for _ in range(epochs):
            for (batch,) in loader:
                optimiser.zero_grad()
                reconstruction = self.model(batch)
                loss = loss_fn(reconstruction, batch)
                loss.backward()
                optimiser.step()
        return self

    def score(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Per-claim reconstruction error (mean squared error). Higher = more anomalous."""
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(feature_matrix, dtype=torch.float32, device=self.device)
            reconstruction = self.model(x)
            errors = torch.mean((reconstruction - x) ** 2, dim=1)
        return errors.cpu().numpy()

    def tune_threshold(self, val_features: np.ndarray, val_labels: np.ndarray,
                        min_recall: float = 0.80) -> float:
        """Selects the reconstruction-error threshold that minimises false positives
        while keeping recall >= min_recall (NFR-05), on the validation split."""
        scores = self.score(val_features)
        candidate_thresholds = np.quantile(scores, np.linspace(0.5, 0.999, 200))
        best_threshold, best_fp_count = candidate_thresholds[0], np.inf
        for threshold in candidate_thresholds:
            predicted_anomaly = scores >= threshold
            true_positives = np.sum(predicted_anomaly & (val_labels == 1))
            false_positives = np.sum(predicted_anomaly & (val_labels == 0))
            actual_positives = np.sum(val_labels == 1)
            recall = true_positives / actual_positives if actual_positives else 0.0
            if recall >= min_recall and false_positives < best_fp_count:
                best_fp_count = false_positives
                best_threshold = threshold
        self.reconstruction_threshold = float(best_threshold)
        return self.reconstruction_threshold
