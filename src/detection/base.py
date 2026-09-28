"""BaseAnomalyDetector (Section 3.6.1 Class Diagram): abstract base for the two
unsupervised anomaly detectors (Isolation Forest, Autoencoder)."""
from abc import ABC, abstractmethod


class BaseAnomalyDetector(ABC):
    @abstractmethod
    def fit(self, feature_matrix):
        """Train on the unsupervised (unlabelled) training split."""
        raise NotImplementedError

    @abstractmethod
    def score(self, feature_matrix):
        """Return an anomaly score per claim. Higher = more anomalous."""
        raise NotImplementedError
