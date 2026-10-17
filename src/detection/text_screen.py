"""TextConsistencyScreen: the third stage-1 component (#50). Flags a claim whose clinical
notes share little vocabulary with its billed code's ICD-11 reference text, so that fraud
visible only in the notes (ghost billing, record fabrication) reaches the Semantic
Consistency Classifier instead of being dropped by the structured-data detectors.

It is a cheap lexical filter, not a detector of fraud on its own: it only decides which
claims the classifier sees, and the classifier's verdict decides what an auditor sees
(src/pipeline.py). Word TF-IDF (1-2 grams, English stop words removed) was chosen over
character n-grams on the v0.5-5k validation split: at the same clean-claim pass rate it
passed more ghost-billing notes on (0.80 vs 0.53 at 10%).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.classification.export import reference_text
from src.data.preprocessing import repair_icd11_code
from src.icd11.cache import ICD11Cache


class TextConsistencyScreen:
    def __init__(self, cache: ICD11Cache, threshold: float | None = None):
        self.cache = cache
        self.threshold = threshold
        self.vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True,
                                          stop_words="english")

    def _references(self, codes) -> list[str]:
        # The billed code after repair, as the classifier sees it (export.export_split).
        return [reference_text(self.cache.get(repair_icd11_code(code))) for code in codes]

    def fit(self, train: pd.DataFrame) -> "TextConsistencyScreen":
        """Vocabulary from the training notes and every training code's reference text."""
        references = sorted(set(self._references(train["icd11_code"])))
        self.vectorizer.fit(list(train["clinical_notes"]) + references)
        return self

    def similarity(self, claims: pd.DataFrame) -> np.ndarray:
        """Cosine similarity of each claim's notes to its code's reference text (TF-IDF
        rows are L2-normalised, so the row-wise dot product is the cosine)."""
        notes = self.vectorizer.transform(claims["clinical_notes"])
        references = self.vectorizer.transform(self._references(claims["icd11_code"]))
        return np.asarray(notes.multiply(references).sum(axis=1)).ravel()

    def score(self, claims: pd.DataFrame) -> np.ndarray:
        """Higher = more suspicious, matching the detectors' convention."""
        return 1.0 - self.similarity(claims)

    def flag(self, claims: pd.DataFrame) -> np.ndarray:
        if self.threshold is None:
            raise ValueError("No threshold: call tune() or pass one from config.yaml")
        return self.similarity(claims) <= self.threshold

    def tune(self, validation: pd.DataFrame, max_clean_pass_rate: float = 0.10) -> float:
        """Sets the threshold so that at most `max_clean_pass_rate` of the validation
        split's clean claims are passed on (Section 3.2.3: validation split only)."""
        clean = validation["is_fraud"].to_numpy() == 0
        self.threshold = threshold_for_pass_rate(self.similarity(validation)[clean], max_clean_pass_rate)
        return self.threshold


def threshold_for_pass_rate(clean_similarity: np.ndarray, max_pass_rate: float) -> float:
    """The largest threshold whose `similarity <= threshold` passes on at most
    `max_pass_rate` of the clean claims. Ties matter: many notes share no word with their
    reference text (similarity 0), so a plain quantile can pass on more than the target.
    If even the lowest value exceeds the target, the threshold sits below every
    similarity and the screen flags nothing."""
    values, counts = np.unique(clean_similarity, return_counts=True)
    pass_rates = np.cumsum(counts) / len(clean_similarity)
    allowed = values[pass_rates <= max_pass_rate]
    return float(allowed[-1]) if len(allowed) else -1.0
