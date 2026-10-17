"""Stage-1 additions for #50: the text-consistency screen and recall-floor tuning of
Isolation Forest."""
import numpy as np
import pandas as pd
import pytest

from src.detection.isolation_forest import IsolationForestDetector
from src.detection.text_screen import TextConsistencyScreen
from src.icd11.client import ICD11Definition

REFERENCES = {
    "JB20": ("Single spontaneous vertex delivery", "Vaginal delivery of a single infant, head first."),
    "CA40": ("Pneumonia", "Infection of the lung parenchyma with cough, fever and consolidation."),
}


class _Cache:
    def get(self, code):
        title, definition = REFERENCES[code]
        return ICD11Definition(code, title, definition, [], [])


def _claims(notes_and_codes, is_fraud):
    return pd.DataFrame({"clinical_notes": [n for n, _ in notes_and_codes],
                         "icd11_code": [c for _, c in notes_and_codes], "is_fraud": is_fraud})


@pytest.fixture
def train():
    return _claims([("Spontaneous vaginal delivery of a single infant, head first, no tears.", "JB20"),
                    ("Cough and fever for three days, consolidation on chest exam: pneumonia.", "CA40"),
                    ("Vertex delivery, healthy infant, mother stable.", "JB20"),
                    ("Lung infection with productive cough and fever, started antibiotics.", "CA40")],
                   [0, 0, 0, 0])


def test_screen_scores_unrelated_notes_as_dissimilar(train):
    screen = TextConsistencyScreen(_Cache()).fit(train)
    batch = _claims([("Vaginal delivery of a single infant, head first.", "JB20"),
                     ("Patient seen and managed as appropriate, reviewed.", "JB20")], [0, 1])
    similarity = screen.similarity(batch)
    assert similarity[0] > similarity[1]
    assert np.allclose(screen.score(batch), 1 - similarity)


def test_screen_tune_caps_the_clean_pass_rate(train):
    screen = TextConsistencyScreen(_Cache()).fit(train)
    validation = _claims([("Vaginal delivery of a single infant.", "JB20")] * 8
                         + [("Fever and cough, pneumonia.", "CA40")] * 8
                         + [("Routine review, nothing further.", "CA40")] * 4, [0] * 16 + [1] * 4)
    # Each clean group is half the clean claims (a tie), so 0.5 is the tightest
    # target that lets the screen flag anything.
    assert screen.tune(validation, max_clean_pass_rate=0.10) == -1.0
    screen.tune(validation, max_clean_pass_rate=0.5)
    flags = screen.flag(validation)
    assert flags[16:].all()          # boilerplate notes are passed on
    assert flags[:16].mean() <= 0.5


def test_screen_without_threshold_refuses_to_flag(train):
    with pytest.raises(ValueError):
        TextConsistencyScreen(_Cache()).fit(train).flag(train)


def test_contamination_for_recall_takes_the_smallest_meeting_the_floor():
    rng = np.random.default_rng(0)
    train_x = rng.normal(size=(400, 4))
    val_x = np.vstack([rng.normal(size=(180, 4)), rng.normal(loc=6, size=(20, 4))])
    val_y = np.r_[np.zeros(180), np.ones(20)].astype(int)
    contamination, detector = IsolationForestDetector.tune_contamination_for_recall(
        train_x, val_x, val_y, min_recall=0.9)
    assert contamination == 0.05  # the outliers are obvious, so the first grid value suffices
    assert (detector.model.predict(val_x[180:]) == -1).mean() >= 0.9


def test_contamination_for_recall_falls_back_to_the_largest():
    rng = np.random.default_rng(1)
    train_x, val_x = rng.normal(size=(300, 3)), rng.normal(size=(100, 3))
    val_y = np.r_[np.zeros(50), np.ones(50)].astype(int)  # indistinguishable: floor unreachable
    contamination, _ = IsolationForestDetector.tune_contamination_for_recall(
        train_x, val_x, val_y, min_recall=0.99)
    assert contamination == 0.40


def test_threshold_never_exceeds_the_target_rate_with_ties():
    from src.detection.text_screen import threshold_for_pass_rate
    similarity = np.r_[np.zeros(12), np.linspace(0.1, 0.9, 88)]  # 12% share no word at all
    assert threshold_for_pass_rate(similarity, 0.05) == -1.0     # cannot split the tie
    t = threshold_for_pass_rate(similarity, 0.15)
    assert 0 < t and (similarity <= t).mean() <= 0.15
