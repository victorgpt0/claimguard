import pandas as pd
import pytest

from src.evaluation.predictions import prediction_path, read_predictions, write_predictions


def test_round_trip_keeps_string_ids_and_blank_scores(tmp_path):
    frame = pd.DataFrame({"claim_id": ["CLM000001", "007"], "flagged": [True, False]})
    path = write_predictions("rules", "validation", frame, tmp_path)
    assert path == prediction_path("rules", "validation", tmp_path)
    back = read_predictions("rules", "validation", tmp_path)
    assert back["claim_id"].tolist() == ["CLM000001", "007"]
    assert back["flagged"].tolist() == [1, 0]
    assert back["score"].isna().all()


def test_duplicate_ids_rejected(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        write_predictions("s", "test", pd.DataFrame({"claim_id": [1, 1], "flagged": [0, 1]}), tmp_path)
