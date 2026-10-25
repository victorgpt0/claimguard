"""Zero-shot LLM baseline (#52): prompt, answer parsing, the funnel rule, and a run with a
fake LLM (the real one needs a GPU)."""
import json

import pandas as pd
import pytest

from src.baselines.zero_shot_llm import (PROMPT, build_prompt, funnel_predictions, parse_answer, run)
from src.evaluation.predictions import read_predictions
from src.icd11.cache import ICD11Cache


@pytest.fixture
def cache_path(tmp_path):
    path = tmp_path / "cache.json"
    path.write_text(json.dumps({
        "JB20.0": {"code": "JB20.0", "title": "Single spontaneous delivery", "definition": "",
                   "inclusion": [], "exclusion": []},
        "DB10.02": {"code": "DB10.02", "title": "Acute appendicitis", "definition": "Inflammation of the appendix.",
                    "inclusion": [], "exclusion": []}}))
    return path


@pytest.mark.parametrize("text, expected", [
    ("INCONSISTENT", "inconsistent"), ("Consistent.", "consistent"), ("**INCONSISTENT**", "inconsistent"),
    ("inconsistent - the notes describe", "inconsistent"), ("The notes are consistent", None),
    ("", None), (None, None), ("MAYBE", None)])
def test_only_the_first_word_counts(text, expected):
    assert parse_answer(text) == expected


def test_prompt_names_code_title_definition_and_notes(cache_path):
    prompt = build_prompt("DB10.02", ICD11Cache(cache_path), "Appendicectomy done.")
    assert "Billed ICD-11 code: DB10.02 — Acute appendicitis." in prompt
    assert "Official definition: Inflammation of the appendix." in prompt
    assert "Clinical notes: Appendicectomy done." in prompt
    assert "(no definition" in build_prompt("JB20.0", ICD11Cache(cache_path), "x")


def test_funnel_follows_the_pipeline_lane_rule():
    ids = ["A", "B", "C", "D"]
    zero_shot = pd.DataFrame({"claim_id": ids, "flagged": [1, 1, 0, 0]})
    stage1 = pd.DataFrame({"claim_id": ids, "flagged": [1, 0, 1, 1]})
    structural = pd.DataFrame({"claim_id": ids, "flagged": [0, 0, 0, 1]})
    systems = funnel_predictions(zero_shot, stage1, structural)
    assert list(systems["zero_shot_llm_text_lane"]["flagged"]) == [1, 0, 0, 0]  # B never reached stage 2
    assert list(systems["zero_shot_llm_funnel"]["flagged"]) == [1, 0, 0, 1]  # D via the structural lane
    with pytest.raises(ValueError):
        funnel_predictions(zero_shot, stage1.head(3), structural)


def test_run_with_a_fake_llm_records_prompt_counts_and_predictions(tmp_path, cache_path):
    data = tmp_path / "processed"
    data.mkdir()
    pd.DataFrame({"claim_id": ["C1", "C2", "C3"], "icd11_code": ["DB10.02", "JB20.0", "jb20.0"],
                  "clinical_notes": ["n1", "n2", "n3"]}).to_parquet(data / "validation.parquet")
    seen = []

    def fake(prompts):
        seen.extend(prompts)
        return ["INCONSISTENT", "consistent", "I am not sure"]

    run_dir = run(data, "validation", cache_path, tmp_path / "runs", fake, "fake-model", tmp_path / "pred")
    summary = json.loads((run_dir / "summary.json").read_text())
    assert (summary["inconsistent"], summary["consistent"], summary["unparseable"]) == (1, 1, 1)
    assert (run_dir / "prompt.txt").read_text() == PROMPT
    assert "JB20.0 — Single spontaneous delivery" in seen[2]  # the malformed code was repaired
    assert list(read_predictions("zero_shot_llm", "validation", tmp_path / "pred")["flagged"]) == [1, 0, 0]
