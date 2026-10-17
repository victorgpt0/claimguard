import json
import sys

import numpy as np
import pandas as pd
import pytest

from src.evaluation import heldout
from src.evaluation.heldout import bootstrap_ci, group_seeds, mcnemar, metrics, seed_summary
from src.evaluation.predictions import write_predictions

Y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
FLAG = np.array([1, 1, 1, 0, 1, 0, 0, 0, 0, 0])


def test_metrics_known_values():
    m = metrics(Y, FLAG)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (3, 1, 1, 5)
    assert m["precision"] == 0.75 and m["recall"] == 0.75 and m["f1"] == 0.75
    assert m["fpr"] == round(1 / 6, 4)
    assert m["auc_roc"] is None


def test_metrics_auc_only_with_score_and_both_classes():
    score = np.array([0.9, 0.8, 0.7, 0.6, 0.1, 0.2, 0.3, 0.4, 0.0, 0.05])
    assert metrics(Y, FLAG, score)["auc_roc"] == 1.0
    assert metrics(np.zeros(4), np.zeros(4), np.arange(4))["auc_roc"] is None


def test_metrics_nothing_flagged_is_zero_not_nan():
    m = metrics(Y, np.zeros(10))
    assert m["precision"] == 0.0 and m["f1"] == 0.0 and m["fpr"] == 0.0


def test_bootstrap_ci_contains_point_estimate_and_is_paired():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    a = np.where(rng.random(200) < 0.8, y, 1 - y)
    score = a + rng.random(200) * 0.1
    cis = bootstrap_ci(y, {"a": (a, score), "a_copy": (a.copy(), None)}, n=300, seed=0)
    point = metrics(y, a, score)
    for metric in ["precision", "recall", "f1", "fpr", "auc_roc"]:
        lo, hi = cis["a"][metric]
        assert lo <= point[metric] <= hi
    # Same resamples for every system: identical predictions give identical intervals.
    assert cis["a"]["f1"] == cis["a_copy"]["f1"]
    assert "auc_roc" not in cis["a_copy"]
    assert bootstrap_ci(y, {"a": (a, None)}, n=300, seed=0) == {"a": cis["a_copy"]}


def test_mcnemar_known_result():
    y = np.zeros(20, dtype=int)
    a = np.zeros(20, dtype=int)          # always right
    b = np.zeros(20, dtype=int)
    b[:8] = 1                            # wrong on 8
    result = mcnemar(y, a, b)
    assert (result["b"], result["c"]) == (8, 0)
    assert result["p_value"] == pytest.approx(2 * 0.5 ** 8)   # exact two-sided: 0.0078125
    assert mcnemar(y, a, a) == {"b": 0, "c": 0, "p_value": 1.0}
    assert mcnemar(y, b, a)["c"] == 8


def test_seed_summary_mean_std():
    s = seed_summary([{"f1": 0.6, "recall": 0.5, "auc_roc": None},
                      {"f1": 0.8, "recall": 0.5, "auc_roc": None}])
    assert s["seeds"] == 2
    assert s["f1"] == {"mean": 0.7, "std": round(np.std([0.6, 0.8], ddof=1), 4)}
    assert s["recall"]["std"] == 0.0
    assert "auc_roc" not in s


def test_group_seeds_auto_and_explicit():
    names = ["pipeline", "structured_only_seed42", "structured_only_seed7", "zero_shot"]
    assert group_seeds(names) == {"structured_only": ["structured_only_seed42", "structured_only_seed7"]}
    assert group_seeds(names, ["so=structured_only_seed42"]) == {"so": ["structured_only_seed42"]}


def _dataset(tmp_path):
    n = 12
    claims = pd.DataFrame({
        "claim_id": [f"C{i}" for i in range(n)],
        "is_fraud": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        "label": ["inconsistent", "inconsistent", "consistent", "consistent"] + ["consistent"] * 8,
        "fraud_typology": ["ghost_billing", "upcoding", "anomalous_dependents", "inflated_caesarean"] + ["none"] * 8,
        "icd11_code": ["1A00"] * 10 + ["ZZ99", "ZZ99"],
        "clinical_notes": ["notes"] * n,
        "has_text_noise": [True, False] * 6,
        "has_structured_noise": [False] * n,
        "claim_amount": [1000.0, 0.0, 500.0, 2000.0] + [100.0] * 8,
    })
    data = tmp_path / "data"
    data.mkdir()
    claims.to_parquet(data / "validation.parquet")
    claims.to_parquet(data / "test.parquet")
    claims.to_parquet(data / "train.parquet")
    root = tmp_path / "predictions"
    write_predictions("pipeline", "validation", pd.DataFrame(
        {"claim_id": claims["claim_id"], "flagged": [1, 1, 0, 1, 1] + [0] * 7,
         "score": [0.9, 0.8, 0.1, 0.7, 0.6] + [0.0] * 7}), root)
    for seed, flags in [(1, [1, 0, 0, 0, 0]), (2, [1, 1, 0, 0, 1])]:
        write_predictions(f"rules_seed{seed}", "validation", pd.DataFrame(
            {"claim_id": claims["claim_id"], "flagged": flags + [0] * 7}), root)
    return data, root


def test_evaluate_end_to_end(tmp_path):
    data, root = _dataset(tmp_path)
    claims = heldout.load_split(data, "validation")
    result = heldout.evaluate(claims, claims["icd11_code"].tolist(),
                              ["pipeline", "rules_seed1", "rules_seed2"], "validation",
                              "pipeline", root=root, n_boot=50)
    pipe = result["systems"]["pipeline"]
    assert pipe["metrics"]["recall"] == 0.75 and pipe["metrics"]["auc_roc"] is not None
    assert result["systems"]["rules_seed1"]["metrics"]["auc_roc"] is None   # blank scores
    assert pipe["cost"]["fraud_kes_missed"] == 500.0
    assert pipe["strata"]["flagged_by_typology"]["ghost_billing"]["rate"] == 1.0
    assert set(result["mcnemar"]) == {"rules_seed1", "rules_seed2"}
    assert result["seed_groups"]["rules"]["seeds"] == 2
    tables = heldout.tables_markdown(result)
    assert "## McNemar" in tables and "| rules |" in tables and "Capitation" in tables


def test_text_truth_uses_label(tmp_path):
    data, root = _dataset(tmp_path)
    claims = heldout.load_split(data, "validation")
    assert heldout.truth(claims, "text").tolist()[:4] == [1, 1, 0, 0]


def test_missing_predictions_rejected(tmp_path):
    data, root = _dataset(tmp_path)
    claims = heldout.load_split(data, "validation")
    write_predictions("partial", "validation", pd.DataFrame({"claim_id": ["C0"], "flagged": [1]}), root)
    with pytest.raises(ValueError, match="no prediction for 11"):
        heldout.join_predictions(claims, "partial", "validation", root)


def test_int_claim_ids_match_string_predictions(tmp_path):
    claims = pd.DataFrame({"claim_id": [1, 2], "is_fraud": [1, 0]})
    claims["claim_id"] = claims["claim_id"].astype(str)
    write_predictions("s", "validation", pd.DataFrame({"claim_id": [2, 1], "flagged": [0, 1]}), tmp_path)
    frame = heldout.join_predictions(claims, "s", "validation", tmp_path)
    assert frame["flagged"].tolist() == [1, 0]


def test_main_requires_final_for_test(tmp_path, monkeypatch):
    data, root = _dataset(tmp_path)
    monkeypatch.setattr(sys, "argv", ["heldout", "--data", str(data), "--split", "test",
                                      "--systems", "pipeline", "--predictions-root", str(root)])
    with pytest.raises(SystemExit):
        heldout.main()


def test_main_writes_results_and_tables(tmp_path, monkeypatch):
    data, root = _dataset(tmp_path)
    out = tmp_path / "eval"
    monkeypatch.setattr(sys, "argv", ["heldout", "--data", str(data), "--systems", "pipeline",
                                      "rules_seed1", "rules_seed2", "--predictions-root", str(root),
                                      "--resamples", "20", "--out", str(out)])
    heldout.main()
    run = next(out.iterdir())
    result = json.loads((run / "results.json").read_text())
    assert result["reference"] == "pipeline" and "mcnemar" in result
    assert (run / "tables.md").read_text().startswith("# Held-out evaluation")
