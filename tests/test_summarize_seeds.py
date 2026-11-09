import json
import statistics

import pytest

from src.classification.summarize_seeds import load_runs, main, summarise

SEEDS = {42: (0.80, 0.70, 0.75), 123: (0.90, 0.60, 0.72), 2024: (0.85, 0.75, 0.80)}


def _write_runs(root, seeds=SEEDS):
    for seed, (p, r, f) in seeds.items():
        run_dir = root / f"20261017T120000_seed{seed}"
        run_dir.mkdir(parents=True)
        metrics = {"precision": p, "recall": r, "f1": f, "flagged": 10, "positives": 12, "epoch": 3}
        (run_dir / "metrics.json").write_text(json.dumps(metrics))
        (run_dir / "config.json").write_text(json.dumps(
            {"run_id": run_dir.name, "seed": seed, "metrics": metrics}))


def test_mean_and_sample_std_over_seeds(tmp_path):
    _write_runs(tmp_path)
    summary = summarise(load_runs(tmp_path))
    f1s = [v[2] for v in SEEDS.values()]
    assert summary["metrics"]["f1"]["mean"] == pytest.approx(statistics.mean(f1s))
    assert summary["metrics"]["f1"]["std"] == pytest.approx(statistics.stdev(f1s))  # ddof=1
    assert summary["metrics"]["precision"]["n"] == 3
    assert {r["seed"] for r in summary["runs"]} == set(SEEDS)


def test_names_the_best_validation_f1_seed(tmp_path):
    _write_runs(tmp_path)
    best = summarise(load_runs(tmp_path))["best"]
    assert best["seed"] == 2024
    assert best["run_id"] == "20261017T120000_seed2024"


def test_single_run_has_no_std(tmp_path):
    _write_runs(tmp_path, {42: SEEDS[42]})
    assert summarise(load_runs(tmp_path))["metrics"]["recall"]["std"] is None


def test_empty_directory_is_an_error(tmp_path):
    with pytest.raises(ValueError):
        summarise(load_runs(tmp_path))


def test_cli_prints_and_saves(tmp_path, capsys):
    runs = tmp_path / "runs"
    _write_runs(runs)
    out = tmp_path / "summary.json"
    main(["--runs", str(runs), "--out", str(out)])
    saved = json.loads(out.read_text())
    assert saved["best"]["seed"] == 2024
    printed = capsys.readouterr().out
    assert "seed 2024" in printed and "±" in printed


def test_cli_default_output_sits_in_the_runs_directory(tmp_path):
    _write_runs(tmp_path)
    main(["--runs", str(tmp_path)])
    assert (tmp_path / "seed_summary.json").exists()
    # The summary file itself is not mistaken for a run on a second pass.
    assert len(load_runs(tmp_path)) == 3
