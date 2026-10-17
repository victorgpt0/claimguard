import pytest

from src.config import load_config


def test_load_config_reads_yaml(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("paths:\n  db_path: /tmp/test.sqlite\nthresholds:\n  retrain_trigger_count: 50\n")
    config = load_config(config_path)
    assert config["paths"]["db_path"] == "/tmp/test.sqlite"
    assert config["thresholds"]["retrain_trigger_count"] == 50


def test_load_config_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does_not_exist.yaml")


def test_relative_paths_resolve_against_the_repo_root(tmp_path):
    from src.config import REPO_ROOT
    config_path = tmp_path / "config.yaml"
    config_path.write_text("paths:\n  db_path: db/sha_fraud.sqlite\n  data_dir: /content/drive/data\n")
    config = load_config(config_path)
    assert config["paths"]["db_path"] == str(REPO_ROOT / "db" / "sha_fraud.sqlite")
    assert config["paths"]["data_dir"] == "/content/drive/data"
