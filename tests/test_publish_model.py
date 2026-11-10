"""src/release/publish_model.py: the dry run lists the files and never touches the Hub;
--upload makes one commit and tags it, against a mocked HfApi (no network)."""
from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest

from src.release import publish_model


@pytest.fixture
def checkpoint(tmp_path):
    ckpt = tmp_path / "evaluated_v1"
    ckpt.mkdir()
    for name in ("adapter_config.json", "adapter_model.safetensors", "config.json", "metrics.json", "README.md"):
        (ckpt / name).write_text("x")
    card = tmp_path / "model_card.md"
    card.write_text("---\nlicense: mit\n---\n# card\n")
    return ckpt, card


def test_plan_puts_the_card_as_readme_and_renames_the_run_config(checkpoint):
    ckpt, card = checkpoint
    names = [name for _, name in publish_model.plan_files(ckpt, card)]
    assert names[0] == "README.md"
    assert names.count("README.md") == 1  # the checkpoint's own template is left out
    assert set(names) == {"README.md", "adapter_config.json", "adapter_model.safetensors",
                          "training_config.json", "metrics.json"}


def test_plan_rejects_a_folder_without_an_adapter(tmp_path, checkpoint):
    _, card = checkpoint
    with pytest.raises(FileNotFoundError, match="adapter_model.safetensors"):
        publish_model.plan_files(tmp_path, card)


def test_dry_run_is_the_default_and_does_not_contact_the_hub(checkpoint, capsys, monkeypatch):
    ckpt, card = checkpoint
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_WRITE_TOKEN", raising=False)
    with mock.patch("huggingface_hub.HfApi") as api:
        code = publish_model.main(["--repo-id", "u/m", "--checkpoint-dir", str(ckpt), "--card", str(card)])
    assert code == 0
    api.assert_not_called()
    out = capsys.readouterr().out
    assert "Dry run" in out and "(private)" in out and "training_config.json" in out


def test_upload_needs_a_token(checkpoint, monkeypatch):
    ckpt, card = checkpoint
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_WRITE_TOKEN", raising=False)
    assert publish_model.main(["--repo-id", "u/m", "--checkpoint-dir", str(ckpt),
                               "--card", str(card), "--upload"]) == 2


def test_upload_commits_once_and_tags_that_commit(checkpoint, capsys, monkeypatch):
    ckpt, card = checkpoint
    monkeypatch.delenv("HF_WRITE_TOKEN", raising=False)
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    api = mock.Mock()
    api.create_commit.return_value = SimpleNamespace(oid="abc123")
    with mock.patch("huggingface_hub.HfApi", return_value=api) as cls:
        code = publish_model.main(["--repo-id", "u/m", "--checkpoint-dir", str(ckpt),
                                   "--card", str(card), "--upload"])
    assert code == 0
    cls.assert_called_once_with(token="hf_test")
    api.create_repo.assert_called_once_with("u/m", repo_type="model", private=True, exist_ok=True)
    operations = api.create_commit.call_args.kwargs["operations"]
    assert sorted(op.path_in_repo for op in operations) == sorted(
        ["README.md", "adapter_config.json", "adapter_model.safetensors", "training_config.json", "metrics.json"])
    api.create_tag.assert_called_once()
    assert api.create_tag.call_args.kwargs["revision"] == "abc123"
    assert api.create_tag.call_args.kwargs["tag"] == "v1.0-evaluated"
    assert "abc123" in capsys.readouterr().out


def test_upload_prefers_the_write_token(checkpoint, monkeypatch):
    ckpt, card = checkpoint
    monkeypatch.setenv("HF_TOKEN", "hf_read")
    monkeypatch.setenv("HF_WRITE_TOKEN", "hf_write")
    api = mock.Mock()
    api.create_commit.return_value = SimpleNamespace(oid="abc123")
    with mock.patch("huggingface_hub.HfApi", return_value=api) as cls:
        publish_model.main(["--repo-id", "u/m", "--checkpoint-dir", str(ckpt), "--card", str(card), "--upload"])
    cls.assert_called_once_with(token="hf_write")
