"""Publish the frozen classifier checkpoint (evaluated_v1) to the Hugging Face Hub with its
model card, and tag the commit so the deployed system can pin that exact revision.

Dry run by default: lists what would be uploaded and touches no network. --upload does it:
  1. creates the model repo if needed (private unless --public),
  2. uploads the adapter files and release/model_card.md as README.md in one commit
     (the checkpoint's own README.md, PEFT's empty template, is left out; its config.json,
     which is the training run's settings rather than a transformers config, is uploaded as
     training_config.json so the Hub does not mistake the repo for a full transformers model),
  3. tags that commit (v1.0-evaluated) and prints its sha to pin.

The token is read from HF_WRITE_TOKEN, or HF_TOKEN if that is unset (the deployed system's
HF_TOKEN only needs read access).

Usage:
  python -m src.release.publish_model --repo-id <user>/claimguard-consistency-classifier \\
      --checkpoint-dir ~/gdrive/claimguard/models/classifier_lora/evaluated_v1            # dry run
  python -m src.release.publish_model ... --upload
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DEFAULT_CARD = Path(__file__).resolve().parents[2] / "release" / "model_card.md"
DEFAULT_TAG = "v1.0-evaluated"
REQUIRED = ("adapter_config.json", "adapter_model.safetensors")
SKIPPED = {"README.md"}
RENAMED = {"config.json": "training_config.json"}


def plan_files(checkpoint_dir: Path, card: Path) -> list[tuple[Path, str]]:
    """(local file, path in the repo) for every file to upload, model card first. Raises if
    the checkpoint is missing an adapter file or the card is missing."""
    if not card.is_file():
        raise FileNotFoundError(f"model card not found: {card}")
    missing = [name for name in REQUIRED if not (checkpoint_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{checkpoint_dir} is not an adapter checkpoint; missing {', '.join(missing)}")
    files = [(card, "README.md")]
    for path in sorted(p for p in checkpoint_dir.iterdir() if p.is_file()):
        if path.name not in SKIPPED:
            files.append((path, RENAMED.get(path.name, path.name)))
    return files


def publish(api, repo_id: str, files: list[tuple[Path, str]], private: bool, tag: str) -> str:
    """Create the repo, upload `files` in one commit, tag it; return the commit sha."""
    from huggingface_hub import CommitOperationAdd

    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    commit = api.create_commit(
        repo_id=repo_id, repo_type="model",
        operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path)) for path, name in files],
        commit_message=f"Release {tag}: frozen evaluated_v1 classifier and model card",
    )
    api.create_tag(repo_id, tag=tag, revision=commit.oid, repo_type="model",
                   tag_message="Checkpoint every reported result was produced with")
    return commit.oid


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo-id", required=True, help="<user or org>/<name> on the Hub")
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--card", type=Path, default=DEFAULT_CARD)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--private", action=argparse.BooleanOptionalAction, default=True,
                        help="private repo (default); --no-private makes it public")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", dest="upload", action="store_false", help="list the files only (default)")
    mode.add_argument("--upload", dest="upload", action="store_true", help="actually upload and tag")
    parser.set_defaults(upload=False)
    args = parser.parse_args(argv)

    files = plan_files(args.checkpoint_dir.expanduser(), args.card)
    visibility = "private" if args.private else "public"
    print(f"{'Uploading to' if args.upload else 'Dry run, would upload to'} {args.repo_id} ({visibility}), tag {args.tag}:")
    for path, name in files:
        print(f"  {name:<28} {path.stat().st_size:>12,} bytes  <- {path}")
    if not args.upload:
        print("Nothing uploaded. Add --upload to publish.")
        return 0

    token = os.environ.get("HF_WRITE_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        print("HF_WRITE_TOKEN is not set (a write token from huggingface.co/settings/tokens).", file=sys.stderr)
        return 2
    from huggingface_hub import HfApi

    sha = publish(HfApi(token=token), args.repo_id, files, args.private, args.tag)
    print(f"Published commit {sha}, tagged {args.tag}.")
    print(f"Pin it: PeftModel.from_pretrained(base, {args.repo_id!r}, revision={sha!r})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
