"""Zero-shot LLM baseline (Section 3.2.4, #52): the same Llama 3.1 8B that writes the
narratives is asked, without any fine-tuning, whether a claim's notes are consistent with
its billed code. It answers the question the Semantic Consistency Classifier answers, so
it is compared with the classifier on the text label, and, put in the classifier's place
inside the funnel, with the pipeline on is_fraud.

This baseline is the only place an LLM decides anything, and it exists to measure that
choice against the fine-tuned classifier; the system itself never does it (IR-02).

Temperature 0. The first word of the answer is parsed; an answer that is neither word is
counted as unparseable and not flagged. The prompt and the counts are saved with the run.

    # Colab (vLLM, GPU):
    python -m src.baselines.zero_shot_llm run --data <processed>/v0.6-20k --split validation \\
        --cache artifacts/icd11_cache.json --out experiments/zero_shot_llm/v0.6-20k
    # after a pipeline run with --predictions on the same split (CPU):
    python -m src.baselines.zero_shot_llm combine --split validation
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from src.data.preprocessing import repair_icd11_code
from src.evaluation.predictions import PREDICTIONS_DIR, read_predictions, write_predictions
from src.icd11.cache import ICD11Cache
from src.narrative.narrative_generator import DEFAULT_VLLM_MODEL

SYSTEM = "zero_shot_llm"

PROMPT = """You are a medical claims auditor. Billed ICD-11 code: {code} — {title}.
Official definition: {definition}
Clinical notes: {notes}
Are the notes consistent with the billed code? Answer with one word: CONSISTENT or INCONSISTENT."""

AnswerFn = Callable[[list[str]], list[str]]


def build_prompt(code: str, cache: ICD11Cache, notes: str) -> str:
    definition = cache.get(code)
    return PROMPT.format(code=code, title=definition.title,
                         definition=definition.definition or "(no definition; the title is used)",
                         notes=notes)


def parse_answer(text: str) -> str | None:
    """"inconsistent", "consistent" or None. Only the first word counts."""
    words = re.findall(r"[A-Za-z]+", text or "")
    first = words[0].upper() if words else ""
    return {"INCONSISTENT": "inconsistent", "CONSISTENT": "consistent"}.get(first)


def score_claims(claims: pd.DataFrame, cache: ICD11Cache, answer: AnswerFn) -> pd.DataFrame:
    codes = [repair_icd11_code(code) for code in claims["icd11_code"]]
    prompts = [build_prompt(code, cache, notes) for code, notes in zip(codes, claims["clinical_notes"])]
    raw = answer(prompts)
    if len(raw) != len(prompts):
        raise RuntimeError(f"{len(raw)} answers for {len(prompts)} prompts")
    parsed = [parse_answer(text) for text in raw]
    return pd.DataFrame({"claim_id": claims["claim_id"].astype(str).to_numpy(), "raw": raw,
                         "answer": parsed,
                         "flagged": [int(a == "inconsistent") for a in parsed]})


def vllm_answers(model: str, gpu_memory_utilization: float = 0.85) -> AnswerFn:
    """Batched generation on the local GPU (Colab). vLLM is imported lazily: Colab only."""
    from vllm import LLM, SamplingParams

    llm = LLM(model=model, dtype="half", max_model_len=2048, gpu_memory_utilization=gpu_memory_utilization,
              max_num_batched_tokens=2048)
    params = SamplingParams(temperature=0.0, max_tokens=5, seed=0)

    def answer(prompts: list[str]) -> list[str]:
        outputs = llm.chat([[{"role": "user", "content": p}] for p in prompts], params, use_tqdm=True)
        return [o.outputs[0].text.strip() for o in outputs]
    return answer


def funnel_predictions(zero_shot: pd.DataFrame, stage1: pd.DataFrame,
                       structural: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The zero-shot answer in the classifier's place, with the pipeline's lane rule: a
    claim is reported when stage 1 flagged it and the LLM says INCONSISTENT (text lane), or
    when the Isolation Forest or autoencoder flagged it (structural lane)."""
    frame = (zero_shot[["claim_id", "flagged"]].rename(columns={"flagged": "llm"})
             .merge(stage1.rename(columns={"flagged": "stage1"})[["claim_id", "stage1"]], on="claim_id")
             .merge(structural.rename(columns={"flagged": "structural"})[["claim_id", "structural"]], on="claim_id"))
    if len(frame) != len(zero_shot):
        raise ValueError("stage-1 predictions do not cover the same claims as the zero-shot run")
    text_lane = frame["llm"] & frame["stage1"]
    return {f"{SYSTEM}_text_lane": pd.DataFrame({"claim_id": frame["claim_id"], "flagged": text_lane}),
            f"{SYSTEM}_funnel": pd.DataFrame({"claim_id": frame["claim_id"],
                                              "flagged": text_lane | frame["structural"]})}


def run(data: Path, split: str, cache_path: Path, out: Path, answer: AnswerFn, model: str,
        root: Path, limit: int | None = None) -> Path:
    claims = pd.read_parquet(data / f"{split}.parquet")
    if limit:
        claims = claims.head(limit)
    scored = score_claims(claims, ICD11Cache(cache_path), answer)
    run_dir = out / f"{split}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "prompt.txt").write_text(PROMPT, encoding="utf-8")
    scored.to_json(run_dir / "answers.jsonl", orient="records", lines=True, force_ascii=False)
    summary = {"model": model, "split": split, "claims": len(scored), "temperature": 0.0,
               "inconsistent": int((scored["answer"] == "inconsistent").sum()),
               "consistent": int((scored["answer"] == "consistent").sum()),
               "unparseable": int(scored["answer"].isna().sum()), "limit": limit}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_predictions(SYSTEM, split, scored[["claim_id", "flagged"]], root)
    print(json.dumps(summary, indent=2))
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    r = commands.add_parser("run", help="ask the LLM about every claim of a split (GPU)")
    r.add_argument("--data", type=Path, required=True, help="processed dataset directory")
    r.add_argument("--split", default="validation")
    r.add_argument("--final", action="store_true", help="required to score the test split")
    r.add_argument("--cache", type=Path, required=True, help="ICD-11 cache JSON")
    r.add_argument("--out", type=Path, default=Path("experiments/zero_shot_llm"))
    r.add_argument("--model", default=DEFAULT_VLLM_MODEL)
    r.add_argument("--limit", type=int, help="first N claims only (state it when reporting)")
    c = commands.add_parser("combine", help="put the answers in the classifier's place in the funnel")
    c.add_argument("--split", default="validation")
    c.add_argument("--final", action="store_true", help="required for the test split")
    for sub in (r, c):
        sub.add_argument("--predictions-root", type=Path, default=PREDICTIONS_DIR)
    args = parser.parse_args()
    if args.split == "test" and not args.final:
        parser.error("the test split is reserved for the final evaluation; pass --final")
    if args.command == "run":
        run(args.data, args.split, args.cache, args.out, vllm_answers(args.model), args.model,
            args.predictions_root, args.limit)
        return
    root = args.predictions_root
    systems = funnel_predictions(read_predictions(SYSTEM, args.split, root),
                                 read_predictions("pipeline_stage1", args.split, root),
                                 read_predictions("pipeline_stage1_structural", args.split, root))
    for system, frame in systems.items():
        print(f"-> {write_predictions(system, args.split, frame, root)}")


if __name__ == "__main__":
    main()
