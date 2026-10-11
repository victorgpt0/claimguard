"""Synthetic clinical-note generation, stage B of dataset construction (Section 3.2.1).

An open-weight LLM writes the free-text clinical justification for every claim, from the
request stage A wrote (src/data/generator.py). This is data synthesis only: the LLM never
sees fraud labels and plays no part in detection, which stays with the fine-tuned
classifier. Notes describe each claim's true clinical situation; stage C then makes some of
them inconsistent by upcoding the billed code or swapping the note.

Why an LLM rather than templates: if consistent notes came from a few fixed templates, the
classifier could learn to recognise the templates instead of judging note-code consistency.
Prompts vary the writing style, detail and structure per claim to widen that diversity.

Runs on Colab with vLLM (GPU). Output is appended per batch, so a disconnected runtime can
resume where it stopped.

Usage (Colab):
  python -m src.data.note_synthesis --requests data/raw/note_requests.jsonl \
      --out data/raw/notes.jsonl --boilerplate-out data/raw/boilerplate_notes.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Callable

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"
# Used with the transformers backend: AWQ checkpoints need extra kernels there, and a 3B
# model in float16 fits a 16 GB T4 comfortably.
FALLBACK_MODEL = "Qwen/Qwen2.5-3B-Instruct"

SYSTEM_PROMPT = (
    "You are a clinician at a Kenyan health facility writing the clinical justification "
    "that accompanies an insurance claim to the Social Health Authority. Write only the "
    "note itself: no title, no preamble, no ICD codes, no patient names."
)

# Each claim gets one style, chosen by a seeded draw, so the corpus mixes registers the way
# real claim documentation does.
STYLES = [
    "Write in terse clinical shorthand, as a busy clinician would.",
    "Write a short narrative paragraph in full sentences.",
    "Use a brief SOAP structure (S:, O:, A:, P:) on separate lines.",
    "Write as a short discharge summary.",
    "Write two or three short sentences covering presentation and management only.",
]
DETAIL = [
    "Mention one or two specific examination findings or vital signs.",
    "Mention a relevant investigation and its result.",
    "Mention the main treatment or procedure given.",
    "Mention the presenting complaint and its duration.",
]

SETTINGS = {
    "outpatient": "seen as an outpatient at a {facility} and not admitted",
    "medical_inpatient": "admitted to a {facility} for {stay} day(s)",
    "maternity": "admitted to a {facility} for delivery and stayed {stay} day(s)",
    "surgical": "admitted to a {facility} for {procedure} and stayed {stay} day(s)",
}

BOILERPLATE_PROMPT = (
    "Write a generic clinical note of one or two sentences that could be pasted onto any "
    "claim for any patient. It must not mention any symptom, body part, finding, test, "
    "diagnosis, drug or procedure. Examples of the kind of note wanted:\n"
    "- Patient reviewed and managed as per protocol. Stable for discharge.\n"
    "- Seen, assessed and treated accordingly. To review if need arises.\n"
    "Write a new one in the same spirit, not a copy. {style}"
)
BOILERPLATE_STYLES = [
    "Make it sound routine.", "Use clinical shorthand.", "Make it sound formal.",
    "Make it very short.", "Mention follow-up in vague terms.",
]

# ICD-11 stem codes: chapter character, a letter, a digit, then a letter or digit, with an
# optional extension (1F40.Z, CA40, JB22.1). Case-sensitive, so ordinary words don't match.
_ICD_CODE = re.compile(r"\b[0-9A-Z][A-Z][0-9][0-9A-Z](\.[0-9A-Z]{1,4})?\b")
_PREFIX = re.compile(r"^\s*(clinical (note|justification)|note|justification)\s*:\s*", re.IGNORECASE)


def build_prompt(request: dict, seed: int) -> list[dict]:
    """Chat messages for one claim's note. Deterministic given (request, seed)."""
    rng = random.Random(f"{seed}:{request['claim_id']}")
    facility = request["facility_type"].replace("_", " ")
    setting = SETTINGS[request["package"]].format(
        facility=facility, stay=request["length_of_stay"], procedure=request["procedure"].lower()
    )
    user = (
        f"Patient: {request['patient_age']}-year-old {request['patient_sex']}, {setting}.\n"
        f"Condition: {request['diagnosis']}.\n"
        f"{rng.choice(STYLES)} {rng.choice(DETAIL)} Keep it under 90 words. You may name "
        f"the working diagnosis in ordinary clinical words, but describe the clinical "
        f"picture rather than restating the condition name word for word."
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def build_boilerplate_prompt(index: int, seed: int) -> list[dict]:
    rng = random.Random(f"{seed}:boilerplate:{index}")
    user = BOILERPLATE_PROMPT.format(style=rng.choice(BOILERPLATE_STYLES))
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def clean_note(text: str) -> str:
    """Strips quoting and 'Clinical note:'-style prefixes the model sometimes adds."""
    text = text.strip().strip('"').strip()
    return _PREFIX.sub("", text).strip()


# Words that make a note clinically specific. A boilerplate note containing any of these
# describes some presentation, so it is not generic and is regenerated. Catalogue
# diagnosis words are added at runtime (see boilerplate_vocabulary).
CLINICAL_TERMS = {
    "pain", "fever", "cough", "vomit", "vomiting", "diarrhoea", "diarrhea", "headache",
    "bleeding", "swelling", "rash", "breath", "breathing", "chest", "abdomen", "abdominal",
    "head", "leg", "arm", "back", "throat", "skin", "eye", "blood", "pressure", "temperature",
    "pulse", "x-ray", "xray", "scan", "ultrasound", "mri", "ct", "culture", "test", "lab",
    "antibiotic", "antibiotics", "paracetamol", "analgesic", "analgesics", "fluids", "iv",
    "surgery", "operation", "theatre", "delivery", "labour", "pregnancy", "infection",
    "fracture", "wound", "tumour", "tumor", "lesion", "malaria", "pneumonia", "diabetes",
    "hypertension", "asthma", "anaemia", "anemia", "sepsis", "dehydration", "ward", "icu",
}
_STOPWORDS = {"unspecified", "specified", "other", "without", "with", "single", "acute",
              "due", "or", "of", "the", "and", "not", "known", "if", "in", "by", "to"}


def boilerplate_vocabulary(diagnoses: list[str]) -> set[str]:
    words = {w for title in diagnoses for w in re.findall(r"[a-z]+", title.lower())
             if len(w) >= 4 and w not in _STOPWORDS}
    return CLINICAL_TERMS | words


def validate_boilerplate(note: str, vocabulary: set[str] = CLINICAL_TERMS) -> str | None:
    """Returns a reason the note is not generic boilerplate, or None. Record fabrication
    (Table 3.1 row 5) is defined by notes that describe no plausible presentation, so
    this is checked rather than trusted to the prompt."""
    words = re.findall(r"[a-z][a-z-]*", note.lower())
    if len(words) < 5:
        return "too short"
    if len(words) > 40:
        return "too long"
    if re.search(r"[0-9*#\[\]]", note):
        return "contains digits or markup"
    specific = sorted(set(words) & vocabulary)
    if specific:
        return f"clinically specific: {', '.join(specific[:3])}"
    return None


def validate_note(note: str, request: dict | None = None) -> str | None:
    """Returns a reason the note is unusable, or None. A note that contains its billed
    ICD-11 code would hand the classifier the answer, so it is rejected."""
    words = len(note.split())
    if words < 5:
        return "too short"
    if words > 150:
        return "too long"
    if request is not None and request["icd11_code"].lower() in note.lower():
        return "contains the ICD-11 code"
    if _ICD_CODE.search(note):
        return "contains an ICD-like code"
    return None


Generate = Callable[[list[list[dict]], list[int]], list[str]]


Check = Callable[[int, str], "str | None"]


def synthesise(prompts: list[list[dict]], seeds: list[int], generate: Generate, check: Check,
               max_attempts: int = 3) -> list[str]:
    """Runs `generate` over all prompts, regenerating (with a new seed) any note for which
    `check(index, note)` returns a failure reason. A note still failing after
    `max_attempts` is kept; `run` records the reason beside it so stage C can drop it."""
    notes = [clean_note(n) for n in generate(prompts, seeds)]
    for attempt in range(1, max_attempts):
        retry = [i for i, n in enumerate(notes) if check(i, n)]
        if not retry:
            break
        regenerated = generate([prompts[i] for i in retry], [seeds[i] + 1000 * attempt for i in retry])
        for i, note in zip(retry, regenerated):
            notes[i] = clean_note(note)
    return notes


def vllm_generator(model: str, temperature: float = 0.8, max_tokens: int = 200) -> Generate:
    """Batched generation with vLLM. Imported lazily: vLLM is installed only on Colab.
    float16 because the T4 has no bfloat16 support. On the 15 GB T4 the default 8,192-token
    prefill chunk ran out of memory in the MLP; prompts are ~150 tokens, so a 2,048-token
    chunk loses little throughput. Quantization is left to vLLM, which picks the faster
    awq_marlin kernel for AWQ checkpoints when the GPU supports it."""
    from vllm import LLM, SamplingParams

    llm = LLM(model=model, dtype="half", max_model_len=1024, gpu_memory_utilization=0.85,
              max_num_batched_tokens=2048, max_num_seqs=128)

    def generate(prompts: list[list[dict]], seeds: list[int]) -> list[str]:
        params = [SamplingParams(temperature=temperature, top_p=0.95, max_tokens=max_tokens, seed=s)
                  for s in seeds]
        outputs = llm.chat(prompts, params, use_tqdm=True)
        return [o.outputs[0].text for o in outputs]

    return generate


def transformers_generator(model: str, temperature: float = 0.8, max_tokens: int = 200,
                           batch_size: int = 32) -> Generate:
    """Fallback when vLLM will not install or start on the assigned GPU. Slower, but plain
    transformers runs anywhere. Seeds are applied per mini-batch, so output depends on the
    batch composition as well as the seeds; the run is still reproducible end to end."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, padding_side="left")
    llm = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float16, device_map="auto")

    def generate(prompts: list[list[dict]], seeds: list[int]) -> list[str]:
        notes = []
        for start in range(0, len(prompts), batch_size):
            chunk = prompts[start:start + batch_size]
            torch.manual_seed(seeds[start])
            texts = [tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
                     for p in chunk]
            inputs = tokenizer(texts, return_tensors="pt", padding=True).to(llm.device)
            with torch.no_grad():
                output = llm.generate(**inputs, do_sample=True, temperature=temperature, top_p=0.95,
                                      max_new_tokens=max_tokens, pad_token_id=tokenizer.eos_token_id)
            new_tokens = output[:, inputs["input_ids"].shape[1]:]
            notes.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
        return notes

    return generate


def _done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["claim_id"] for line in path.open()}


def run(requests_path: Path, out_path: Path, boilerplate_out: Path, generate: Generate,
        seed: int = 42, n_boilerplate: int = 300, batch_size: int = 500,
        meta: dict | None = None) -> None:
    requests = [json.loads(line) for line in requests_path.open()]
    done = _done_ids(out_path)
    pending = [r for r in requests if r["claim_id"] not in done]
    print(f"{len(done)} notes already written, {len(pending)} to generate")

    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        prompts = [build_prompt(r, seed) for r in batch]
        seeds = [seed + int(r["claim_id"][3:]) for r in batch]
        notes = synthesise(prompts, seeds, generate, lambda i, n: validate_note(n, batch[i]))
        with out_path.open("a") as f:
            for request, note in zip(batch, notes):
                f.write(json.dumps({"claim_id": request["claim_id"], "clinical_notes": note,
                                    "rejected_reason": validate_note(note, request)}) + "\n")
        print(f"  {start + len(batch)}/{len(pending)}")

    if not boilerplate_out.exists():
        vocabulary = boilerplate_vocabulary(sorted({r["diagnosis"] for r in requests}))
        prompts = [build_boilerplate_prompt(i, seed) for i in range(n_boilerplate)]
        notes = synthesise(prompts, [seed + i for i in range(n_boilerplate)], generate,
                           lambda i, n: validate_boilerplate(n, vocabulary), max_attempts=5)
        with boilerplate_out.open("w") as f:
            for note in notes:
                f.write(json.dumps({"clinical_notes": note,
                                    "rejected_reason": validate_boilerplate(note, vocabulary)}) + "\n")

    if meta is not None:
        out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--boilerplate-out", type=Path, required=True)
    parser.add_argument("--backend", choices=["vllm", "transformers"], default="vllm")
    parser.add_argument("--model", default=None,
                        help=f"default: {DEFAULT_MODEL} (vllm) or {FALLBACK_MODEL} (transformers)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-boilerplate", type=int, default=300)
    args = parser.parse_args()

    model = args.model or (DEFAULT_MODEL if args.backend == "vllm" else FALLBACK_MODEL)
    generate = vllm_generator(model) if args.backend == "vllm" else transformers_generator(model)
    meta = {"backend": args.backend, "model": model, "seed": args.seed, "temperature": 0.8,
            "top_p": 0.95, "max_tokens": 200, "system_prompt": SYSTEM_PROMPT}
    run(args.requests, args.out, args.boilerplate_out, generate,
        seed=args.seed, n_boilerplate=args.n_boilerplate, meta=meta)


if __name__ == "__main__":
    main()
