"""AuditNarrativeGenerator (FR-06; Section 3.6.1 Class Diagram). An LLM invoked ONLY for
claims the classifier labels inconsistent — never for detection. Accepts the clinical
notes string, the classifier's verdict, and the retrieved ICD-11 API definition; produces
a plain-language reasoning string.

This is the demoted role the supervisor's feedback established: the LLM explains a
verdict already reached by src/classification/classifier.py, it does not reach the
verdict itself. No task-specific fine-tuning required (Section 3.2.3) — pure
inference/prompting (IR-02).
"""
from __future__ import annotations

import os

PROMPT_TEMPLATE = """You are assisting a health-insurance fraud auditor. A claim's \
clinical justification has been flagged as INCONSISTENT with its billed procedure code \
by an automated classifier. Explain, in plain language a non-technical auditor can act \
on, why the clinical notes do not support the billed code, using the official code \
definition below. Do not invent clinical details not present in the notes.

Billed ICD-11 code definition: {icd11_definition}

Claim's clinical justification: {clinical_notes}

Classifier confidence: {confidence:.2f}

Write a 2-4 sentence plain-language explanation of the inconsistency."""


# 4-bit build of Llama 3.1 8B Instruct (Llama 3.1 Community License) that fits a Colab T4
# alongside the classifier. Not gated, unlike meta-llama's original repository.
DEFAULT_VLLM_MODEL = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"


class AuditNarrativeGenerator:
    def __init__(self, provider: str, model_name: str, gpu_memory_utilization: float = 0.7):
        """`provider` is "vllm" (open-weight model on the local GPU, used on Colab),
        "huggingface" (Hugging Face Inference Providers, needs account credit) or
        "openai" — see config.yaml."""
        self.provider = provider
        self.model_name = model_name
        # vLLM reserves this share of GPU memory up front; the rest is left for the
        # classifier, which runs on the same GPU.
        self.gpu_memory_utilization = gpu_memory_utilization
        self._vllm = None

    def generate(self, clinical_notes: str, verdict, icd11_definition: str) -> str:
        """Builds the structured prompt (clinical notes + verdict + ICD-11 definition)
        and returns the LLM's plain-language narrative. Prompt template developed
        iteratively across sprints (Section 3.2.3) based on the researcher's review of output
        quality."""
        if verdict.verdict != "inconsistent":
            raise ValueError(
                "AuditNarrativeGenerator must only be invoked for inconsistent verdicts "
                f"(got {verdict.verdict!r}) — the classifier gates this call, not the LLM."
            )
        prompt = PROMPT_TEMPLATE.format(
            icd11_definition=icd11_definition,
            clinical_notes=clinical_notes,
            confidence=verdict.confidence,
        )
        if self.provider == "vllm":
            return self._generate_vllm(prompt)
        if self.provider == "huggingface":
            return self._generate_huggingface(prompt)
        elif self.provider == "openai":
            return self._generate_openai(prompt)
        raise ValueError(f"Unknown LLM provider: {self.provider!r}")

    def _generate_vllm(self, prompt: str) -> str:
        """Loads the model on first use and keeps it for the rest of the run. Imported
        lazily: vLLM is installed only on Colab. float16 because the T4 has no bfloat16."""
        from vllm import LLM, SamplingParams

        if self._vllm is None:
            self._vllm = LLM(model=self.model_name, dtype="half", max_model_len=2048,
                             gpu_memory_utilization=self.gpu_memory_utilization,
                             max_num_batched_tokens=2048)
        params = SamplingParams(temperature=0.3, max_tokens=200, seed=0)
        outputs = self._vllm.chat([{"role": "user", "content": prompt}], params, use_tqdm=False)
        return outputs[0].outputs[0].text.strip()

    def _generate_huggingface(self, prompt: str) -> str:
        from huggingface_hub import InferenceClient

        # Chat API rather than text_generation: most models served by Hugging Face's
        # inference providers expose only the "conversational" task, and chat_completion
        # applies the model's own chat template.
        client = InferenceClient(model=self.model_name, token=os.environ.get("HF_TOKEN"))
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    def _generate_openai(self, prompt: str) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
