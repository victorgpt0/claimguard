"""AuditNarrativeGenerator (FR-06; Section 3.6.1 Class Diagram). An LLM invoked ONLY for
claims the classifier labels inconsistent — never for detection. Accepts the clinical
notes string, the classifier's verdict, and the retrieved ICD-11 API definition; produces
a plain-language reasoning string.

This is the demoted role the supervisor's feedback established: the LLM explains a
verdict already reached by src/classification/classifier.py, it does not reach the
verdict itself. No task-specific fine-tuning required (Section 3.2.3) — pure
inference/prompting (IR-02).
"""


class AuditNarrativeGenerator:
    def __init__(self, provider: str, model_name: str):
        self.provider = provider  # "huggingface" | "openai" — see config.yaml
        self.model_name = model_name

    def generate(self, clinical_notes: str, verdict, icd11_definition: dict) -> str:
        """Builds the structured prompt (clinical notes + verdict + ICD-11 definition)
        and returns the LLM's plain-language narrative. Prompt template developed
        iteratively across sprints (Section 3.2.3) based on domain-reviewer feedback."""
        raise NotImplementedError
