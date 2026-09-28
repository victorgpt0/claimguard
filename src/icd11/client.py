"""ICD11APIClient (FR-06; Section 3.8.8). Accepts an ICD-11 code string, returns the
official code title, definition, and inclusion/exclusion criteria via the WHO ICD-11 API
(icd.who.int/icdapi). Used twice in the pipeline: as an input feature to the classifier
(training + inference) and, for claims labelled inconsistent, injected into the LLM
narrative prompt (Section 2.2.6) — never for detection itself.

Credentials come from ICD11_CLIENT_ID / ICD11_CLIENT_SECRET env vars — see .env.example.
Retry/backoff behaviour per NFR-02: a small, fixed number of retries; on continued
failure, log the claim as failed rather than silently dropping or crashing (IR-01).
"""
import os


class ICD11APIClient:
    def __init__(self, base_url: str, max_retries: int = 3):
        self.base_url = base_url
        self.max_retries = max_retries
        self.client_id = os.environ.get("ICD11_CLIENT_ID")
        self.client_secret = os.environ.get("ICD11_CLIENT_SECRET")

    def get_code_definition(self, icd11_code: str) -> dict:
        """Returns {'title': ..., 'definition': ..., 'inclusion': [...], 'exclusion': [...]}."""
        raise NotImplementedError
