"""ICD11APIClient (FR-06; Section 3.8.8). Accepts an ICD-11 code string, returns the
official code title, definition, and inclusion/exclusion criteria via the WHO ICD-11 API.
Used twice in the pipeline: as an input feature to the classifier (training + inference)
and, for claims labelled inconsistent, injected into the LLM narrative prompt (Section
2.2.6) — never for detection itself.

This project runs the API self-hosted (whoicd/icd-api Docker image — see
docker-compose.yml at repo root), not WHO's cloud instance, so no OAuth2 credentials are
needed: local deployments skip the client_id/client_secret token exchange the cloud API
requires. `base_url` just needs to match docker-compose.yml's host-side port mapping.

Retry/backoff behaviour per NFR-02: a small, fixed number of retries; on continued
failure, the caller should log the claim as failed rather than silently dropping or
crashing (IR-01) — this client raises ICD11APIError so callers can catch it explicitly.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from urllib.parse import quote

import requests

logger = logging.getLogger(__name__)


class ICD11APIError(Exception):
    """Raised when the ICD-11 API is unreachable or returns an error after all retries."""


@dataclass
class ICD11Definition:
    code: str
    title: str
    definition: str
    inclusion: list[str]
    exclusion: list[str]


class ICD11APIClient:
    def __init__(self, base_url: str, release: str = "2026-01", max_retries: int = 3,
                 backoff_seconds: float = 1.0, timeout_seconds: float = 10.0):
        self.base_url = base_url.rstrip("/")  # e.g. http://localhost:8081 — see config.yaml
        self.release = release
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.timeout_seconds = timeout_seconds
        self._headers = {"Accept": "application/json", "Accept-Language": "en", "API-Version": "v2"}

    def _get(self, url: str) -> dict:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.get(url, headers=self._headers, timeout=self.timeout_seconds)
                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:
                last_error = exc
                logger.warning("ICD-11 API request failed (attempt %d/%d): %s",
                               attempt, self.max_retries, exc)
                if attempt < self.max_retries:
                    time.sleep(self.backoff_seconds * attempt)
        raise ICD11APIError(f"ICD-11 API unreachable after {self.max_retries} attempts") from last_error

    def health_check(self) -> bool:
        """Confirms the local container is up before the pipeline starts relying on it."""
        try:
            self._get(f"{self.base_url}/icd/release/11/{self.release}/mms")
            return True
        except ICD11APIError:
            return False

    def get_code_definition(self, icd11_code: str) -> ICD11Definition:
        """Resolves a billed ICD-11 code to its official title/definition/inclusion/
        exclusion criteria. Two-step lookup per the ICD-11 API's own design: `codeinfo`
        resolves a code string to its entity URI (`stemId`), then that URI is fetched for
        the full entity record."""
        codeinfo_url = (
            f"{self.base_url}/icd/release/11/{self.release}/mms/codeinfo/"
            f"{quote(icd11_code, safe='')}"
        )
        codeinfo = self._get(codeinfo_url)
        stem_id = codeinfo.get("stemId")
        if not stem_id:
            raise ICD11APIError(f"No stemId returned for code {icd11_code!r}: {codeinfo}")

        entity = self._get(stem_id)
        return ICD11Definition(
            code=icd11_code,
            title=_extract_label(entity.get("title")),
            definition=_extract_label(entity.get("definition")),
            inclusion=[_extract_label(i) for i in entity.get("inclusion", [])],
            exclusion=[_extract_label(e.get("linearizationReference", e))
                       for e in entity.get("exclusion", [])],
        )


def _extract_label(value) -> str:
    """The ICD-11 API returns most text fields as {'@language': 'en', '@value': '...'}
    rather than plain strings; this normalises either shape to a plain string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get("@value", "")
    return str(value)
