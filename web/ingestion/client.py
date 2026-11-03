"""The audit system's side of IR-09: a client for the claims platform's API (sha_cdp/app.py).
It knows only the API, never the platform's database. Settings: SHA_CDP_URL, SHA_CDP_API_KEY
(read-only), and SHA_CDP_ADMIN_TOKEN for the simulation clock controls."""
from __future__ import annotations

from dataclasses import dataclass

import requests
from django.conf import settings

TIMEOUT = 15


class PlatformError(Exception):
    """The platform could not be reached or refused the request."""


@dataclass
class Page:
    as_of: str
    claims: list[dict]
    has_more: bool
    next: dict | None


class PlatformClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None, admin_token: str | None = None,
                 session: requests.Session | None = None):
        self.base_url = (base_url or settings.SHA_CDP_URL).rstrip("/")
        self.api_key = api_key if api_key is not None else settings.SHA_CDP_API_KEY
        self.admin_token = admin_token if admin_token is not None else settings.SHA_CDP_ADMIN_TOKEN
        self.session = session or requests.Session()

    def _call(self, method: str, path: str, token: str, timeout: float = TIMEOUT, **kwargs) -> dict:
        try:
            response = self.session.request(method, f"{self.base_url}{path}", timeout=timeout,
                                            headers={"Authorization": f"Bearer {token}"}, **kwargs)
        except requests.RequestException as exc:
            raise PlatformError(f"claims platform unreachable at {self.base_url}: {type(exc).__name__}") from None
        if response.status_code == 401:
            raise PlatformError("the claims platform refused the key (SHA_CDP_API_KEY / SHA_CDP_ADMIN_TOKEN)")
        if response.status_code >= 400:
            raise PlatformError(f"claims platform answered {response.status_code}: {response.text[:200]}")
        return response.json()

    def paid_claims(self, after_paid_at: str | None, after_seq: int, limit: int = 500) -> Page:
        params = {"after_seq": after_seq, "limit": limit}
        if after_paid_at:
            params["after_paid_at"] = after_paid_at
        body = self._call("GET", "/api/v1/claims/paid", self.api_key, params=params)
        return Page(body["as_of"], body["claims"], body["has_more"], body["next"])

    def stats(self) -> dict:
        # Short timeout: the pipeline page shows these and must not hang on a slow platform.
        return self._call("GET", "/api/v1/stats", self.api_key, timeout=3)

    def change_clock(self, **change) -> dict:
        return self._call("POST", "/api/v1/clock", self.admin_token, json=change)
