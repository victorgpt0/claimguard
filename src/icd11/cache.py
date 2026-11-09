"""ICD-11 definition cache. ICD-11 code definitions are static reference data (they don't
change per-request), so there's no reason to hit the self-hosted container — or ever
require Colab to reach it — on every lookup. Build this cache once (or refresh it
periodically) against the local container, upload the resulting JSON file to
Google Drive alongside the rest of `artifacts/` (DR-05), and both classifier
fine-tuning (Colab) and the deployed dashboard read from it first, falling back to a
live client only when a code is genuinely missing.

See docs/environment.md's "Note: the ICD-11 container and Colab don't share a network"
for the full rationale.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from src.icd11.client import ICD11APIClient, ICD11APIError, ICD11Definition

logger = logging.getLogger(__name__)


class ICD11Cache:
    def __init__(self, cache_path: str | Path):
        self.cache_path = Path(cache_path)
        self._entries: dict[str, ICD11Definition] = {}
        if self.cache_path.exists():
            self.load()

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, code: str) -> bool:
        return code in self._entries

    def load(self) -> None:
        raw = json.loads(self.cache_path.read_text(encoding="utf-8"))
        self._entries = {
            code: ICD11Definition(**entry) for code, entry in raw.items()
        }

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        serialisable = {
            code: {
                "code": d.code,
                "title": d.title,
                "definition": d.definition,
                "inclusion": d.inclusion,
                "exclusion": d.exclusion,
                "definition_source": d.definition_source,
            }
            for code, d in self._entries.items()
        }
        self.cache_path.write_text(json.dumps(serialisable, indent=2, ensure_ascii=False),
                                    encoding="utf-8")

    def get(self, code: str, client: ICD11APIClient | None = None) -> ICD11Definition:
        """Cache-first lookup. Raises KeyError if the code is missing and no client was
        given to fetch it; raises ICD11APIError if the client call itself fails."""
        if code in self._entries:
            return self._entries[code]
        if client is None:
            raise KeyError(
                f"ICD-11 code {code!r} not in cache and no client provided to fetch it. "
                "Either pre-populate the cache (bulk_prefetch) or pass a live client."
            )
        definition = client.get_code_definition(code)
        self._entries[code] = definition
        self.save()
        return definition

    def bulk_prefetch(self, codes: list[str], client: ICD11APIClient,
                       save_every: int = 25) -> dict[str, str]:
        """Fetches every code in `codes` not already cached, from a live client — the
        operation you run occasionally against the local container to build/refresh the
        cache, per docs/environment.md's recommended workflow. Returns {code: error} for
        any codes that failed, rather than aborting the whole batch on one bad code."""
        missing = [c for c in codes if c not in self._entries]
        errors: dict[str, str] = {}
        logger.info("Prefetching %d of %d requested ICD-11 codes (%d already cached)",
                    len(missing), len(codes), len(codes) - len(missing))
        for i, code in enumerate(missing, start=1):
            try:
                self._entries[code] = client.get_code_definition(code)
            except ICD11APIError as exc:
                errors[code] = str(exc)
                logger.warning("Failed to prefetch %r: %s", code, exc)
            if i % save_every == 0:
                self.save()
        self.save()
        return errors
