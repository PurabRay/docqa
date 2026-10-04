"""TTLAnswerCache: in-process answer cache with expiry (cache.backend: memory)."""

from __future__ import annotations

import time
from collections.abc import Callable

from docqa.domain.models import Answer


class TTLAnswerCache:
    """Implements the AnswerCache port for a single API process.

    Args:
        ttl_seconds: How long an answer stays valid.
        clock: Returns monotonic seconds; injected for tests.
    """

    def __init__(self, ttl_seconds: float, clock: Callable[[], float] = time.monotonic) -> None:
        self._ttl = ttl_seconds
        self._clock = clock
        self._entries: dict[str, tuple[float, Answer, list[str]]] = {}

    async def get(self, key: str) -> Answer | None:
        """The cached answer, or None if missing or expired."""
        entry = self._entries.get(key)
        if entry is None:
            return None
        expires_at, answer, _ = entry
        if self._clock() >= expires_at:
            del self._entries[key]
            return None
        return answer

    async def set(self, key: str, answer: Answer, doc_ids: list[str]) -> None:
        """Store an answer for ttl_seconds."""
        self._entries[key] = (self._clock() + self._ttl, answer, list(doc_ids))

    async def purge_document(self, doc_id: str) -> int:
        """Remove every answer that depends on ``doc_id``."""
        doomed = [key for key, (_, _, doc_ids) in self._entries.items() if doc_id in doc_ids]
        for key in doomed:
            del self._entries[key]
        return len(doomed)
