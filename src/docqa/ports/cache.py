"""Answer cache port: exact-match cache keyed by documents and prompt version (async, B10)."""

from __future__ import annotations

from typing import Protocol

from docqa.domain.models import Answer


class AnswerCache(Protocol):
    """Caches final answers; entries that depend on a deleted document are purged."""

    async def get(self, key: str) -> Answer | None:
        """Return the cached answer for ``key``, or None."""
        ...

    async def set(self, key: str, answer: Answer, doc_ids: list[str]) -> None:
        """Cache ``answer`` and remember which documents it depends on."""
        ...

    async def purge_document(self, doc_id: str) -> int:
        """Drop every entry that depends on ``doc_id`` and return how many."""
        ...
