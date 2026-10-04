"""Vector store port: write chunks, wait for search sync, hybrid search, delete."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from docqa.domain.models import AccessFilter, EmbeddedChunk, HybridQuery, RetrievedChunk


class VectorStore(Protocol):
    """Where chunks and their vectors live. Every search needs an AccessFilter."""

    async def ensure_indexes(self) -> None:
        """Create B-tree and search indexes and wait until they are READY."""
        ...

    async def upsert(self, chunks: Sequence[EmbeddedChunk]) -> None:
        """Insert or replace chunks by id, so retries never duplicate."""
        ...

    async def wait_until_searchable(
        self, doc_id: str, owner_id: str, expected: int, timeout_s: float = 30
    ) -> None:
        """Block until search returns ``expected`` chunks; raise IndexSyncTimeoutError."""
        ...

    async def search(
        self, query: HybridQuery, access: AccessFilter, limit: int
    ) -> list[RetrievedChunk]:
        """Hybrid search restricted to ``access``."""
        ...

    async def delete_document(self, doc_id: str, owner_id: str) -> int:
        """Delete every chunk of a document and return how many were removed."""
        ...
