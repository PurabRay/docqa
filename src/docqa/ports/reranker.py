"""Re-ranker port: re-score retrieved chunks against the question."""

from __future__ import annotations

from typing import Protocol

from docqa.domain.models import RetrievedChunk


class Reranker(Protocol):
    """A cross-encoder (or a no-op) that keeps the best ``top_k`` chunks. CPU-bound."""

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        """Return the ``top_k`` chunks with ``rerank_score`` set, best first."""
        ...
