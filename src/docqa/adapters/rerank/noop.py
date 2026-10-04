"""NoopReranker: keeps the fused order (re-ranker-off ablation and fast tests)."""

from __future__ import annotations

from docqa.domain.models import RetrievedChunk


class NoopReranker:
    """Implements the Reranker port without a model."""

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        """Return the first ``top_k`` chunks unchanged; abstention then uses fused scores."""
        return chunks[:top_k]
