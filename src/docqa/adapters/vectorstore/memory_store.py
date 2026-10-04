"""MemoryVectorStore: the VectorStore port in plain Python, for unit and contract tests.

Cosine similarity for the vector branch, rank_bm25 for the text branch, rrf_fuse to
combine. The access filter is applied BEFORE scoring, exactly like the pre-filters in
MongoDB, so other owners' chunks are never ranked.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

from rank_bm25 import BM25Okapi

from docqa.adapters.mongo.pipelines import vector_filter
from docqa.config_schema import RetrievalConfig
from docqa.domain.models import AccessFilter, EmbeddedChunk, HybridQuery, RetrievedChunk
from docqa.retrieval.rrf import rrf_fuse

WORD = re.compile(r"\w+")


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity; 0 for a zero vector."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norms = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norms if norms else 0.0


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens for BM25."""
    return WORD.findall(text.lower())


class MemoryVectorStore:
    """Implements the VectorStore port in memory."""

    def __init__(self, retrieval: RetrievalConfig) -> None:
        self._cfg = retrieval
        self._chunks: dict[str, EmbeddedChunk] = {}

    async def ensure_indexes(self) -> None:
        """Nothing to build."""

    async def upsert(self, chunks: Sequence[EmbeddedChunk]) -> None:
        """Insert or replace by chunk id."""
        for item in chunks:
            self._chunks[item.chunk.id] = item

    async def wait_until_searchable(
        self, doc_id: str, owner_id: str, expected: int, timeout_s: float = 30
    ) -> None:
        """Writes are visible at once."""

    async def search(
        self, query: HybridQuery, access: AccessFilter, limit: int
    ) -> list[RetrievedChunk]:
        """Hybrid search over the allowed chunks only."""
        vector_filter(access)  # same AccessFilterMissingError as the Mongo pipelines
        allowed = [
            item
            for item in self._chunks.values()
            if item.chunk.owner_id == access.owner_id and item.chunk.doc_id in access.doc_ids
        ]
        fused = rrf_fuse(
            {
                "vector": self._vector_ranking(query, allowed),
                "text": self._text_ranking(query, allowed),
            },
            weights={"vector": self._cfg.weights.vector, "text": self._cfg.weights.text},
        )
        return [
            RetrievedChunk(
                chunk=self._chunks[item.id].chunk,
                fused_score=item.score,
                vector_rank=item.ranks.get("vector"),
                text_rank=item.ranks.get("text"),
            )
            for item in fused[: min(limit, self._cfg.fused_k)]
        ]

    async def delete_document(self, doc_id: str, owner_id: str) -> int:
        """Delete the document's chunks and return how many."""
        doomed = [
            cid
            for cid, item in self._chunks.items()
            if item.chunk.doc_id == doc_id and item.chunk.owner_id == owner_id
        ]
        for cid in doomed:
            del self._chunks[cid]
        return len(doomed)

    def _vector_ranking(self, query: HybridQuery, allowed: list[EmbeddedChunk]) -> list[str]:
        ranked = sorted(
            allowed, key=lambda item: cosine(query.vector, item.embedding), reverse=True
        )
        return [item.chunk.id for item in ranked[: self._cfg.vector_k]]

    def _text_ranking(self, query: HybridQuery, allowed: list[EmbeddedChunk]) -> list[str]:
        if not allowed:
            return []
        scores = BM25Okapi([tokenize(item.chunk.text) for item in allowed]).get_scores(
            tokenize(query.text)
        )
        ranked = sorted(zip(scores, allowed, strict=True), key=lambda pair: pair[0], reverse=True)
        # Like $search, only chunks that match at least one query term are returned.
        query_terms = set(tokenize(query.text))
        matching = [item for _, item in ranked if query_terms & set(tokenize(item.chunk.text))]
        return [item.chunk.id for item in matching[: self._cfg.text_k]]
