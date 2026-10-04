"""HybridRetriever: embed the question, then ask the store for fused candidates."""

from __future__ import annotations

import asyncio
import time

from docqa.domain.models import AccessFilter, HybridQuery, RetrievedChunk
from docqa.ports.embedder import DenseEmbedder
from docqa.ports.tracer import Tracer
from docqa.ports.vector_store import VectorStore


class HybridRetriever:
    """Turns a question into the store's top ``limit`` fused candidates."""

    def __init__(
        self, embedder: DenseEmbedder, store: VectorStore, limit: int, tracer: Tracer
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._limit = limit
        self._tracer = tracer

    async def retrieve(self, question: str, access: AccessFilter) -> list[RetrievedChunk]:
        """Embed the question in a worker thread, then run hybrid search (one span each)."""
        with self._tracer.span("embed_query"):
            vector = await asyncio.to_thread(self._embedder.embed_query, question)
        with self._tracer.span("retrieve") as span:
            start = time.perf_counter()
            results = await self._store.search(
                HybridQuery(text=question, vector=vector), access, self._limit
            )
            span.set(db_ms=int((time.perf_counter() - start) * 1000), candidates=len(results))
        return results
