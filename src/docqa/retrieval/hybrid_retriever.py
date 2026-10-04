"""HybridRetriever: embed the question, then ask the store for fused candidates."""

from __future__ import annotations

import asyncio

from docqa.domain.models import AccessFilter, HybridQuery, RetrievedChunk
from docqa.ports.embedder import DenseEmbedder
from docqa.ports.vector_store import VectorStore


class HybridRetriever:
    """Turns a question into the store's top ``limit`` fused candidates."""

    def __init__(self, embedder: DenseEmbedder, store: VectorStore, limit: int) -> None:
        self._embedder = embedder
        self._store = store
        self._limit = limit

    async def retrieve(self, question: str, access: AccessFilter) -> list[RetrievedChunk]:
        """Embed the question in a worker thread and run hybrid search."""
        vector = await asyncio.to_thread(self._embedder.embed_query, question)
        query = HybridQuery(text=question, vector=vector)
        return await self._store.search(query, access, self._limit)
