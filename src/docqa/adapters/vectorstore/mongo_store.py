"""MongoVectorStore: chunk documents with float32 vectors in chunks_<model>.

Write side here; hybrid search lives in mongo_search.py.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence

from pymongo import ReplaceOne
from pymongo.asynchronous.database import AsyncDatabase

from docqa.adapters.mongo import codecs, indexes
from docqa.adapters.mongo.client import Document, translate_errors
from docqa.adapters.mongo.pipelines import text_count_pipeline, vector_count_pipeline
from docqa.adapters.vectorstore import mongo_search
from docqa.config_schema import MongoConfig, RetrievalConfig, SearchIndex
from docqa.domain.errors import EmbeddingVersionMismatchError, IndexSyncTimeoutError
from docqa.domain.models import AccessFilter, EmbeddedChunk, HybridQuery, RetrievedChunk


class MongoVectorStore:
    """Implements the VectorStore port.

    Args:
        db: Handle to the docqa database.
        cfg: The ``mongodb`` config section.
        retrieval: The ``retrieval`` config section (fusion mode, k values, weights).
        collection: Name of the chunks collection for the active model.
        embed_model: The configured embedding model; other models are refused.
        search_indexes: Definitions for ensure_indexes().
    """

    def __init__(
        self,
        db: AsyncDatabase[Document],
        cfg: MongoConfig,
        retrieval: RetrievalConfig,
        collection: str,
        embed_model: str,
        search_indexes: list[SearchIndex],
    ) -> None:
        self._chunks = db[collection]
        self._cfg = cfg
        self._retrieval = retrieval
        self._embed_model = embed_model
        self._search_indexes = search_indexes
        self._model_checked = False

    async def ensure_indexes(self) -> None:
        """Create or update both search indexes and wait until they are READY."""
        await indexes.ensure_search_indexes(self._chunks, self._search_indexes)
        await indexes.wait_until_ready(
            self._chunks,
            self._search_indexes,
            self._cfg.index_ready_timeout_s,
            self._cfg.index_poll_interval_s,
        )

    async def upsert(self, chunks: Sequence[EmbeddedChunk]) -> None:
        """Replace-or-insert chunks by _id in unordered bulk batches."""
        await self._check_model()
        if any(item.embed_model != self._embed_model for item in chunks):
            raise EmbeddingVersionMismatchError()
        ops = [
            ReplaceOne({"_id": item.chunk.id}, codecs.chunk_to_bson(item), upsert=True)
            for item in chunks
        ]
        with translate_errors():
            for start in range(0, len(ops), self._cfg.bulk_batch):
                batch = ops[start : start + self._cfg.bulk_batch]
                await self._chunks.bulk_write(batch, ordered=False)

    async def wait_until_searchable(
        self, doc_id: str, owner_id: str, expected: int, timeout_s: float = 30
    ) -> None:
        """Poll both search indexes until each returns ``expected`` chunks of the document.

        Raises:
            IndexSyncTimeoutError: If they have not caught up after ``timeout_s``.
        """
        if expected == 0:
            return
        access = AccessFilter(owner_id=owner_id, doc_ids=[doc_id])
        probe = await self._probe_vector(access)
        deadline = time.monotonic() + timeout_s
        while (counts := await self._search_counts(access, probe, expected)) != (
            expected,
            expected,
        ):
            if time.monotonic() > deadline:
                raise IndexSyncTimeoutError(
                    f"Indexed {counts} of {expected} chunks after {timeout_s:.0f}s."
                )
            await asyncio.sleep(self._cfg.sync_poll_interval_s)

    async def search(
        self, query: HybridQuery, access: AccessFilter, limit: int
    ) -> list[RetrievedChunk]:
        """Hybrid search restricted to ``access``; at most ``limit`` results, best first."""
        await self._check_model()
        results = await mongo_search.search(self._chunks, query, access, self._retrieval, self._cfg)
        return results[:limit]

    async def delete_document(self, doc_id: str, owner_id: str) -> int:
        """Delete every chunk of the document; return how many were deleted."""
        with translate_errors():
            result = await self._chunks.delete_many({"doc_id": doc_id, "owner_id": owner_id})
        return result.deleted_count

    async def _check_model(self) -> None:
        """Refuse a collection that already holds chunks from another embedding model."""
        if self._model_checked:
            return
        with translate_errors():
            other = await self._chunks.find_one(
                {"embed_model": {"$ne": self._embed_model}}, {"embed_model": 1}
            )
        if other is not None:
            raise EmbeddingVersionMismatchError(
                f"Collection holds {other['embed_model']!r} vectors, "
                f"config uses {self._embed_model!r}."
            )
        self._model_checked = True

    async def _probe_vector(self, access: AccessFilter) -> list[float]:
        with translate_errors():
            doc = await self._chunks.find_one(
                {"doc_id": access.doc_ids[0], "owner_id": access.owner_id}, {"embedding": 1}
            )
        if doc is None:
            raise IndexSyncTimeoutError("No chunks were written for this document.")
        return codecs.vector_from_bson(doc["embedding"])

    async def _search_counts(
        self, access: AccessFilter, probe: list[float], expected: int
    ) -> tuple[int, int]:
        """(chunks visible to $search, chunks visible to $vectorSearch)."""
        text_index, vector_index = self._cfg.text_index, self._cfg.vector_index
        with translate_errors():
            text = await (
                await self._chunks.aggregate(text_count_pipeline(access, text_index))
            ).to_list()
            vector_pipeline = vector_count_pipeline(probe, access, vector_index, expected)
            vector = await (await self._chunks.aggregate(vector_pipeline)).to_list()
        text_total = int(text[0]["count"]["total"]) if text else 0
        return text_total, len(vector)
