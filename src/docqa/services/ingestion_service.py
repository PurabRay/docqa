"""Ingestion use case: extract -> chunk -> embed -> upsert -> wait for sync -> ready.

Status moves queued -> extracting -> indexing -> syncing -> ready, or to failed
with the error message. Each step's duration is logged.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from docqa.config_schema import ChunkingConfig
from docqa.domain.errors import DocQAError
from docqa.domain.models import Chunk, EmbeddedChunk, IngestionStatus, IngestJob
from docqa.ingestion.chunker import chunk_pages
from docqa.ingestion.extractor import extract_pages
from docqa.ingestion.sanitizer import sanitize
from docqa.ports.embedder import DenseEmbedder
from docqa.ports.repository import DocumentRepository
from docqa.ports.vector_store import VectorStore

log = logging.getLogger(__name__)


class IngestionService:
    """Runs one ingestion job end to end."""

    def __init__(
        self,
        repo: DocumentRepository,
        store: VectorStore,
        embedder: DenseEmbedder,
        chunking: ChunkingConfig,
        injection_patterns: list[re.Pattern[str]],
        sync_timeout_s: float,
    ) -> None:
        self._repo = repo
        self._store = store
        self._embedder = embedder
        self._chunking = chunking
        self._patterns = injection_patterns
        self._sync_timeout_s = sync_timeout_s

    async def ingest(self, job: IngestJob) -> None:
        """Ingest the document; on a DocQAError mark it failed with the message."""
        try:
            await self._run(job)
        except DocQAError as err:
            log.warning("ingest.failed doc_id=%s code=%s", job.doc_id, err.code)
            await self._repo.set_status(job.doc_id, IngestionStatus.FAILED, error=err.message)

    async def _run(self, job: IngestJob) -> None:
        await self._repo.set_status(job.doc_id, IngestionStatus.EXTRACTING)
        with _timed("extract_and_chunk", job.doc_id):
            chunks = await asyncio.to_thread(self.prepare_chunks, job)

        await self._repo.set_status(job.doc_id, IngestionStatus.INDEXING)
        with _timed("embed", job.doc_id):
            vectors = await asyncio.to_thread(
                self._embedder.embed_documents, [c.text for c in chunks]
            )
        with _timed("upsert", job.doc_id):
            await self._store.upsert(
                [
                    EmbeddedChunk(
                        chunk=chunk, embedding=vector, embed_model=self._embedder.model_id
                    )
                    for chunk, vector in zip(chunks, vectors, strict=True)
                ]
            )

        await self._repo.set_status(job.doc_id, IngestionStatus.SYNCING)
        with _timed("sync", job.doc_id):
            await self._store.wait_until_searchable(
                job.doc_id, job.owner_id, len(chunks), self._sync_timeout_s
            )
        await self._repo.set_status(job.doc_id, IngestionStatus.READY, chunk_count=len(chunks))

    def prepare_chunks(self, job: IngestJob) -> list[Chunk]:
        """Extract, sanitise and chunk the PDF (CPU-bound; runs in a thread)."""
        pages = [
            sanitize(page, self._patterns) for page in extract_pages(Path(job.path), job.doc_id)
        ]
        return chunk_pages(
            pages, self._chunking, doc_id=job.doc_id, owner_id=job.owner_id, filename=job.filename
        )


@contextmanager
def _timed(step: str, doc_id: str) -> Iterator[None]:
    start = time.perf_counter()
    yield
    log.info("ingest.%s doc_id=%s ms=%d", step, doc_id, (time.perf_counter() - start) * 1000)
