"""Composition root: the one place that builds concrete objects and wires them together.

Nothing else in DocQA constructs clients or adapters. The API lifespan calls
``build_container``, then ``Container.start`` and finally ``Container.close``.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from pathlib import Path

from docqa.adapters.embedding.fastembed_dense import FastEmbedDenseEmbedder
from docqa.adapters.mongo.client import MongoConnection
from docqa.adapters.mongo.health import MongoHealthProbe
from docqa.adapters.mongo.indexes import IndexAction, initialize_database
from docqa.adapters.mongo.storage_meter import MongoStorageMeter
from docqa.adapters.storage.mongo_repository import MongoDocumentRepository
from docqa.adapters.vectorstore.mongo_store import MongoVectorStore
from docqa.guardrails.storage_guard import StorageGuard
from docqa.ingestion.sanitizer import compile_patterns
from docqa.ingestion.worker import IngestionWorker
from docqa.ports.health import HealthProbe
from docqa.ports.repository import DocumentRepository
from docqa.services.document_service import DocumentService
from docqa.services.ingestion_service import IngestionService
from docqa.settings import Settings


@dataclass
class Container:
    """Every long-lived object the app needs."""

    settings: Settings
    mongo: MongoConnection
    repository: DocumentRepository
    health: HealthProbe
    store: MongoVectorStore
    ingestion: IngestionService
    worker: IngestionWorker
    documents: DocumentService
    _worker_task: asyncio.Task[None] | None = field(default=None, repr=False)

    def start(self) -> None:
        """Start the background ingestion worker (needs a running event loop)."""
        self._worker_task = asyncio.create_task(self.worker.run_forever())

    async def close(self) -> None:
        """Stop the worker and release connections."""
        if self._worker_task is not None:
            self._worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker_task
        await self.mongo.close()


def build_container(settings: Settings) -> Container:
    """Build the container from settings. Opens no network connection yet.

    Raises:
        ConfigurationError: If MONGODB_URI (or the configured uri_env) is missing.
    """
    mongo = MongoConnection(settings.mongodb_uri(), settings.mongodb)
    repository = MongoDocumentRepository(mongo.db)
    store = MongoVectorStore(
        mongo.db,
        settings.mongodb,
        settings.chunks_collection,
        settings.embedding.dense_model,
        list(settings.search_indexes.values()),
    )
    ingestion = IngestionService(
        repository,
        store,
        FastEmbedDenseEmbedder(settings.embedding),
        settings.chunking,
        compile_patterns(settings.ingestion.injection_patterns),
        settings.mongodb.sync_timeout_s,
    )
    worker = IngestionWorker(ingestion, settings.ingestion.queue_size)
    guard = StorageGuard(
        MongoStorageMeter(mongo.db), settings.mongodb.storage_cap_mb, settings.mongodb.max_documents
    )
    documents = DocumentService(
        repository,
        store,
        guard,
        worker.submit,
        settings.limits,
        Path(settings.ingestion.upload_dir),
        settings.embedding.dense_model,
    )
    index_names = [settings.mongodb.vector_index, settings.mongodb.text_index]
    return Container(
        settings=settings,
        mongo=mongo,
        repository=repository,
        health=MongoHealthProbe(mongo, settings.chunks_collection, index_names),
        store=store,
        ingestion=ingestion,
        worker=worker,
        documents=documents,
    )


async def init_database(container: Container) -> dict[str, IndexAction]:
    """Create every collection and index from config and wait for search indexes."""
    settings = container.settings
    return await initialize_database(
        container.mongo.db,
        chunks_collection=settings.chunks_collection,
        btree=settings.btree_indexes,
        search_indexes=list(settings.search_indexes.values()),
        timeout_s=settings.mongodb.index_ready_timeout_s,
        poll_interval_s=settings.mongodb.index_poll_interval_s,
    )
