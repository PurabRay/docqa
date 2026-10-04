"""Composition root: the one place that builds concrete objects and wires them together.

Nothing else in DocQA constructs clients or adapters. The API lifespan calls
``build_container`` on startup and ``Container.close`` on shutdown.
"""

from __future__ import annotations

from dataclasses import dataclass

from docqa.adapters.mongo.client import MongoConnection
from docqa.adapters.mongo.health import MongoHealthProbe
from docqa.adapters.mongo.indexes import IndexAction, initialize_database
from docqa.adapters.storage.mongo_repository import MongoDocumentRepository
from docqa.ports.health import HealthProbe
from docqa.ports.repository import DocumentRepository
from docqa.settings import Settings


@dataclass
class Container:
    """Every long-lived object the app needs."""

    settings: Settings
    mongo: MongoConnection
    repository: DocumentRepository
    health: HealthProbe

    async def close(self) -> None:
        """Release connections."""
        await self.mongo.close()


def build_container(settings: Settings) -> Container:
    """Build the container from settings. Opens no network connection yet.

    Raises:
        ConfigurationError: If MONGODB_URI (or the configured uri_env) is missing.
    """
    mongo = MongoConnection(settings.mongodb_uri(), settings.mongodb)
    index_names = [settings.mongodb.vector_index, settings.mongodb.text_index]
    return Container(
        settings=settings,
        mongo=mongo,
        repository=MongoDocumentRepository(mongo.db),
        health=MongoHealthProbe(mongo, settings.chunks_collection, index_names),
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
