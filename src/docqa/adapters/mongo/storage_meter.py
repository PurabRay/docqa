"""MongoStorageMeter: database size from dbStats and the live document count."""

from __future__ import annotations

from pymongo.asynchronous.database import AsyncDatabase

from docqa.adapters.mongo.client import Document, translate_errors
from docqa.adapters.mongo.names import DOCUMENTS
from docqa.domain.models import IngestionStatus
from docqa.ports.storage import StorageUsage

BYTES_PER_MB = 1024 * 1024


class MongoStorageMeter:
    """Implements the StorageMeter port."""

    def __init__(self, db: AsyncDatabase[Document]) -> None:
        self._db = db

    async def usage(self) -> StorageUsage:
        """Data + index storage in MB, and documents that are not deleted."""
        with translate_errors():
            stats = await self._db.command("dbStats")
            documents = await self._db[DOCUMENTS].count_documents(
                {"status": {"$ne": IngestionStatus.DELETED.value}}
            )
        size = stats.get("storageSize", 0) + stats.get("indexSize", 0)
        return StorageUsage(size_mb=size / BYTES_PER_MB, documents=documents)
