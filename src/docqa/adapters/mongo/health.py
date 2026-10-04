"""MongoHealthProbe: is MongoDB reachable, and are both search indexes READY?"""

from __future__ import annotations

from pymongo.errors import OperationFailure

from docqa.adapters.mongo.client import MongoConnection
from docqa.adapters.mongo.indexes import list_search_indexes
from docqa.domain.errors import DatabaseUnavailableError
from docqa.ports.health import DatabaseHealth


class MongoHealthProbe:
    """Implements the HealthProbe port for MongoDB.

    Args:
        connection: The process-wide MongoDB connection.
        chunks_collection: Collection that carries the search indexes.
        index_names: Search indexes that must be READY.
    """

    def __init__(
        self, connection: MongoConnection, chunks_collection: str, index_names: list[str]
    ) -> None:
        self._connection = connection
        self._chunks = connection.db[chunks_collection]
        self._index_names = index_names

    async def check(self) -> DatabaseHealth:
        """Ping the server and read search index statuses. Never raises."""
        try:
            await self._connection.ping()
            existing = await list_search_indexes(self._chunks)
        except DatabaseUnavailableError:
            return DatabaseHealth(reachable=False)
        except OperationFailure:
            # Reachable, but search is not available (e.g. a plain mongod without mongot).
            return DatabaseHealth(
                reachable=True, search_indexes=dict.fromkeys(self._index_names, "UNAVAILABLE")
            )
        statuses = {
            name: str(existing[name].get("status", "UNKNOWN")) if name in existing else "MISSING"
            for name in self._index_names
        }
        return DatabaseHealth(reachable=True, search_indexes=statuses)
