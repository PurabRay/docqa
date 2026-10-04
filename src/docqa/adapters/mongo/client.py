"""The single AsyncMongoClient for the process, and PyMongo error translation.

``bootstrap.py`` creates one ``MongoConnection``; every other adapter receives
its ``db`` handle, never a URI. Only the PyMongo Async API is used (no Motor,
no sync client inside the event loop).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from pymongo import AsyncMongoClient
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.errors import AutoReconnect, NetworkTimeout, ServerSelectionTimeoutError

from docqa.config_schema import MongoConfig
from docqa.domain.errors import DatabaseUnavailableError

APP_NAME = "docqa"  # shows up in MongoDB server logs and Atlas metrics

Document = dict[str, Any]


@contextmanager
def translate_errors() -> Iterator[None]:
    """Turn PyMongo connectivity errors into DatabaseUnavailableError.

    Works around awaits too: ``with translate_errors(): await coll.find_one(...)``.
    """
    try:
        yield
    except (ServerSelectionTimeoutError, AutoReconnect, NetworkTimeout) as err:
        # The message names the error class only: PyMongo messages can contain hosts.
        raise DatabaseUnavailableError(f"MongoDB is unreachable ({type(err).__name__}).") from err


class MongoConnection:
    """Owns the AsyncMongoClient and the docqa database handle.

    Args:
        uri: MongoDB connection string (never logged).
        cfg: The ``mongodb`` config section.
    """

    def __init__(self, uri: str, cfg: MongoConfig) -> None:
        self.client: AsyncMongoClient[Document] = AsyncMongoClient(
            uri,
            appname=APP_NAME,
            maxPoolSize=cfg.max_pool_size,
            serverSelectionTimeoutMS=cfg.server_selection_timeout_ms,
            tz_aware=True,  # datetimes come back as aware UTC, matching the domain models
        )
        self.db: AsyncDatabase[Document] = self.client[cfg.database]

    async def ping(self) -> None:
        """Round-trip to the server.

        Raises:
            DatabaseUnavailableError: If the server cannot be reached.
        """
        with translate_errors():
            await self.client.admin.command("ping")

    async def close(self) -> None:
        """Close every pooled connection."""
        await self.client.close()
