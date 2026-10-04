"""Storage meter port: how full the database is, for the storage guard."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel


class StorageUsage(BaseModel):
    """Data plus index size, and how many documents are stored."""

    size_mb: float
    documents: int


class StorageMeter(Protocol):
    """Reads current storage usage."""

    async def usage(self) -> StorageUsage:
        """Return the current usage."""
        ...
