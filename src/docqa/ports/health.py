"""Health probe port: what GET /health reports about the database."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel


class DatabaseHealth(BaseModel):
    """Database reachability plus the status of every expected search index."""

    reachable: bool
    search_indexes: dict[str, str] = {}

    @property
    def healthy(self) -> bool:
        """True when the database answers and every search index is READY."""
        return (
            self.reachable
            and bool(self.search_indexes)
            and all(status == "READY" for status in self.search_indexes.values())
        )


class HealthProbe(Protocol):
    """Checks the database. Never raises: problems are reported in the result."""

    async def check(self) -> DatabaseHealth:
        """Return the current database health."""
        ...
