"""Refuse new uploads once the database nears the free cluster's 512 MB."""

from __future__ import annotations

from docqa.domain.errors import StorageCapReachedError
from docqa.ports.storage import StorageMeter, StorageUsage


def check_capacity(usage: StorageUsage, cap_mb: float, max_documents: int) -> None:
    """Raise StorageCapReachedError if either limit is reached."""
    if usage.size_mb >= cap_mb:
        raise StorageCapReachedError(f"Storage is at {usage.size_mb:.0f} MB (cap {cap_mb:.0f} MB).")
    if usage.documents >= max_documents:
        raise StorageCapReachedError(f"The {max_documents}-document limit is reached.")


class StorageGuard:
    """Checks capacity before an upload is accepted."""

    def __init__(self, meter: StorageMeter, cap_mb: float, max_documents: int) -> None:
        self._meter = meter
        self._cap_mb = cap_mb
        self._max_documents = max_documents

    async def ensure_capacity(self) -> None:
        """Raise StorageCapReachedError if there is no room for another document."""
        check_capacity(await self._meter.usage(), self._cap_mb, self._max_documents)
