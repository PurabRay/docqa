"""MongoTTLCache: answer cache in the answer_cache collection (cache.backend: mongo).

For more than one API replica. MongoDB's TTL monitor deletes expired entries (it runs
about once a minute), so get() also checks expires_at itself.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta

from pymongo.asynchronous.database import AsyncDatabase

from docqa.adapters.mongo.client import Document, translate_errors
from docqa.adapters.mongo.names import ANSWER_CACHE
from docqa.domain.models import Answer, utc_now


class MongoTTLCache:
    """Implements the AnswerCache port on MongoDB.

    Args:
        db: Handle to the docqa database.
        ttl_seconds: How long an answer stays valid.
        clock: Returns "now" (aware UTC); injected for tests.
    """

    def __init__(
        self,
        db: AsyncDatabase[Document],
        ttl_seconds: float,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self._entries = db[ANSWER_CACHE]
        self._ttl = timedelta(seconds=ttl_seconds)
        self._clock = clock

    async def ensure_indexes(self) -> None:
        """TTL index on expires_at and a lookup index on doc_ids."""
        with translate_errors():
            await self._entries.create_index("expires_at", expireAfterSeconds=0)
            await self._entries.create_index("doc_ids")

    async def get(self, key: str) -> Answer | None:
        """The cached answer, or None if missing or expired."""
        with translate_errors():
            doc = await self._entries.find_one({"_id": key, "expires_at": {"$gt": self._clock()}})
        return Answer.model_validate(doc["answer"]) if doc else None

    async def set(self, key: str, answer: Answer, doc_ids: list[str]) -> None:
        """Store an answer until now + ttl."""
        entry = {
            "answer": answer.model_dump(),
            "doc_ids": doc_ids,
            "expires_at": self._clock() + self._ttl,
        }
        with translate_errors():
            await self._entries.replace_one({"_id": key}, entry, upsert=True)

    async def purge_document(self, doc_id: str) -> int:
        """Remove every answer that depends on ``doc_id``."""
        with translate_errors():
            result = await self._entries.delete_many({"doc_ids": doc_id})
        return result.deleted_count
