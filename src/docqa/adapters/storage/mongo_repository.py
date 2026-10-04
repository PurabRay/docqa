"""MongoDocumentRepository: documents, turns and feedback in MongoDB."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from pymongo import DESCENDING
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.errors import DuplicateKeyError

from docqa.adapters.mongo import codecs
from docqa.adapters.mongo.client import Document, translate_errors
from docqa.adapters.mongo.names import DOCUMENTS, FEEDBACK, TURNS
from docqa.domain.errors import DocumentNotFoundError, DuplicateDocumentError
from docqa.domain.models import DocumentRecord, Feedback, IngestionStatus, Turn, utc_now


class MongoDocumentRepository:
    """Implements the DocumentRepository port.

    Args:
        db: Handle to the docqa database.
        clock: Returns "now"; injected so tests can control timestamps.
    """

    def __init__(
        self, db: AsyncDatabase[Document], clock: Callable[[], datetime] = utc_now
    ) -> None:
        self._documents = db[DOCUMENTS]
        self._turns = db[TURNS]
        self._feedback = db[FEEDBACK]
        self._clock = clock

    async def create(self, doc: DocumentRecord) -> None:
        """Insert a document record.

        Raises:
            DuplicateDocumentError: The owner already has a file with this SHA-256.
        """
        try:
            with translate_errors():
                await self._documents.insert_one(codecs.document_to_bson(doc))
        except DuplicateKeyError as err:
            raise DuplicateDocumentError() from err

    async def set_status(
        self,
        doc_id: str,
        status: IngestionStatus,
        error: str | None = None,
        chunk_count: int | None = None,
    ) -> None:
        """Update a document's status, error and (optionally) chunk count.

        Raises:
            DocumentNotFoundError: No document has this id.
        """
        update: dict[str, object] = {
            "status": status.value,
            "error": error,
            "updated_at": self._clock(),
        }
        if chunk_count is not None:
            update["chunk_count"] = chunk_count
        with translate_errors():
            result = await self._documents.update_one({"_id": doc_id}, {"$set": update})
        if result.matched_count == 0:
            raise DocumentNotFoundError()

    async def get(self, doc_id: str, owner_id: str) -> DocumentRecord | None:
        """Return the owner's document, or None (another owner's id also gives None)."""
        with translate_errors():
            doc = await self._documents.find_one({"_id": doc_id, "owner_id": owner_id})
        return codecs.document_from_bson(doc) if doc else None

    async def list_for_owner(self, owner_id: str) -> list[DocumentRecord]:
        """Return the owner's documents, newest first."""
        with translate_errors():
            cursor = self._documents.find({"owner_id": owner_id}).sort("created_at", DESCENDING)
            docs = await cursor.to_list()
        return [codecs.document_from_bson(doc) for doc in docs]

    async def append_turn(self, turn: Turn) -> None:
        """Store one chat turn (content must already be PII-scrubbed)."""
        with translate_errors():
            await self._turns.insert_one(codecs.turn_to_bson(turn))

    async def last_turns(self, session_id: str, n: int = 3) -> list[Turn]:
        """Return the last ``n`` turns of a session, oldest first."""
        with translate_errors():
            cursor = (
                self._turns.find({"session_id": session_id}).sort("created_at", DESCENDING).limit(n)
            )
            docs = await cursor.to_list()
        return [codecs.turn_from_bson(doc) for doc in reversed(docs)]

    async def add_feedback(self, fb: Feedback) -> None:
        """Store a thumbs up or down."""
        with translate_errors():
            await self._feedback.insert_one(codecs.feedback_to_bson(fb))
