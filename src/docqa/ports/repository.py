"""Repository port: documents, chat turns and feedback."""

from __future__ import annotations

from typing import Protocol

from docqa.domain.models import DocumentRecord, Feedback, IngestionStatus, Turn


class DocumentRepository(Protocol):
    """Persistent metadata. Raises DuplicateDocumentError and DocumentNotFoundError."""

    async def create(self, doc: DocumentRecord) -> None:
        """Insert a new document record."""
        ...

    async def set_status(
        self, doc_id: str, status: IngestionStatus, error: str | None = None
    ) -> None:
        """Move a document to ``status``, recording ``error`` if it failed."""
        ...

    async def get(self, doc_id: str, owner_id: str) -> DocumentRecord | None:
        """Return one of the owner's documents, or None."""
        ...

    async def list_for_owner(self, owner_id: str) -> list[DocumentRecord]:
        """Return the owner's documents, newest first."""
        ...

    async def append_turn(self, turn: Turn) -> None:
        """Store one (already PII-scrubbed) chat turn."""
        ...

    async def last_turns(self, session_id: str, n: int = 3) -> list[Turn]:
        """Return the last ``n`` turns of a session, oldest first."""
        ...

    async def add_feedback(self, fb: Feedback) -> None:
        """Store a thumbs up or down."""
        ...
