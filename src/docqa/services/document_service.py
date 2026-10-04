"""Document use cases: upload, list, get and delete (FR-1, FR-4, FR-12)."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Callable
from pathlib import Path

from docqa.config_schema import LimitsConfig
from docqa.domain.errors import DocQAError, DocumentNotFoundError, FileTooLargeError
from docqa.domain.models import DocumentRecord, IngestionStatus, IngestJob, new_id
from docqa.guardrails.storage_guard import StorageGuard
from docqa.ingestion.validator import BYTES_PER_MB, validate_pdf
from docqa.ports.repository import DocumentRepository
from docqa.ports.vector_store import VectorStore


class DocumentService:
    """Accepts uploads, queues them for ingestion, and deletes documents."""

    def __init__(
        self,
        repo: DocumentRepository,
        store: VectorStore,
        guard: StorageGuard,
        submit: Callable[[IngestJob], None],
        limits: LimitsConfig,
        upload_dir: Path,
        embed_model: str,
    ) -> None:
        self._repo = repo
        self._store = store
        self._guard = guard
        self._submit = submit
        self._limits = limits
        self._upload_dir = upload_dir
        self._embed_model = embed_model

    async def upload(self, owner_id: str, filename: str, data: bytes) -> DocumentRecord:
        """Validate, check capacity, record (409 on a duplicate) and queue a PDF."""
        if len(data) > self._limits.max_pdf_mb * BYTES_PER_MB:
            raise FileTooLargeError(f"The file is larger than {self._limits.max_pdf_mb} MB.")
        doc_id = new_id()
        path = self._upload_dir / f"{doc_id}.pdf"
        await asyncio.to_thread(self._save, path, data)
        try:
            info = await asyncio.to_thread(validate_pdf, path, self._limits)
            await self._guard.ensure_capacity()
            record = DocumentRecord(
                id=doc_id,
                owner_id=owner_id,
                filename=filename,
                sha256=hashlib.sha256(data).hexdigest(),
                page_count=info.page_count,
                embed_model=self._embed_model,
            )
            await self._repo.create(record)
        except DocQAError:
            path.unlink(missing_ok=True)
            raise
        self._submit(IngestJob(doc_id=doc_id, owner_id=owner_id, filename=filename, path=str(path)))
        return record

    async def list(self, owner_id: str) -> list[DocumentRecord]:
        """The owner's documents, newest first, without deleted ones."""
        docs = await self._repo.list_for_owner(owner_id)
        return [doc for doc in docs if doc.status is not IngestionStatus.DELETED]

    async def get(self, owner_id: str, doc_id: str) -> DocumentRecord:
        """One document, or DocumentNotFoundError."""
        doc = await self._repo.get(doc_id, owner_id)
        if doc is None:
            raise DocumentNotFoundError()
        return doc

    async def delete(self, owner_id: str, doc_id: str) -> None:
        """Remove the document's chunks and file, and mark it deleted."""
        doc = await self.get(owner_id, doc_id)
        if doc.status is IngestionStatus.DELETED:
            raise DocumentNotFoundError()
        await self._store.delete_document(doc_id, owner_id)
        await self._repo.set_status(doc_id, IngestionStatus.DELETED)
        (self._upload_dir / f"{doc_id}.pdf").unlink(missing_ok=True)

    def _save(self, path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
