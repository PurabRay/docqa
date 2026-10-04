"""Domain models (docs/DESIGN.md, "Domain models"). No MongoDB, HTTP or vendor code here.

Query events are added when the query service lands (M3, prompt 7).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Fixed namespace so chunk ids are the same on every machine and every re-ingest.
CHUNK_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "https://github.com/PurabRay/docqa/chunks")


def chunk_id(doc_id: str, index: int) -> str:
    """Stable id of the ``index``-th chunk of a document."""
    return str(uuid.uuid5(CHUNK_NAMESPACE, f"{doc_id}:{index}"))


def utc_now() -> datetime:
    """Current time, timezone-aware UTC."""
    return datetime.now(UTC)


def new_id() -> str:
    """Random id for new records."""
    return str(uuid.uuid4())


class IngestionStatus(StrEnum):
    """Lifecycle of an uploaded document."""

    QUEUED = "queued"
    EXTRACTING = "extracting"
    INDEXING = "indexing"
    SYNCING = "syncing"
    READY = "ready"
    FAILED = "failed"
    DELETED = "deleted"


class Page(BaseModel):
    """One extracted PDF page as Markdown."""

    doc_id: str
    number: int = Field(ge=1)  # 1-based, as printed in citations
    text: str
    headings: list[str] = []
    flagged_injection: bool = False


class Chunk(BaseModel):
    """A searchable piece of a document. ``id`` is chunk_id(doc_id, index)."""

    id: str
    doc_id: str
    owner_id: str
    filename: str
    text: str
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    section: str | None = None
    content_hash: str
    flagged_injection: bool = False
    # Character offsets in ``text`` where pages page_start+1, page_start+2, ... begin.
    page_offsets: list[int] = []

    def page_at(self, offset: int) -> int:
        """The page that holds the character at ``offset``."""
        return self.page_start + sum(1 for start in self.page_offsets if start <= offset)


class EmbeddedChunk(BaseModel):
    """A chunk plus its vector, ready to write to the vector store."""

    chunk: Chunk
    embedding: list[float] = Field(min_length=1)
    embed_model: str


class RetrievedChunk(BaseModel):
    """A chunk returned by search, with the scores that ranked it."""

    chunk: Chunk
    fused_score: float = 0.0
    vector_rank: int | None = None
    text_rank: int | None = None
    rerank_score: float | None = None


class Citation(BaseModel):
    """A verified pointer from an answer to a page of a document."""

    chunk_id: str
    doc_name: str
    page: int = Field(ge=1)
    quote: str  # verbatim text the claim rests on


class Answer(BaseModel):
    """A validated answer and its surviving citations."""

    text: str
    citations: list[Citation]
    dropped_citations: int = 0
    verified: bool  # False when no citation survived validation
    abstained: bool = False
    provider: str
    prompt_version: str


class DocumentRecord(BaseModel):
    """Metadata about one uploaded PDF."""

    id: str = Field(default_factory=new_id)
    owner_id: str = Field(min_length=1)
    filename: str
    sha256: str = Field(min_length=64, max_length=64)
    page_count: int = Field(ge=0)
    chunk_count: int = Field(default=0, ge=0)
    status: IngestionStatus = IngestionStatus.QUEUED
    error: str | None = None
    embed_model: str
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class IngestJob(BaseModel):
    """One uploaded document waiting to be ingested."""

    doc_id: str
    owner_id: str
    filename: str
    path: str


class Turn(BaseModel):
    """One chat message. Content is PII-scrubbed before it gets here."""

    id: str = Field(default_factory=new_id)
    session_id: str = Field(min_length=1)
    role: Literal["user", "assistant"]
    content_scrubbed: str
    trace_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)


class Feedback(BaseModel):
    """Thumbs up (1) or down (-1) on one answer."""

    id: str = Field(default_factory=new_id)
    trace_id: str = Field(min_length=1)
    session_id: str
    rating: Literal[-1, 1]
    comment: str | None = None
    created_at: datetime = Field(default_factory=utc_now)


class AccessFilter(BaseModel):
    """Who is asking and which documents they may search. Every search needs one."""

    model_config = ConfigDict(frozen=True)

    owner_id: str = Field(min_length=1)
    doc_ids: list[str] = Field(min_length=1)

    @field_validator("owner_id")
    @classmethod
    def _owner_not_blank(cls, owner_id: str) -> str:
        if not owner_id.strip():
            raise ValueError("owner_id must not be blank")
        return owner_id

    @field_validator("doc_ids")
    @classmethod
    def _doc_ids_not_blank(cls, doc_ids: list[str]) -> list[str]:
        if any(not doc_id.strip() for doc_id in doc_ids):
            raise ValueError("doc_ids must not contain blank ids")
        return doc_ids


class HybridQuery(BaseModel):
    """Query text plus its embedding, for hybrid search."""

    text: str = Field(min_length=1)
    vector: list[float] = Field(min_length=1)
