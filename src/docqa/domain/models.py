"""Domain models (docs/DESIGN.md, "Domain models"). No MongoDB, HTTP or vendor code here.

The QueryEvent union at the bottom is what QueryService streams; each ``type`` is the
SSE event name from docs/DESIGN.md.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal

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
    """A final answer with its verified citations (docs/DESIGN.md)."""

    text: str
    citations: list[Citation]
    abstained: bool = False
    provider: str  # which model actually answered
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


class QueryRequest(BaseModel):
    """A question about some of the owner's documents."""

    owner_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    doc_ids: list[str] = Field(min_length=1)


# ---------------------------------------------------------------- query events


class Meta(BaseModel):
    """First event: the trace id and, for follow-ups, the rewritten question."""

    type: Literal["meta"] = "meta"
    trace_id: str
    rewritten_question: str | None = None


class Token(BaseModel):
    """One streamed piece of the answer."""

    type: Literal["token"] = "token"
    text: str


class Citations(BaseModel):
    """Citations that survived validation, and how many were dropped."""

    type: Literal["citations"] = "citations"
    citations: list[Citation]
    dropped: int = 0


class Abstained(BaseModel):
    """Sent instead of tokens when the documents do not answer the question."""

    type: Literal["abstain"] = "abstain"
    reason: Literal["low_relevance", "insufficient_context"]


class Degraded(BaseModel):
    """Passages-only fallback when every LLM provider is unavailable."""

    type: Literal["degraded"] = "degraded"
    reason: Literal["all_providers_unavailable"] = "all_providers_unavailable"
    passages: list[RetrievedChunk]


class Error(BaseModel):
    """An unrecoverable error, reported in-stream."""

    type: Literal["error"] = "error"
    code: str
    message: str


class Completed(BaseModel):
    """Last event ("done"): the answer, who gave it, how long it took, token counts."""

    type: Literal["done"] = "done"
    answer: Answer | None = None
    provider: str | None = None
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached: bool = False


QueryEvent = Annotated[
    Meta | Token | Citations | Abstained | Degraded | Error | Completed,
    Field(discriminator="type"),
]
