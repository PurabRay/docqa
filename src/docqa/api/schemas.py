"""HTTP request and response bodies. Kept separate from the domain models."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ErrorBody(BaseModel):
    """Body of every error response."""

    code: str
    message: str


class HealthBody(BaseModel):
    """Body of GET /health."""

    status: Literal["ok", "unhealthy"]
    mongodb: Literal["ok", "unreachable"]
    search_indexes: dict[str, str]
    llm_providers: dict[str, str]
    version: str
    config_hash: str


class QueryBody(BaseModel):
    """Body of POST /query."""

    session_id: str = Field(min_length=1)
    question: str
    doc_ids: list[str] = Field(min_length=1)


class FeedbackBody(BaseModel):
    """Body of POST /feedback."""

    trace_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    rating: Literal[-1, 1]
    comment: str | None = Field(default=None, max_length=1000)


class UploadAccepted(BaseModel):
    """Body of 202 from POST /documents."""

    doc_id: str
    status: str


class DocumentBody(BaseModel):
    """One document as the API shows it."""

    doc_id: str
    filename: str
    page_count: int
    chunk_count: int
    status: str
    error: str | None
    created_at: datetime
