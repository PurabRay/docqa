"""HTTP request and response bodies. Kept separate from the domain models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


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
