"""GET /health: database and search-index status, version and config hash."""

from __future__ import annotations

from importlib.metadata import version
from typing import Annotated

from fastapi import APIRouter, Depends, Response, status

from docqa.api.deps import get_container
from docqa.api.schemas import HealthBody
from docqa.bootstrap import Container

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthBody)
async def health(
    response: Response, container: Annotated[Container, Depends(get_container)]
) -> HealthBody:
    """Return 200 when MongoDB answers and every search index is READY, else 503."""
    db = await container.health.check()
    if not db.healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthBody(
        status="ok" if db.healthy else "unhealthy",
        mongodb="ok" if db.reachable else "unreachable",
        search_indexes=db.search_indexes,
        llm_providers={},  # filled in once the LLM router lands (M2)
        version=version("docqa"),
        config_hash=container.settings.config_hash(),
    )
