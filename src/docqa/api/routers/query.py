"""POST /query: the answer as Server-Sent Events (FR-8)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from sse_starlette.sse import EventSourceResponse

from docqa.api.deps import get_container, owner_id
from docqa.api.schemas import QueryBody
from docqa.api.sse import sse_events
from docqa.bootstrap import Container
from docqa.domain.models import QueryRequest

router = APIRouter(tags=["query"])


@router.post("/query")
async def query(
    body: QueryBody, container: Annotated[Container, Depends(get_container)]
) -> EventSourceResponse:
    """Validate first (so 400/429 are real HTTP errors), then stream the events.

    sse-starlette sends a ping every ``sse_ping_s`` and stops the generator when the
    client disconnects.
    """
    request = QueryRequest(
        owner_id=owner_id(body.session_id),
        session_id=body.session_id,
        question=body.question,
        doc_ids=body.doc_ids,
    )
    await container.query.check(request)
    return EventSourceResponse(
        sse_events(container.query.answer(request)),
        ping=int(container.settings.generation.sse_ping_s),
    )
