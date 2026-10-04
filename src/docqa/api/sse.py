"""QueryEvent -> Server-Sent Event (docs/DESIGN.md, "Server-Sent Events from POST /query")."""

from __future__ import annotations

from collections.abc import AsyncIterator

from docqa.domain.models import Completed, QueryEvent


def to_sse(event: QueryEvent) -> dict[str, str]:
    """The SSE event name is the event's ``type``; data is its JSON without ``type``.

    ``done`` carries only {provider, latency_ms, input_tokens, output_tokens, cached}.
    """
    exclude = {"type", "answer"} if isinstance(event, Completed) else {"type"}
    return {"event": event.type, "data": event.model_dump_json(exclude=exclude)}


async def sse_events(events: AsyncIterator[QueryEvent]) -> AsyncIterator[dict[str, str]]:
    """Convert a QueryEvent stream for sse-starlette's EventSourceResponse."""
    async for event in events:
        yield to_sse(event)
