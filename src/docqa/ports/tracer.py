"""Tracer port: one trace per use case, one span per step (docs/DESIGN.md)."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Protocol

# Metadata stays primitive (decision B9: no Any in ports).
MetaValue = str | int | float | bool | None


class SpanHandle(Protocol):
    """A running span; attributes can be added before it closes."""

    def set(self, **attributes: MetaValue) -> None:
        """Attach attributes such as token counts or latency."""
        ...


class TraceHandle(SpanHandle, Protocol):
    """A running trace; its id goes to the client and comes back with feedback."""

    trace_id: str


class Tracer(Protocol):
    """Starts traces and spans, and records feedback scores."""

    def trace(self, name: str, **meta: MetaValue) -> AbstractContextManager[TraceHandle]:
        """Open a trace for one use case (query, ingest, ...)."""
        ...

    def span(self, name: str, **meta: MetaValue) -> AbstractContextManager[SpanHandle]:
        """Open a span for one step inside the current trace."""
        ...

    def score(self, trace_id: str, name: str, value: float, comment: str | None = None) -> None:
        """Attach a score (e.g. user feedback) to an existing trace."""
        ...
