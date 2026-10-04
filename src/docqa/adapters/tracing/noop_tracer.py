"""NoopTracer: same port, records nothing (CI, local mode, tests)."""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager

from docqa.ports.tracer import MetaValue


class NoopSpan:
    """A span that keeps its attributes in memory (handy in tests)."""

    def __init__(self, trace_id: str = "") -> None:
        self.trace_id = trace_id
        self.attributes: dict[str, MetaValue] = {}

    def set(self, **attributes: MetaValue) -> None:
        """Remember attributes."""
        self.attributes.update(attributes)


class NoopTracer:
    """Implements the Tracer port without a backend; trace ids are still unique."""

    @contextmanager
    def trace(self, name: str, **meta: MetaValue) -> Iterator[NoopSpan]:
        """A trace with a fresh 32-hex-character id."""
        yield NoopSpan(trace_id=uuid.uuid4().hex)

    @contextmanager
    def span(self, name: str, **meta: MetaValue) -> Iterator[NoopSpan]:
        """A span that goes nowhere."""
        yield NoopSpan()

    def score(self, trace_id: str, name: str, value: float, comment: str | None = None) -> None:
        """Scores go nowhere."""

    def flush(self) -> None:
        """Nothing to flush."""
