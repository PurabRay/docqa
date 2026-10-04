"""LangfuseTracer: traces, spans, LLM generations and feedback scores in Langfuse.

The SDK batches and sends in the background, so tracing never adds request latency.
If Langfuse cannot be reached, a warning is logged once and requests carry on with
a local trace id.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from typing import Any

from docqa.adapters.tracing.noop_tracer import NoopSpan
from docqa.ports.tracer import MetaValue

log = logging.getLogger(__name__)

# Span names that are LLM calls; Langfuse shows them as generations with tokens and cost.
GENERATION_SPANS = {"generate", "rewrite"}
USAGE_KEYS = {"input_tokens": "input", "output_tokens": "output"}


class _LangfuseSpan:
    """Port handle over a Langfuse observation."""

    def __init__(self, observation: Any, generation: bool) -> None:
        self._observation = observation
        self._generation = generation
        self._metadata: dict[str, MetaValue] = {}
        self.trace_id: str = observation.trace_id

    def set(self, **attributes: MetaValue) -> None:
        """Usage and cost go to the generation fields; everything else to metadata."""
        update: dict[str, Any] = {}
        if self._generation:
            usage = {USAGE_KEYS[k]: int(v or 0) for k, v in attributes.items() if k in USAGE_KEYS}
            if usage:
                update["usage_details"] = usage
            if "cost_usd" in attributes:
                update["cost_details"] = {"total": float(attributes["cost_usd"] or 0.0)}
            if "model" in attributes:
                update["model"] = str(attributes["model"])
        self._metadata.update(attributes)
        self._observation.update(metadata=dict(self._metadata), **update)


class LangfuseTracer:
    """Implements the Tracer port on the Langfuse SDK (v4)."""

    def __init__(self, client: Any) -> None:
        """``client`` is a langfuse.Langfuse (tests pass a fake with the same methods)."""
        self._client = client
        self._warned = False

    @contextmanager
    def trace(self, name: str, **meta: MetaValue) -> Iterator[_LangfuseSpan | NoopSpan]:
        """Root observation of one use case; ``user`` becomes the trace's user id."""
        with ExitStack() as stack:
            handle = self._open(stack, name, generation=False, meta=meta)
            if isinstance(handle, _LangfuseSpan):
                self._safe(
                    handle._observation.update_trace,
                    name=name,
                    user_id=meta.get("user"),
                    metadata=meta,
                )
            yield handle

    @contextmanager
    def span(self, name: str, **meta: MetaValue) -> Iterator[_LangfuseSpan | NoopSpan]:
        """Child observation of the current trace."""
        with ExitStack() as stack:
            yield self._open(stack, name, generation=name in GENERATION_SPANS, meta=meta)

    def score(self, trace_id: str, name: str, value: float, comment: str | None = None) -> None:
        """Attach a numeric score to a trace."""
        self._safe(
            self._client.create_score, trace_id=trace_id, name=name, value=value, comment=comment
        )

    def flush(self) -> None:
        """Send buffered events (called on shutdown)."""
        self._safe(self._client.flush)

    def _open(
        self, stack: ExitStack, name: str, generation: bool, meta: dict[str, MetaValue]
    ) -> _LangfuseSpan | NoopSpan:
        as_type = "generation" if generation else "span"
        try:
            manager = self._client.start_as_current_observation(
                name=name, as_type=as_type, metadata=meta
            )
            return _LangfuseSpan(stack.enter_context(manager), generation)
        except Exception as err:  # vendor boundary: tracing must never break a request
            self._warn(err)
            return NoopSpan(trace_id=uuid.uuid4().hex)

    def _safe(self, call: Any, **kwargs: Any) -> None:
        try:
            call(**kwargs)
        except Exception as err:  # vendor boundary, as above
            self._warn(err)

    def _warn(self, err: Exception) -> None:
        if not self._warned:
            log.warning("tracing.unavailable error=%s", type(err).__name__)
            self._warned = True
