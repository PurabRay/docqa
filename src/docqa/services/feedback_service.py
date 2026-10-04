"""Feedback use case (FR-11): store thumbs up/down and send it to the trace as a score."""

from __future__ import annotations

import asyncio
from typing import Literal

from docqa.domain.models import Feedback
from docqa.guardrails.pii import scrub
from docqa.ports.repository import DocumentRepository
from docqa.ports.tracer import Tracer

SCORE_NAME = "user_feedback"


class FeedbackService:
    """Records feedback in MongoDB and Langfuse."""

    def __init__(self, repo: DocumentRepository, tracer: Tracer) -> None:
        self._repo = repo
        self._tracer = tracer

    async def add(
        self, trace_id: str, session_id: str, rating: Literal[-1, 1], comment: str | None
    ) -> None:
        """Store the (PII-scrubbed) feedback and attach it to the trace."""
        clean = await asyncio.to_thread(scrub, comment) if comment else None
        await self._repo.add_feedback(
            Feedback(trace_id=trace_id, session_id=session_id, rating=rating, comment=clean)
        )
        self._tracer.score(trace_id, SCORE_NAME, float(rating), clean)
