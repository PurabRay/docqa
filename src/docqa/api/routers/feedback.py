"""POST /feedback: thumbs up/down on an answer (FR-11)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Response, status

from docqa.api.deps import get_container
from docqa.api.schemas import FeedbackBody
from docqa.bootstrap import Container

router = APIRouter(tags=["feedback"])


@router.post("/feedback", status_code=status.HTTP_204_NO_CONTENT)
async def feedback(
    body: FeedbackBody, container: Annotated[Container, Depends(get_container)]
) -> Response:
    """Store the rating and attach it to the trace."""
    await container.feedback.add(body.trace_id, body.session_id, body.rating, body.comment)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
