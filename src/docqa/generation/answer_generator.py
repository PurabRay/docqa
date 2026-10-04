"""Stream the answer text to the caller, then parse the full reply into LLMAnswer.

Parsing tries, in order: structured JSON, a fenced ```json block, and finally plain
text with no citations (and a warning). Citations are never invented.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from typing import Literal

from pydantic import BaseModel, ValidationError

from docqa.domain.models import Token
from docqa.generation.json_stream import FieldStreamer
from docqa.generation.schemas import LLMAnswer
from docqa.ports.llm import LLMClient, Message

FENCED_JSON = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)
NO_CITATIONS_WARNING = "The model did not return valid citations; the answer is unverified."
MAX_ANSWER_CHARS = 1200  # same cap as LLMAnswer.answer

ParseStyle = Literal["structured", "fenced", "plain"]


class Final(BaseModel):
    """The parsed reply, sent once after the last Token."""

    parsed: LLMAnswer
    style: ParseStyle
    warning: str | None = None
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0


def parse_answer(raw: str, shown_text: str) -> tuple[LLMAnswer, ParseStyle, str | None]:
    """Parse the model's full reply; ``shown_text`` is what was streamed to the user."""
    if structured := _try_parse(raw.strip()):
        return structured, "structured", None
    match = FENCED_JSON.search(raw)
    if match and (fenced := _try_parse(match.group(1))):
        return fenced, "fenced", None
    text = (shown_text or raw).strip()[:MAX_ANSWER_CHARS]
    return (
        LLMAnswer(answer=text, citations=[], sufficient_context=True),
        "plain",
        NO_CITATIONS_WARNING,
    )


def _try_parse(text: str) -> LLMAnswer | None:
    try:
        return LLMAnswer.model_validate_json(text)
    except ValidationError:
        return None


class AnswerGenerator:
    """Runs the answer prompt through an LLM whose streams return LLMAnswer JSON."""

    def __init__(self, llm: LLMClient, max_tokens: int) -> None:
        self._llm = llm
        self._max_tokens = max_tokens

    async def generate(self, messages: list[Message]) -> AsyncIterator[Token | Final]:
        """Yield Token events, then exactly one Final."""
        streamer, shown = FieldStreamer("answer"), ""
        provider, input_tokens, output_tokens = "", 0, 0
        async for delta in self._llm.stream(messages, max_tokens=self._max_tokens):
            provider = delta.provider or provider
            if delta.done:
                input_tokens, output_tokens = delta.input_tokens, delta.output_tokens
                continue
            if text := streamer.feed(delta.text):
                shown += text
                yield Token(text=text)
        answer, style, warning = parse_answer(streamer.raw, shown)
        yield Final(
            parsed=answer,
            style=style,
            warning=warning,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
