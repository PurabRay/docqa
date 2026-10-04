"""LLM port: chat completion and streaming over any OpenAI-compatible provider."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Literal, Protocol

from pydantic import BaseModel


class Message(BaseModel):
    """One chat message sent to an LLM."""

    role: Literal["system", "user", "assistant"]
    content: str


class LLMResult(BaseModel):
    """A complete (non-streamed) reply."""

    text: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0


class LLMDelta(BaseModel):
    """A streamed piece of a reply; the last one has done=True and the token counts."""

    text: str = ""
    done: bool = False
    provider: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


class LLMClient(Protocol):
    """A chat model. Adapters raise RateLimitedError or ProviderUnavailableError."""

    name: str

    async def complete(
        self,
        messages: list[Message],
        *,
        json_schema: type[BaseModel] | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> LLMResult:
        """Return the full reply, constrained to ``json_schema`` when given."""
        ...

    def stream(
        self,
        messages: list[Message],
        *,
        json_schema: type[BaseModel] | None = None,
        max_tokens: int = 512,
    ) -> AsyncIterator[LLMDelta]:
        """Yield the reply piece by piece, then one final delta with done=True."""
        ...
