"""StubLLM: canned replies after a configurable delay, for tests and load tests."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from pydantic import BaseModel

from docqa.ports.llm import LLMDelta, LLMResult, Message


class StubLLM:
    """Implements the LLMClient port without a network.

    Args:
        reply: The text (often JSON) to return for every call.
        delay_s: Total time a reply takes, spread over the streamed pieces.
        piece_size: Characters per streamed delta.
        name: Provider name to report.
    """

    def __init__(
        self, reply: str, delay_s: float = 0.0, piece_size: int = 8, name: str = "stub"
    ) -> None:
        self.name = name
        self.reply = reply
        self._delay_s = delay_s
        self._piece_size = piece_size
        self.calls: list[list[Message]] = []

    async def complete(
        self,
        messages: list[Message],
        *,
        json_schema: type[BaseModel] | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> LLMResult:
        """Return the canned reply."""
        self.calls.append(messages)
        await asyncio.sleep(self._delay_s)
        return LLMResult(text=self.reply, provider=self.name, input_tokens=1, output_tokens=1)

    async def stream(
        self,
        messages: list[Message],
        *,
        json_schema: type[BaseModel] | None = None,
        max_tokens: int = 512,
    ) -> AsyncIterator[LLMDelta]:
        """Stream the canned reply in pieces."""
        self.calls.append(messages)
        pieces = [
            self.reply[i : i + self._piece_size]
            for i in range(0, len(self.reply), self._piece_size)
        ]
        for piece in pieces:
            await asyncio.sleep(self._delay_s / max(len(pieces), 1))
            yield LLMDelta(text=piece, provider=self.name)
        yield LLMDelta(done=True, provider=self.name, input_tokens=1, output_tokens=len(pieces))
