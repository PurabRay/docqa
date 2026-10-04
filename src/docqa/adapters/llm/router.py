"""FallbackLLMRouter: try providers in config order; also implements LLMClient.

* A provider whose breaker is open is skipped.
* A per-minute 429 is retried (at most ``retries`` times) before moving on;
  a daily-quota 429 moves on at once.
* stream() falls back only BEFORE the first token. Once text has reached the user,
  switching models mid-answer would mix two answers, so a mid-stream error is raised.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from functools import partial
from typing import TypeVar

from pydantic import BaseModel
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_fixed

from docqa.adapters.llm.circuit_breaker import CircuitBreaker
from docqa.domain.errors import (
    AllProvidersUnavailableError,
    ProviderUnavailableError,
    RateLimitedError,
)
from docqa.ports.llm import LLMClient, LLMDelta, LLMResult, Message

T = TypeVar("T")
FALLBACK_ERRORS = (RateLimitedError, ProviderUnavailableError)


def _is_minute_429(err: BaseException) -> bool:
    return isinstance(err, RateLimitedError) and not err.daily_quota


class FallbackLLMRouter:
    """Routes each call to the first healthy provider."""

    name = "router"

    def __init__(
        self,
        clients: Sequence[LLMClient],
        breakers: dict[str, CircuitBreaker],
        retries: int,
        retry_wait_s: float,
    ) -> None:
        self._clients = clients
        self._breakers = breakers
        self._retries = retries
        self._retry_wait_s = retry_wait_s

    async def complete(
        self,
        messages: list[Message],
        *,
        json_schema: type[BaseModel] | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> LLMResult:
        """Complete with the first provider that answers."""
        for client in self._healthy():
            try:
                call = partial(
                    client.complete,
                    messages,
                    json_schema=json_schema,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                result = await self._with_retry(call)
            except FALLBACK_ERRORS as err:
                self._breakers[client.name].record_failure(err)
                continue
            self._breakers[client.name].record_success()
            return result
        raise AllProvidersUnavailableError()

    async def stream(
        self, messages: list[Message], *, max_tokens: int = 512
    ) -> AsyncIterator[LLMDelta]:
        """Stream from the first provider that produces a first delta."""
        for client in self._healthy():
            try:
                opener = partial(self._open, client, messages, max_tokens)
                stream, first = await self._with_retry(opener)
            except FALLBACK_ERRORS as err:
                self._breakers[client.name].record_failure(err)
                continue
            yield first
            async for delta in stream:  # errors from here on are not retried elsewhere
                yield delta
            self._breakers[client.name].record_success()
            return
        raise AllProvidersUnavailableError()

    def _healthy(self) -> list[LLMClient]:
        return [c for c in self._clients if not self._breakers[c.name].is_open()]

    async def _open(
        self, client: LLMClient, messages: list[Message], max_tokens: int
    ) -> tuple[AsyncIterator[LLMDelta], LLMDelta]:
        """Start a stream and wait for its first delta (where most failures show up)."""
        stream = client.stream(messages, max_tokens=max_tokens)
        return stream, await anext(stream)

    async def _with_retry(self, call: Callable[[], Awaitable[T]]) -> T:
        retrying = AsyncRetrying(
            retry=retry_if_exception(_is_minute_429),
            stop=stop_after_attempt(self._retries + 1),
            wait=wait_fixed(self._retry_wait_s),
            reraise=True,
        )
        return await retrying(call)
