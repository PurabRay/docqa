"""CircuitBreaker on a fake clock, and FallbackLLMRouter with fake providers."""

from datetime import UTC, datetime

import pytest

from docqa.adapters.llm.circuit_breaker import CircuitBreaker, next_utc_midnight
from docqa.adapters.llm.router import FallbackLLMRouter
from docqa.adapters.llm.stub import StubLLM
from docqa.domain.errors import (
    AllProvidersUnavailableError,
    ProviderUnavailableError,
    RateLimitedError,
)
from docqa.ports.llm import LLMDelta, Message

MESSAGES = [Message(role="user", content="hi")]
NOON = datetime(2026, 10, 4, 12, 0, tzinfo=UTC).timestamp()


class Clock:
    def __init__(self, now=NOON):
        self.now = now

    def __call__(self):
        return self.now


# ---------------------------------------------------------------- breaker


def test_breaker_opens_after_n_failures_then_half_opens_then_closes():
    clock, changes = Clock(), []
    breaker = CircuitBreaker(
        "p",
        failures_to_open=3,
        open_seconds=60,
        clock=clock,
        listeners=[lambda name, old, new: changes.append((old, new))],
    )
    for _ in range(2):
        breaker.record_failure(ProviderUnavailableError())
    assert breaker.state == "closed"
    breaker.record_failure(ProviderUnavailableError())
    assert breaker.is_open()
    clock.now += 61
    assert breaker.state == "half_open" and not breaker.is_open()
    breaker.record_success()
    assert breaker.state == "closed"
    assert changes == [("closed", "open"), ("half_open", "closed")]


def test_failure_while_half_open_reopens_at_once():
    clock = Clock()
    breaker = CircuitBreaker("p", failures_to_open=1, open_seconds=60, clock=clock)
    breaker.record_failure(ProviderUnavailableError())
    clock.now += 61
    breaker.record_failure(ProviderUnavailableError())
    assert breaker.is_open()


def test_daily_quota_opens_until_the_next_utc_day():
    clock = Clock()
    breaker = CircuitBreaker("p", failures_to_open=3, open_seconds=60, clock=clock)
    breaker.record_failure(RateLimitedError(daily_quota=True))
    clock.now += 11.9 * 3600  # 23:54 UTC
    assert breaker.is_open()
    clock.now = next_utc_midnight(NOON) + 1
    assert not breaker.is_open()


# ---------------------------------------------------------------- router


class Failing:
    """A provider that raises the given errors in turn, then succeeds."""

    def __init__(self, name, errors, reply="ok"):
        self.name, self.errors, self.calls = name, list(errors), 0
        self.ok = StubLLM(reply, name=name)

    async def complete(self, messages, **kw):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return await self.ok.complete(messages, **kw)

    async def stream(self, messages, **kw):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        async for delta in self.ok.stream(messages, **kw):
            yield delta


class BreaksMidStream:
    name = "midstream"

    async def stream(self, messages, **kw):
        yield LLMDelta(text="first token", provider=self.name)
        raise ProviderUnavailableError("connection dropped")


def router(*clients, retries=1):
    breakers = {c.name: CircuitBreaker(c.name, 3, 60, clock=Clock()) for c in clients}
    return FallbackLLMRouter(list(clients), breakers, retries=retries, retry_wait_s=0), breakers


@pytest.mark.parametrize(
    "error", [RateLimitedError(daily_quota=True), ProviderUnavailableError("timeout")]
)
async def test_falls_back_to_the_next_provider(error):
    first, second = Failing("gemini", [error]), Failing("groq", [], reply="from groq")
    result = await router(first, second)[0].complete(MESSAGES)
    assert (result.provider, result.text, first.calls) == ("groq", "from groq", 1)


async def test_per_minute_429_is_retried_once_on_the_same_provider():
    first, second = Failing("gemini", [RateLimitedError()]), Failing("groq", [])
    result = await router(first, second)[0].complete(MESSAGES)
    assert result.provider == "gemini" and first.calls == 2 and second.calls == 0


async def test_skips_a_provider_whose_breaker_is_open():
    first, second = Failing("gemini", []), Failing("groq", [])
    route, breakers = router(first, second)
    for _ in range(3):
        breakers["gemini"].record_failure(ProviderUnavailableError())
    assert (await route.complete(MESSAGES)).provider == "groq" and first.calls == 0


def all_down():
    return router(
        Failing("a", [ProviderUnavailableError()]), Failing("b", [ProviderUnavailableError()])
    )[0]


async def test_raises_when_every_provider_fails():
    with pytest.raises(AllProvidersUnavailableError):
        await all_down().complete(MESSAGES)
    with pytest.raises(AllProvidersUnavailableError):
        [d async for d in all_down().stream(MESSAGES)]


async def test_stream_falls_back_before_the_first_token():
    first, second = (
        Failing("gemini", [ProviderUnavailableError()]),
        Failing("groq", [], reply="streamed"),
    )
    deltas = [d async for d in router(first, second)[0].stream(MESSAGES)]
    assert "".join(d.text for d in deltas) == "streamed" and {d.provider for d in deltas} == {
        "groq"
    }


async def test_stream_never_falls_back_mid_answer():
    backup = Failing("groq", [])
    seen = []
    with pytest.raises(ProviderUnavailableError):
        async for delta in router(BreaksMidStream(), backup)[0].stream(MESSAGES):
            seen.append(delta.text)
    assert seen == ["first token"] and backup.calls == 0
