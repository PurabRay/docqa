"""Real calls to hosted providers. Skipped unless the keys are set (in env or .env).

Run: uv run pytest -m live -s
These spend a tiny amount of free-tier quota (a few short requests).
"""

import pytest

from docqa.adapters.llm.circuit_breaker import CircuitBreaker
from docqa.adapters.llm.openai_compatible import OpenAICompatibleLLM
from docqa.adapters.llm.router import FallbackLLMRouter
from docqa.ports.llm import Message
from docqa.settings import load_settings

SETTINGS = load_settings()
PROVIDERS = SETTINGS.llm.providers
PROMPT = [Message(role="user", content="Reply with the single word: ok")]


def key(env_name: str) -> str | None:
    return SETTINGS.optional_secret(env_name)


def needs(env_name: str):
    return pytest.mark.skipif(key(env_name) is None, reason=f"{env_name} not set")


def client(name: str, api_key: str | None = None) -> OpenAICompatibleLLM:
    provider = PROVIDERS[name]
    return OpenAICompatibleLLM(name, provider, api_key or key(provider.api_key_env))


@needs("GEMINI_API_KEY")
async def test_gemini_answers():
    result = await client("gemini_flash").complete(PROMPT, max_tokens=20)
    print(f"\ngemini: {result.text!r} ({result.input_tokens} in / {result.output_tokens} out)")
    assert result.text.strip() and result.provider == "gemini_flash"


@needs("GROQ_API_KEY")
async def test_groq_answers():
    result = await client("groq_oss_20b").complete(PROMPT, max_tokens=200)
    print(f"\ngroq: {result.text!r}")
    assert result.provider == "groq_oss_20b"


@needs("GROQ_API_KEY")
async def test_router_answers_from_groq_when_the_gemini_key_is_broken():
    clients = [client("gemini_flash", api_key="deliberately-broken-key"), client("groq_oss_20b")]
    breakers = {c.name: CircuitBreaker(c.name, 3, 60) for c in clients}
    router = FallbackLLMRouter(clients, breakers, retries=0, retry_wait_s=0)
    result = await router.complete(PROMPT, max_tokens=200)
    print(f"\nrouter answered with {result.provider}: {result.text!r}")
    assert result.provider == "groq_oss_20b"
