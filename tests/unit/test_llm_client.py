"""OpenAICompatibleLLM against a mocked HTTP transport (no network)."""

import json

import httpx2
import pytest

from docqa.adapters.llm.openai_compatible import OpenAICompatibleLLM
from docqa.domain.errors import ProviderUnavailableError, RateLimitedError
from docqa.generation.schemas import LLMAnswer
from docqa.ports.llm import Message
from docqa.settings import load_settings

PROVIDERS = load_settings(env_file=None).llm.providers
MESSAGES = [Message(role="user", content="hi")]
COMPLETION = {
    "id": "x",
    "object": "chat.completion",
    "created": 0,
    "model": "m",
    "choices": [
        {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "hello"}}
    ],
    "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
}


def client(name, handler):
    transport = httpx2.MockTransport(handler)
    return OpenAICompatibleLLM(
        name, PROVIDERS[name], "key", http_client=httpx2.AsyncClient(transport=transport)
    )


def recording(seen, response=None):
    def handler(request):
        seen.append(json.loads(request.content))
        return response or httpx2.Response(200, json=COMPLETION)

    return handler


async def test_gemini_asks_for_a_json_schema():
    seen = []
    result = await client("gemini_flash", recording(seen)).complete(MESSAGES, json_schema=LLMAnswer)
    body = seen[0]
    assert body["model"] == "gemini-flash-latest" and body["temperature"] == 0.0
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["name"] == "LLMAnswer"
    assert (result.text, result.provider, result.input_tokens, result.output_tokens) == (
        "hello",
        "gemini_flash",
        7,
        2,
    )


async def test_groq_without_schema_support_uses_json_mode():
    seen = []
    await client("groq_oss_20b", recording(seen)).complete(MESSAGES, json_schema=LLMAnswer)
    assert seen[0]["response_format"] == {"type": "json_object"}


async def test_plain_request_has_no_response_format():
    seen = []
    await client("groq_oss_20b", recording(seen)).complete(MESSAGES)
    assert "response_format" not in seen[0]


async def test_ollama_turns_thinking_off():
    seen = []
    await client("ollama_qwen", recording(seen)).complete(MESSAGES)
    assert seen[0]["think"] is False and seen[0]["temperature"] == 0.0


@pytest.mark.parametrize(
    "message, daily",
    [
        ("Rate limit reached for model on tokens per day (TPD): Limit 200000", True),
        ("Quota exceeded for metric GenerateRequestsPerDayPerProjectPerModel", True),
        ("Rate limit reached on tokens per minute (TPM): Limit 8000", False),
    ],
)
async def test_429_says_whether_the_daily_quota_is_spent(message, daily):
    response = httpx2.Response(429, json={"error": {"message": message, "type": "rate_limit"}})
    with pytest.raises(RateLimitedError) as caught:
        await client("groq_oss_20b", recording([], response)).complete(MESSAGES)
    assert caught.value.daily_quota is daily


@pytest.mark.parametrize("status", [500, 503, 401])
async def test_server_errors_and_bad_keys_make_the_provider_unavailable(status):
    response = httpx2.Response(status, json={"error": {"message": "nope"}})
    with pytest.raises(ProviderUnavailableError, match=str(status)):
        await client("gemini_flash", recording([], response)).complete(MESSAGES)


async def test_timeout_makes_the_provider_unavailable():
    def handler(request):
        raise httpx2.ReadTimeout("slow", request=request)

    with pytest.raises(ProviderUnavailableError, match="timed out"):
        await client("gemini_flash", handler).complete(MESSAGES)


def sse(*events) -> bytes:
    return b"".join(f"data: {json.dumps(e)}\n\n".encode() for e in events) + b"data: [DONE]\n\n"


def chunk(content=None, usage=None):
    choices = (
        []
        if content is None
        else [{"index": 0, "delta": {"content": content}, "finish_reason": None}]
    )
    return {
        "id": "x",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "m",
        "choices": choices,
        "usage": usage,
    }


async def test_stream_yields_text_then_a_final_usage_record():
    body = sse(
        chunk("Hel"),
        chunk("lo"),
        chunk(usage={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}),
    )
    seen = []
    response = httpx2.Response(200, content=body, headers={"content-type": "text/event-stream"})
    deltas = [d async for d in client("groq_oss_20b", recording(seen, response)).stream(MESSAGES)]
    assert [d.text for d in deltas] == ["Hel", "lo", ""]
    assert deltas[-1].done and (deltas[-1].input_tokens, deltas[-1].output_tokens) == (5, 2)
    assert seen[0]["stream"] is True and seen[0]["stream_options"] == {"include_usage": True}


async def test_stream_uses_the_schema_given_at_construction():
    seen = []
    response = httpx2.Response(
        200, content=sse(chunk("{}")), headers={"content-type": "text/event-stream"}
    )
    llm = OpenAICompatibleLLM(
        "gemini_flash",
        PROVIDERS["gemini_flash"],
        "key",
        response_schema=LLMAnswer,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(recording(seen, response))),
    )
    [d async for d in llm.stream(MESSAGES)]
    assert seen[0]["response_format"]["json_schema"]["name"] == "LLMAnswer"
