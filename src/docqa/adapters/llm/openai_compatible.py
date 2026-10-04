"""OpenAICompatibleLLM: one client for Gemini, Groq, Mistral, OpenRouter and Ollama.

All of them expose POST {base_url}/chat/completions. Structured output uses the
provider's best option: JSON schema, else JSON mode, else plain text.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from typing import Any

import httpx2  # the openai SDK (v3) ships with this httpx fork
import openai
from openai import AsyncOpenAI
from pydantic import BaseModel

from docqa.config_schema import ProviderConfig
from docqa.domain.errors import ProviderUnavailableError, RateLimitedError
from docqa.ports.llm import LLMDelta, LLMResult, Message

# Words providers use in 429 bodies when the DAILY quota (not the per-minute one) is spent.
DAILY_QUOTA = re.compile(r"per[\s_-]?day|daily|PerDay|\bTPD\b|\bRPD\b", re.IGNORECASE)


@contextmanager
def translate_errors(provider: str) -> Iterator[None]:
    """Map openai SDK errors to RateLimitedError / ProviderUnavailableError."""
    try:
        yield
    except openai.RateLimitError as err:
        daily = bool(DAILY_QUOTA.search(f"{err.message} {err.body}"))
        raise RateLimitedError(f"{provider} rate limit", daily_quota=daily) from err
    except (openai.APITimeoutError, openai.APIConnectionError) as err:
        raise ProviderUnavailableError(f"{provider} timed out or is unreachable") from err
    except openai.APIStatusError as err:
        # 5xx, and also 401/403/404: a broken key or model name makes this provider unusable.
        raise ProviderUnavailableError(f"{provider} returned HTTP {err.status_code}") from err


class OpenAICompatibleLLM:
    """Implements the LLMClient port for one provider.

    Args:
        name: Provider name from config (e.g. "gemini_flash").
        cfg: The provider's config entry.
        api_key: Its API key, or None for keyless providers (Ollama).
        http_client: Optional httpx2 client; tests pass one with a MockTransport.
    """

    def __init__(
        self,
        name: str,
        cfg: ProviderConfig,
        api_key: str | None,
        http_client: httpx2.AsyncClient | None = None,
    ) -> None:
        self.name = name
        self._cfg = cfg
        self._client = AsyncOpenAI(
            base_url=cfg.base_url,
            api_key=api_key or "not-needed",
            timeout=cfg.timeout_s,
            max_retries=0,  # the router decides about retries and fallbacks
            http_client=http_client,
        )

    async def complete(
        self,
        messages: list[Message],
        *,
        json_schema: type[BaseModel] | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> LLMResult:
        """One non-streamed completion."""
        request = self.request(messages, json_schema, max_tokens, temperature)
        with translate_errors(self.name):
            response = await self._client.chat.completions.create(**request)
        usage = response.usage
        return LLMResult(
            text=response.choices[0].message.content or "",
            provider=self.name,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )

    async def stream(
        self,
        messages: list[Message],
        *,
        json_schema: type[BaseModel] | None = None,
        max_tokens: int = 512,
    ) -> AsyncIterator[LLMDelta]:
        """Stream text deltas, then a final delta carrying token usage."""
        request = self.request(messages, json_schema, max_tokens, 0.0)
        request.update(stream=True, stream_options={"include_usage": True})
        input_tokens = output_tokens = 0
        with translate_errors(self.name):
            async for event in await self._client.chat.completions.create(**request):
                if event.usage:
                    input_tokens, output_tokens = (
                        event.usage.prompt_tokens,
                        event.usage.completion_tokens,
                    )
                if event.choices and event.choices[0].delta.content:
                    yield LLMDelta(text=event.choices[0].delta.content, provider=self.name)
        yield LLMDelta(
            done=True, provider=self.name, input_tokens=input_tokens, output_tokens=output_tokens
        )

    def request(
        self,
        messages: list[Message],
        json_schema: type[BaseModel] | None,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        """The chat.completions keyword arguments for this provider."""
        request: dict[str, Any] = {
            "model": self._cfg.model,
            "messages": [m.model_dump() for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_schema is not None and self._cfg.json_schema:
            schema = {"name": json_schema.__name__, "schema": json_schema.model_json_schema()}
            request["response_format"] = {"type": "json_schema", "json_schema": schema}
        elif json_schema is not None and self._cfg.json_mode:
            request["response_format"] = {"type": "json_object"}
        if self._cfg.extra_body:
            request["extra_body"] = self._cfg.extra_body
        return request
