"""LLM-as-judge on Mistral (OpenAI-compatible endpoint), rate-limited to ~1 call/s."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from docqa.adapters.llm.openai_compatible import OpenAICompatibleLLM
from docqa.domain.errors import DocQAError
from docqa.generation.prompt_registry import PromptRegistry, PromptTemplate
from docqa.ports.llm import LLMClient, Message
from docqa.settings import Settings

JUDGE_MAX_TOKENS = 200  # a score and one sentence of reason


class Judge:
    """Fills a judge prompt, calls the model at most once per ``interval_s``, parses JSON."""

    def __init__(self, llm: LLMClient, template: PromptTemplate, interval_s: float) -> None:
        self._llm = llm
        self._template = template
        self._interval_s = interval_s
        self._last_call = 0.0

    async def ask(self, **fields: str) -> dict[str, Any] | None:
        """Return the judge's JSON reply, or None if the call or parse failed."""
        user = self._template.user
        for name, value in fields.items():
            user = user.replace("{" + name + "}", value)
        messages = [
            Message(role="system", content=self._template.system),
            Message(role="user", content=user),
        ]
        await self._wait_turn()
        try:
            result = await self._llm.complete(messages, max_tokens=JUDGE_MAX_TOKENS)
        except DocQAError:
            return None
        return parse_json(result.text)

    async def correctness(self, question: str, gold_answer: str, answer: str) -> int | None:
        """1-5 correctness score, or None if the judge failed (counted as a failure)."""
        reply = await self.ask(question=question, gold_answer=gold_answer, answer=answer)
        score = reply.get("score") if reply else None
        return int(score) if isinstance(score, int | float) and 1 <= score <= 5 else None

    async def _wait_turn(self) -> None:
        wait = self._last_call + self._interval_s - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_call = time.monotonic()


def parse_json(text: str) -> dict[str, Any] | None:
    """Parse a JSON object, tolerating a ```json fence around it."""
    cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def build_judge(settings: Settings, prompt_version: str) -> Judge | None:
    """The configured judge, or None when MISTRAL_API_KEY is not set."""
    cfg = settings.eval.judge
    key = settings.optional_secret(cfg.api_key_env) if cfg.api_key_env else None
    if cfg.api_key_env and key is None:
        return None
    template = PromptRegistry(settings.prompts_dir).get("judge", prompt_version)
    return Judge(OpenAICompatibleLLM("judge", cfg, key), template, settings.eval.judge_interval_s)
