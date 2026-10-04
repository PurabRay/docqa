"""CachingLLM: replays generator outputs so re-running metrics costs no free-tier quota.

Key = sha256(prompt messages, prompt version, model order, config_hash). The messages
already contain the question and the retrieved chunks.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from docqa.ports.llm import LLMClient, LLMDelta, LLMResult, Message


class CachingLLM:
    """Wraps the answer LLM; stream() replays a cached reply when the key matches."""

    def __init__(self, inner: LLMClient, cache_file: Path, salt: str) -> None:
        self.name = inner.name
        self._inner = inner
        self._file = cache_file
        self._salt = salt  # prompt version + model order + config_hash
        self._entries: dict[str, dict[str, Any]] = (
            json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.is_file() else {}
        )

    def key(self, messages: list[Message]) -> str:
        """Cache key for one prompt."""
        payload = json.dumps([m.model_dump() for m in messages]) + self._salt
        return hashlib.sha256(payload.encode()).hexdigest()

    async def complete(self, messages: list[Message], **kwargs: Any) -> LLMResult:
        """Not cached (only the answer stream is)."""
        return await self._inner.complete(messages, **kwargs)

    async def stream(
        self, messages: list[Message], *, max_tokens: int = 512
    ) -> AsyncIterator[LLMDelta]:
        """Replay from the cache, or stream from the inner LLM and remember the reply."""
        key = self.key(messages)
        if key in self._entries:
            entry = self._entries[key]
            yield LLMDelta(text=entry["text"], provider=entry["provider"])
            yield LLMDelta(
                done=True,
                provider=entry["provider"],
                input_tokens=entry["input_tokens"],
                output_tokens=entry["output_tokens"],
            )
            return
        text, final = "", LLMDelta(done=True)
        async for delta in self._inner.stream(messages, max_tokens=max_tokens):
            if delta.done:
                final = delta
            else:
                text += delta.text
            yield delta
        self._entries[key] = {
            "text": text,
            "provider": final.provider,
            "input_tokens": final.input_tokens,
            "output_tokens": final.output_tokens,
        }
        self._save()

    def _save(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        self._file.write_text(json.dumps(self._entries), encoding="utf-8")
