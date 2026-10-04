"""Rewrite a follow-up into a standalone question using the last turns (FR-10)."""

from __future__ import annotations

from docqa.domain.errors import AllProvidersUnavailableError
from docqa.domain.models import Turn
from docqa.generation.prompt_registry import PromptTemplate
from docqa.ports.llm import LLMClient, Message


class QueryRewriter:
    """No-op on the first turn; otherwise asks the LLM for a standalone question."""

    def __init__(self, llm: LLMClient, template: PromptTemplate, max_tokens: int) -> None:
        self._llm = llm
        self._template = template
        self._max_tokens = max_tokens

    async def rewrite(self, question: str, history: list[Turn]) -> str:
        """Return the standalone question (the original if there is no history or no LLM)."""
        if not history:
            return question
        conversation = "\n".join(f"{turn.role}: {turn.content_scrubbed}" for turn in history)
        user = self._template.user.replace("{history}", conversation).replace(
            "{question}", question
        )
        messages = [
            Message(role="system", content=self._template.system),
            Message(role="user", content=user),
        ]
        try:
            result = await self._llm.complete(messages, max_tokens=self._max_tokens)
        except AllProvidersUnavailableError:
            return question  # fail soft: retrieval still works on the raw follow-up
        lines = result.text.strip().splitlines()
        first = lines[0].strip().strip("\"'").strip() if lines else ""
        return first or question
