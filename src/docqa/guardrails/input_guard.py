"""Check a question before any work: length limit (blocks), injection heuristics (flags)."""

from __future__ import annotations

import re

from docqa.domain.errors import InputRejectedError
from docqa.ingestion.sanitizer import looks_like_injection


class InputGuard:
    """Rejects over-long or empty questions; reports instruction-like ones."""

    def __init__(self, max_chars: int, injection_patterns: list[re.Pattern[str]]) -> None:
        self._max_chars = max_chars
        self._patterns = injection_patterns

    def check(self, question: str) -> bool:
        """Return True if the question looks like a prompt injection (answer it anyway).

        Raises:
            InputRejectedError: The question is empty or longer than the limit.
        """
        if not question.strip():
            raise InputRejectedError("The question is empty.")
        if len(question) > self._max_chars:
            raise InputRejectedError(f"Questions are limited to {self._max_chars} characters.")
        return looks_like_injection(question, self._patterns)
