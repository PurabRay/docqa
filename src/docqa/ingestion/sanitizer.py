"""Strip control characters and flag instruction-like text. Never deletes content."""

from __future__ import annotations

import re
from collections.abc import Iterable

from docqa.domain.models import Page

# Control characters except tab (\x09), newline (\x0a) and carriage return (\x0d).
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def compile_patterns(patterns: Iterable[str]) -> list[re.Pattern[str]]:
    """Compile the configured injection patterns, case-insensitive."""
    return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]


def looks_like_injection(text: str, patterns: list[re.Pattern[str]]) -> bool:
    """True if any pattern matches the text."""
    return any(pattern.search(text) for pattern in patterns)


def sanitize(page: Page, patterns: list[re.Pattern[str]]) -> Page:
    """Return a cleaned copy of the page, flagged if it contains instruction-like text."""
    text = CONTROL_CHARS.sub("", page.text)
    return page.model_copy(
        update={"text": text, "flagged_injection": looks_like_injection(text, patterns)}
    )
