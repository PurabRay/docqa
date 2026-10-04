"""Token counting and a recursive splitter: paragraphs, lines, sentences, words, characters."""

from __future__ import annotations

import re
from collections.abc import Callable
from functools import lru_cache

import tiktoken

TokenCounter = Callable[[str], int]

# Tried in order until every piece fits.
SPLIT_PATTERNS = [
    re.compile(r"\n\s*\n"),  # paragraphs
    re.compile(r"\n"),  # lines
    re.compile(r"(?<=[.!?])\s+"),  # sentences
    re.compile(r"\s+"),  # words
]
SENTENCE_END = SPLIT_PATTERNS[2]


@lru_cache(maxsize=4)
def token_counter(encoding_name: str) -> TokenCounter:
    """Return a function that counts tokens with the named tiktoken encoding."""
    encoding = tiktoken.get_encoding(encoding_name)
    return lambda text: len(encoding.encode(text, disallowed_special=()))


def split_sentences(text: str) -> list[str]:
    """Split a paragraph into sentences (whitespace between them is dropped)."""
    return [sentence for sentence in SENTENCE_END.split(text) if sentence.strip()]


def split_to_fit(text: str, max_tokens: int, count: TokenCounter) -> list[str]:
    """Split ``text`` into pieces of at most ``max_tokens`` tokens, keeping all characters.

    Only whitespace at split points is dropped. A single character that alone
    exceeds ``max_tokens`` is returned as is.
    """
    if count(text) <= max_tokens or len(text) <= 1:
        return [text]
    for pattern in SPLIT_PATTERNS:
        parts = [part for part in pattern.split(text) if part.strip()]
        if len(parts) > 1:
            return [piece for part in parts for piece in split_to_fit(part, max_tokens, count)]
    middle = len(text) // 2
    return split_to_fit(text[:middle], max_tokens, count) + split_to_fit(
        text[middle:], max_tokens, count
    )
