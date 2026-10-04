"""Token counting and a recursive splitter: paragraphs, lines, sentences, words, characters."""

from __future__ import annotations

import re
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

import tiktoken
from tiktoken.load import load_tiktoken_bpe

TokenCounter = Callable[[str], int]

# cl100k_base ships with the repo so token counting never needs the network (CLAUDE.md
# rule 9). Pattern, special tokens and hash are copied from tiktoken_ext.openai_public.
TOKENIZER_DIR = Path(__file__).parent / "tokenizers"
CL100K_SHA256 = "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"
CL100K_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+"""
    r"""|\s++$|\s*[\r\n]|\s+(?!\S)|\s"""
)
CL100K_SPECIAL_TOKENS = {
    "<|endoftext|>": 100257,
    "<|fim_prefix|>": 100258,
    "<|fim_middle|>": 100259,
    "<|fim_suffix|>": 100260,
    "<|endofprompt|>": 100276,
}

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
    """Return a function that counts tokens with the named tiktoken encoding.

    cl100k_base is loaded from the repo; any other encoding is fetched by tiktoken.
    """
    if encoding_name == "cl100k_base":
        encoding = tiktoken.Encoding(
            name="cl100k_base",
            pat_str=CL100K_PATTERN,
            mergeable_ranks=load_tiktoken_bpe(
                str(TOKENIZER_DIR / "cl100k_base.tiktoken"), expected_hash=CL100K_SHA256
            ),
            special_tokens=CL100K_SPECIAL_TOKENS,
        )
    else:
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
