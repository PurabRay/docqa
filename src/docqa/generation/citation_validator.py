"""Keep only citations we can prove (FR-6).

A citation survives if its chunk_id was actually retrieved AND its quote appears in
that chunk's text (after normalising whitespace and curly quotes). The page is the
one that holds the quote, which matters when a chunk spans a page break.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from docqa.domain.models import Citation, RetrievedChunk
from docqa.generation.schemas import CitedClaim, LLMAnswer

QUOTE_CHARS = str.maketrans(
    {
        "\N{LEFT SINGLE QUOTATION MARK}": "'",
        "\N{RIGHT SINGLE QUOTATION MARK}": "'",
        "\N{LEFT DOUBLE QUOTATION MARK}": '"',
        "\N{RIGHT DOUBLE QUOTATION MARK}": '"',
    }
)
WHITESPACE = re.compile(r"\s+")


class ValidatedAnswer(BaseModel):
    """The answer text with only verified citations."""

    text: str
    citations: list[Citation]
    dropped: int
    verified: bool  # False when no citation survived
    sufficient_context: bool


def normalise(text: str) -> str:
    """Straight quotes, single spaces, no Markdown bold markers, trimmed."""
    return WHITESPACE.sub(" ", text.translate(QUOTE_CHARS).replace("**", "")).strip()


def validate_citations(answer: LLMAnswer, chunks: list[RetrievedChunk]) -> ValidatedAnswer:
    """Drop citations that point at unretrieved chunks or quote text the chunk lacks."""
    by_id = {c.chunk.id: c for c in chunks}
    citations = [c for claim in answer.citations if (c := _verify(claim, by_id)) is not None]
    return ValidatedAnswer(
        text=answer.answer,
        citations=citations,
        dropped=len(answer.citations) - len(citations),
        verified=bool(citations),
        sufficient_context=answer.sufficient_context,
    )


def _verify(claim: CitedClaim, by_id: dict[str, RetrievedChunk]) -> Citation | None:
    item = by_id.get(claim.chunk_id)
    quote = normalise(claim.quote)
    if item is None or not quote:
        return None
    page = _page_of_quote(item, quote)
    if page is None:
        return None
    return Citation(
        chunk_id=claim.chunk_id, doc_name=item.chunk.filename, page=page, quote=claim.quote
    )


def _page_of_quote(item: RetrievedChunk, quote: str) -> int | None:
    """The page whose part of the chunk contains the quote; None if it is not there."""
    chunk = item.chunk
    if quote not in normalise(chunk.text):
        return None
    starts = [0, *chunk.page_offsets]
    ends = [*chunk.page_offsets, len(chunk.text)]
    for start, end in zip(starts, ends, strict=True):
        if quote in normalise(chunk.text[start:end]):
            return chunk.page_at(start)
    return chunk.page_start  # the quote straddles a page break: cite where it begins
