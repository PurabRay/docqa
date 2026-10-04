"""Citation accuracy: the cited quote appears verbatim on the cited page (PRD)."""

from __future__ import annotations

import re
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

import pymupdf

WHITESPACE = re.compile(r"\s+")
PageText = Callable[[str, int], str]  # (doc_name, page) -> page text


def quote_on_page(quote: str, page_text: str) -> bool:
    """Verbatim match; only runs of whitespace are treated as equal."""
    return WHITESPACE.sub(" ", quote).strip() in WHITESPACE.sub(" ", page_text)


def citation_accuracy(citations: list[tuple[str, int, str]], page_text: PageText) -> float | None:
    """Share of (doc_name, page, quote) citations whose quote is on that page; None if none."""
    if not citations:
        return None
    ok = sum(quote_on_page(quote, page_text(doc, page)) for doc, page, quote in citations)
    return ok / len(citations)


def corpus_page_text(corpus: Path) -> PageText:
    """Read page text from eval/corpus with PyMuPDF (cached per page)."""

    @lru_cache(maxsize=4096)
    def page_text(doc_name: str, page: int) -> str:
        path = corpus / doc_name
        if not path.is_file():
            return ""
        with pymupdf.open(path) as doc:
            return str(doc[page - 1].get_text()) if 1 <= page <= doc.page_count else ""

    return page_text
