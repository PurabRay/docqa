"""Turn a validated PDF into one Markdown Page per PDF page (FR-3).

pymupdf4llm gives reading order, headings and Markdown tables. If a table on a
page comes out garbled (rows with different column counts), that page's tables
are re-extracted with pdfplumber.
"""

from __future__ import annotations

import re
from pathlib import Path

import pdfplumber
import pymupdf4llm

from docqa.domain.errors import InvalidUploadError
from docqa.domain.models import Page

TABLE_ROW = re.compile(r"^\|.*\|\s*$")


def extract_pages(path: Path, doc_id: str) -> list[Page]:
    """Extract every page with its headings; page numbers are 1-based.

    Raises:
        InvalidUploadError: The PDF libraries could not read the file.
    """
    try:
        return _extract(path, doc_id)
    except Exception as err:
        # Vendor boundary: PyMuPDF, pymupdf4llm and pdfplumber raise many unrelated
        # types (RuntimeError, ValueError, IndexError, ...) on malformed PDFs.
        raise InvalidUploadError("The PDF could not be read; it may be damaged.") from err


def _extract(path: Path, doc_id: str) -> list[Page]:
    page_chunks = pymupdf4llm.to_markdown(str(path), page_chunks=True, show_progress=False)
    pages = []
    for number, chunk in enumerate(page_chunks, start=1):
        text = chunk["text"]
        if is_table_garbled(text):
            text = replace_tables(text, _pdfplumber_tables(path, number))
        headings = [title for _level, title, _page in chunk.get("toc_items", [])]
        pages.append(
            Page(
                doc_id=doc_id,
                number=number,
                text=text,
                headings=headings or markdown_headings(text),
            )
        )
    return pages


def markdown_headings(text: str) -> list[str]:
    """Heading lines ("# **Title**") without the Markdown markup."""
    return [line.lstrip("#").strip(" *") for line in text.splitlines() if line.startswith("#")]


def is_table_garbled(text: str) -> bool:
    """True if any Markdown table on the page has rows with different column counts."""
    return any(len({row.count("|") for row in table}) > 1 for table in _markdown_tables(text))


def replace_tables(text: str, tables: list[str]) -> str:
    """Drop the page's Markdown table rows and append the replacement tables."""
    kept = [line for line in text.splitlines() if not TABLE_ROW.match(line)]
    return "\n".join(kept).rstrip() + "\n\n" + "\n\n".join(tables) + "\n"


def to_markdown_table(rows: list[list[str | None]]) -> str:
    """Render pdfplumber rows as a Markdown table (first row is the header)."""
    cells = [[(cell or "").replace("\n", " ").strip() for cell in row] for row in rows]
    lines = ["|" + "|".join(row) + "|" for row in cells]
    lines.insert(1, "|" + "|".join("---" for _ in cells[0]) + "|")
    return "\n".join(lines)


def _markdown_tables(text: str) -> list[list[str]]:
    tables: list[list[str]] = []
    current: list[str] = []
    for line in [*text.splitlines(), ""]:
        if TABLE_ROW.match(line):
            current.append(line)
        elif current:
            tables.append(current)
            current = []
    return tables


def _pdfplumber_tables(path: Path, number: int) -> list[str]:
    with pdfplumber.open(path) as pdf:
        tables = pdf.pages[number - 1].extract_tables()
    return [to_markdown_table(rows) for rows in tables if rows]
