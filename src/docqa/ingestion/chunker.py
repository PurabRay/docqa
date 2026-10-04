"""Heading- and page-aware chunking (FR-4).

1. Each page is cut into units: headings, tables and sentences, each tagged
   with its page number and the paragraph (block) it came from.
2. Units are packed greedily into chunks of at most ``max_tokens``. A heading
   starts a new chunk, a table gets a chunk of its own (with the headings just
   above it), and each new chunk
   repeats up to ``overlap_tokens`` of trailing sentences from the previous one.

Because every unit remembers its page, a chunk that crosses a page break gets
the right page_start and page_end.

CLI: python -m docqa.ingestion.chunker path/to/file.pdf
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from docqa.config_schema import ChunkingConfig
from docqa.domain.models import Chunk, Page, chunk_id
from docqa.ingestion.text_split import (
    SPLIT_PATTERNS,
    TokenCounter,
    split_sentences,
    split_to_fit,
    token_counter,
)

PARAGRAPHS = SPLIT_PATTERNS[0]


@dataclass(frozen=True)
class Unit:
    """The smallest piece the packer moves around."""

    text: str
    page: int
    block: int  # units from the same paragraph are joined with a space
    kind: Literal["text", "heading", "table"]
    section: str | None


def chunk_pages(
    pages: list[Page], cfg: ChunkingConfig, *, doc_id: str, owner_id: str, filename: str
) -> list[Chunk]:
    """Split pages into chunks with page ranges, section headings and stable ids."""
    count = token_counter(cfg.tokenizer)
    flagged_pages = {page.number for page in pages if page.flagged_injection}
    chunks = []
    for index, group in enumerate(pack(to_units(pages, cfg.max_tokens, count), cfg, count)):
        text, offsets = join_with_pages(group)
        chunks.append(
            Chunk(
                id=chunk_id(doc_id, index),
                doc_id=doc_id,
                owner_id=owner_id,
                filename=filename,
                text=text,
                page_start=min(unit.page for unit in group),
                page_end=max(unit.page for unit in group),
                section=group[0].section,
                content_hash=hashlib.sha256(text.encode()).hexdigest(),
                flagged_injection=any(unit.page in flagged_pages for unit in group),
                page_offsets=offsets,
            )
        )
    return chunks


def to_units(pages: list[Page], max_tokens: int, count: TokenCounter) -> list[Unit]:
    """Cut pages into headings, tables and sentences, each at most ``max_tokens``."""
    units: list[Unit] = []
    section: str | None = None
    block = 0
    for page in pages:
        for paragraph in PARAGRAPHS.split(page.text):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            block += 1
            kind = _kind(paragraph)
            if kind == "heading":
                section = paragraph.lstrip("#").strip(" *")
            pieces = [paragraph] if kind != "text" else split_sentences(paragraph)
            for piece in pieces:
                for part in split_to_fit(piece, max_tokens, count):
                    units.append(Unit(part.strip(), page.number, block, kind, section))
    return [unit for unit in units if unit.text]


def pack(units: list[Unit], cfg: ChunkingConfig, count: TokenCounter) -> list[list[Unit]]:
    """Group units into chunks of at most ``cfg.max_tokens`` tokens."""
    groups: list[list[Unit]] = []
    current: list[Unit] = []
    for unit in units:
        boundary = _is_boundary(current, unit, cfg.respect_headings)
        if current and (boundary or count(join([*current, unit])) > cfg.max_tokens):
            groups.append(current)
            current = [] if boundary else _overlap(current, cfg.overlap_tokens, count)
            if current and count(join([*current, unit])) > cfg.max_tokens:
                current = []
        current.append(unit)
    if current:
        groups.append(current)
    return groups


def join(units: list[Unit]) -> str:
    """Join units: a space inside a paragraph, a blank line between paragraphs."""
    return join_with_pages(units)[0]


def join_with_pages(units: list[Unit]) -> tuple[str, list[int]]:
    """Join units and return the text offsets where each new page begins."""
    text, offsets = "", []
    for previous, unit in zip([None, *units], units, strict=False):
        if previous is not None:
            text += " " if previous.block == unit.block else "\n\n"
            offsets += [len(text)] * (unit.page - previous.page)
        text += unit.text
    return text, offsets


def _kind(paragraph: str) -> Literal["text", "heading", "table"]:
    if paragraph.startswith("#"):
        return "heading"
    lines = paragraph.splitlines()
    if all(line.startswith("|") for line in lines):
        return "table"
    return "text"


def _is_boundary(current: list[Unit], unit: Unit, respect_headings: bool) -> bool:
    """Tables get their own chunk (with the headings above them); headings start chunks."""
    if not current or current[-1].kind == "table":
        return bool(current)
    has_body = any(u.kind != "heading" for u in current)
    if unit.kind == "table":
        return has_body
    return respect_headings and unit.kind == "heading" and has_body


def _overlap(previous: list[Unit], overlap_tokens: int, count: TokenCounter) -> list[Unit]:
    """Trailing sentences of the previous chunk that fit in ``overlap_tokens``."""
    tail: list[Unit] = []
    for unit in reversed(previous):
        if unit.kind != "text" or count(join([unit, *tail])) > overlap_tokens:
            break
        tail.insert(0, unit)
    return tail


def main(path: str) -> None:
    """Print each chunk's page range, token count and opening words."""
    from docqa.ingestion.extractor import extract_pages
    from docqa.settings import load_settings

    cfg = load_settings(env_file=None).chunking
    pages = extract_pages(Path(path), doc_id="cli")
    count = token_counter(cfg.tokenizer)
    for chunk in chunk_pages(pages, cfg, doc_id="cli", owner_id="cli", filename=Path(path).name):
        preview = chunk.text[:70].replace("\n", " ")
        pages_label = f"p.{chunk.page_start}-{chunk.page_end}"
        print(f"{pages_label:<9} {count(chunk.text):>4} tok  [{chunk.section}]  {preview}")


if __name__ == "__main__":
    main(sys.argv[1])
