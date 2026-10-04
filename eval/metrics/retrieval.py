"""Retrieval metrics on gold pages: Recall@k (chunks sent to the LLM) and MRR@k (fused list)."""

from __future__ import annotations

from pydantic import BaseModel


class ChunkRef(BaseModel):
    """Where a retrieved chunk came from."""

    doc_name: str
    page_start: int
    page_end: int


def hits_gold(chunk: ChunkRef, gold_doc: str, gold_pages: list[int]) -> bool:
    """True if the chunk is from the gold document and covers a gold page."""
    return chunk.doc_name == gold_doc and any(
        chunk.page_start <= p <= chunk.page_end for p in gold_pages
    )


def recall_at_k(top: list[ChunkRef], gold_doc: str, gold_pages: list[int], k: int) -> float:
    """1.0 if a gold page is among the first ``k`` chunks, else 0.0."""
    return 1.0 if any(hits_gold(c, gold_doc, gold_pages) for c in top[:k]) else 0.0


def reciprocal_rank(fused: list[ChunkRef], gold_doc: str, gold_pages: list[int], k: int) -> float:
    """1/rank of the first gold-page chunk within the first ``k``, else 0.0."""
    for rank, chunk in enumerate(fused[:k], start=1):
        if hits_gold(chunk, gold_doc, gold_pages):
            return 1.0 / rank
    return 0.0
