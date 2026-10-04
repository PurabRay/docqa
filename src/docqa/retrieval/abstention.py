"""Decide whether to answer or abstain (FR-7)."""

from __future__ import annotations

from docqa.domain.models import RetrievedChunk


def best_score(chunks: list[RetrievedChunk]) -> float | None:
    """Highest re-ranker score, falling back to the fused score when not re-ranked."""
    scores = [c.rerank_score if c.rerank_score is not None else c.fused_score for c in chunks]
    return max(scores) if scores else None


def should_abstain(chunks: list[RetrievedChunk], threshold: float) -> bool:
    """Abstain when nothing was retrieved or the best score is below ``threshold``."""
    score = best_score(chunks)
    return score is None or score < threshold
