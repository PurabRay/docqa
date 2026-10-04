"""Reciprocal Rank Fusion in Python (fusion: app, and the in-memory test store).

Same formula as MongoDB's $rankFusion: score(d) = sum over lists of
weight / (k + rank), with 1-based ranks and k = 60.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic import BaseModel

RRF_K = 60  # fixed by $rankFusion; kept equal so both fusion modes rank alike


class ScoredId(BaseModel):
    """A fused result: id, RRF score, and its 1-based rank in each input list."""

    id: str
    score: float
    ranks: dict[str, int]


def rrf_fuse(
    ranked_lists: Mapping[str, Sequence[str]],
    k: int = RRF_K,
    weights: Mapping[str, float] | None = None,
) -> list[ScoredId]:
    """Fuse named ranked lists of ids, best first. Ties keep first-seen order."""
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    for name, ids in ranked_lists.items():
        weight = 1.0 if weights is None else weights.get(name, 1.0)
        for rank, doc_id in enumerate(ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)
            ranks.setdefault(doc_id, {})[name] = rank
    order = sorted(scores, key=lambda doc_id: scores[doc_id], reverse=True)
    return [ScoredId(id=doc_id, score=scores[doc_id], ranks=ranks[doc_id]) for doc_id in order]
