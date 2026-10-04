"""Pure builders for search aggregation pipelines. Unit-tested without a database.

Every pipeline takes an AccessFilter and pre-filters owner_id and doc_id inside the
search stage itself, so another owner's chunks are never even scored.

$rankFusion rules (docs/DESIGN.md): sub-pipelines may only use $search,
$vectorSearch, $match, $sort and $geoNear, so $project comes after the fusion.
"""

from __future__ import annotations

from typing import Any

from docqa.config_schema import RetrievalConfig
from docqa.domain.errors import AccessFilterMissingError
from docqa.domain.models import AccessFilter

Stage = dict[str, Any]


def _require(access: AccessFilter | None) -> AccessFilter:
    if access is None or not access.owner_id or not access.doc_ids:
        raise AccessFilterMissingError()
    return access


def vector_filter(access: AccessFilter) -> dict[str, Any]:
    """$vectorSearch.filter: owner and documents (both must be filter fields)."""
    access = _require(access)
    return {"owner_id": access.owner_id, "doc_id": {"$in": access.doc_ids}}


def text_filter(access: AccessFilter) -> list[dict[str, Any]]:
    """$search compound.filter clauses: owner and documents (both token fields)."""
    access = _require(access)
    return [
        {"equals": {"path": "owner_id", "value": access.owner_id}},
        {"in": {"path": "doc_id", "value": access.doc_ids}},
    ]


def text_count_pipeline(access: AccessFilter, index: str) -> list[Stage]:
    """Count the chunks the full-text index can see for these documents."""
    search = {
        "index": index,
        "compound": {"filter": text_filter(access)},
        "count": {"type": "total"},
    }
    return [{"$searchMeta": search}]


def vector_count_pipeline(
    vector: list[float], access: AccessFilter, index: str, limit: int
) -> list[Stage]:
    """Exact (ENN) vector search over these documents, returning ids only."""
    search = {
        "index": index,
        "path": "embedding",
        "queryVector": vector,
        "exact": True,
        "limit": limit,
        "filter": vector_filter(access),
    }
    return [{"$vectorSearch": search}, {"$project": {"_id": 1}}]


def vector_stage(
    vector: list[float], access: AccessFilter, index: str, cfg: RetrievalConfig
) -> Stage:
    """Approximate vector search over the allowed documents."""
    return {
        "$vectorSearch": {
            "index": index,
            "path": "embedding",
            "queryVector": vector,
            "numCandidates": cfg.num_candidates,
            "limit": cfg.vector_k,
            "filter": vector_filter(access),
        }
    }


def text_stage(text: str, access: AccessFilter, index: str) -> Stage:
    """Full-text (BM25) search over the allowed documents."""
    return {
        "$search": {
            "index": index,
            "compound": {
                "must": [{"text": {"query": text, "path": "text"}}],
                "filter": text_filter(access),
            },
        }
    }


def hybrid_pipeline(
    q_vec: list[float],
    q_text: str,
    access: AccessFilter,
    cfg: RetrievalConfig,
    *,
    vector_index: str,
    text_index: str,
) -> list[Stage]:
    """One $rankFusion aggregation over a vector branch and a text branch."""
    branches = {
        "vector": [vector_stage(q_vec, access, vector_index, cfg)],
        "text": [text_stage(q_text, access, text_index), {"$limit": cfg.text_k}],
    }
    return [
        {
            "$rankFusion": {
                "input": {"pipelines": branches},
                "combination": {
                    "weights": {"vector": cfg.weights.vector, "text": cfg.weights.text}
                },
                "scoreDetails": True,
            }
        },
        {"$limit": cfg.fused_k},
        {"$project": {"embedding": 0}},
        {"$addFields": {"fusion": {"$meta": "scoreDetails"}}},
    ]


def vector_only_pipeline(
    q_vec: list[float], access: AccessFilter, cfg: RetrievalConfig, *, vector_index: str
) -> list[Stage]:
    """Vector branch alone (fusion: app, and the vector-only ablation)."""
    return [
        vector_stage(q_vec, access, vector_index, cfg),
        {"$project": {"embedding": 0}},
        {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
    ]


def text_only_pipeline(
    q_text: str, access: AccessFilter, cfg: RetrievalConfig, *, text_index: str
) -> list[Stage]:
    """Text branch alone (fusion: app)."""
    return [
        text_stage(q_text, access, text_index),
        {"$limit": cfg.text_k},
        {"$project": {"embedding": 0}},
        {"$addFields": {"score": {"$meta": "searchScore"}}},
    ]
