"""Pure builders for search aggregation pipelines. Unit-tested without a database.

Every pipeline takes an AccessFilter and pre-filters owner_id and doc_id inside the
search stage itself, so another owner's chunks are never even scored.
The hybrid $rankFusion pipeline arrives in M3.
"""

from __future__ import annotations

from typing import Any

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
