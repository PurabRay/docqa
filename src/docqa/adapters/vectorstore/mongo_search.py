"""Hybrid search over chunks_<model>, in either fusion mode.

server: one $rankFusion aggregation (MongoDB 8.0+).
app: the two branches as separate aggregations run concurrently, fused with rrf_fuse
     (fallback if a cluster rejects $rankFusion).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from pymongo.asynchronous.collection import AsyncCollection
from pymongo.errors import OperationFailure

from docqa.adapters.mongo import codecs
from docqa.adapters.mongo.client import Document, translate_errors
from docqa.adapters.mongo.pipelines import (
    Stage,
    hybrid_pipeline,
    text_only_pipeline,
    vector_only_pipeline,
)
from docqa.config_schema import RetrievalCfg
from docqa.domain.errors import ConfigurationError
from docqa.domain.models import AccessFilter, HybridQuery, RetrievedChunk
from docqa.retrieval.rrf import rrf_fuse

log = logging.getLogger(__name__)


async def search(
    chunks: AsyncCollection[Document], query: HybridQuery, access: AccessFilter, cfg: RetrievalCfg
) -> list[RetrievedChunk]:
    """Run hybrid search in the configured fusion mode, best first."""
    start = time.perf_counter()
    if cfg.fusion == "server":
        results = await _server(chunks, query, access, cfg)
    else:
        results = await _app(chunks, query, access, cfg)
    db_ms = (time.perf_counter() - start) * 1000
    log.info("retrieval.done fusion=%s candidates=%d db_ms=%d", cfg.fusion, len(results), db_ms)
    return results


async def _server(
    chunks: AsyncCollection[Document], query: HybridQuery, access: AccessFilter, cfg: RetrievalCfg
) -> list[RetrievedChunk]:
    pipeline = hybrid_pipeline(query.vector, query.text, access, cfg)
    try:
        docs = await _aggregate(chunks, pipeline)
    except OperationFailure as err:
        raise ConfigurationError(
            f"The cluster rejected $rankFusion ({err.code}); set retrieval.fusion: app."
        ) from err
    return [from_fused(doc) for doc in docs]


async def _app(
    chunks: AsyncCollection[Document], query: HybridQuery, access: AccessFilter, cfg: RetrievalCfg
) -> list[RetrievedChunk]:
    vector_docs, text_docs = await asyncio.gather(
        _aggregate(chunks, vector_only_pipeline(query.vector, access, cfg)),
        _aggregate(chunks, text_only_pipeline(query.text, access, cfg)),
    )
    by_id = {doc["_id"]: doc for doc in [*vector_docs, *text_docs]}
    fused = rrf_fuse(
        {"vector": [d["_id"] for d in vector_docs], "text": [d["_id"] for d in text_docs]},
        weights={"vector": cfg.w_vector, "text": cfg.w_text},
    )
    return [
        codecs.retrieved_from_bson(
            by_id[item.id], item.score, item.ranks.get("vector"), item.ranks.get("text")
        )
        for item in fused[: cfg.fused_k]
    ]


async def _aggregate(chunks: AsyncCollection[Document], pipeline: list[Stage]) -> list[Document]:
    with translate_errors():
        return await (await chunks.aggregate(pipeline)).to_list()


def from_fused(doc: Document) -> RetrievedChunk:
    """Read the fused score and each branch's rank from $rankFusion's scoreDetails.

    A branch that did not return the chunk reports rank "NA"; that becomes None.
    """
    details: dict[str, Any] = doc.get("fusion", {})
    ranks = {
        d["inputPipelineName"]: d["rank"]
        for d in details.get("details", [])
        if isinstance(d.get("rank"), int)
    }
    return codecs.retrieved_from_bson(
        doc, float(details.get("value", 0.0)), ranks.get("vector"), ranks.get("text")
    )
