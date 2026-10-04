"""Check that the cluster at MONGODB_URI accepts $rankFusion with owner pre-filters.

Writes a few tiny chunks for two owners into the real chunks collection, waits for
the search indexes, runs hybrid_pipeline as owner A, prints the results, then deletes
the test chunks. Run `make initdb` first (it reuses the two existing search indexes,
so the free cluster's 3-index limit is respected). Usage:

    MONGODB_URI=$MONGODB_URI_ATLAS uv run python scripts/smoke_rankfusion.py
"""

from __future__ import annotations

import asyncio
import sys

from pymongo.errors import OperationFailure

from docqa.adapters.mongo.client import MongoConnection
from docqa.adapters.mongo.pipelines import hybrid_pipeline
from docqa.adapters.vectorstore.mongo_store import MongoVectorStore
from docqa.domain.models import AccessFilter, Chunk, EmbeddedChunk, chunk_id
from docqa.settings import Settings, load_settings

OWNERS = {
    "smoke-owner-a": ["The refund window is 30 days.", "Shipping takes five days."],
    "smoke-owner-b": ["The refund window is 30 days for owner B only.", "Owner B secret clause."],
}


def tiny_chunks(settings: Settings) -> list[EmbeddedChunk]:
    """Two chunks per owner; owner B's first chunk matches the query vector exactly."""
    dim, items = settings.embedding.dim, []
    for owner, texts in OWNERS.items():
        for i, text in enumerate(texts):
            vector = [0.01] * dim
            vector[0 if owner.endswith("b") and i == 0 else i + 1] = 1.0
            chunk = Chunk(
                id=chunk_id(owner, i),
                doc_id=f"doc-{owner}",
                owner_id=owner,
                filename="smoke.pdf",
                text=text,
                page_start=1,
                page_end=1,
                content_hash="smoke",
            )
            items.append(
                EmbeddedChunk(
                    chunk=chunk, embedding=vector, embed_model=settings.embedding.dense_model
                )
            )
    return items


async def run(settings: Settings, mongo: MongoConnection) -> int:
    """Write, search as owner A, report."""
    store = MongoVectorStore(
        mongo.db,
        settings.mongodb,
        settings.retrieval,
        settings.chunks_collection,
        settings.embedding.dense_model,
        list(settings.search_indexes.values()),
    )
    await store.upsert(tiny_chunks(settings))
    for owner, texts in OWNERS.items():
        await store.wait_until_searchable(f"doc-{owner}", owner, len(texts), timeout_s=120)

    query_vector = [1.0] + [0.0] * (settings.embedding.dim - 1)
    access = AccessFilter(
        owner_id="smoke-owner-a", doc_ids=["doc-smoke-owner-a", "doc-smoke-owner-b"]
    )
    pipeline = hybrid_pipeline(
        query_vector,
        "refund window",
        access,
        settings.retrieval,
        vector_index=settings.mongodb.vector_index,
        text_index=settings.mongodb.text_index,
    )
    try:
        docs = await (await mongo.db[settings.chunks_collection].aggregate(pipeline)).to_list()
    except OperationFailure as err:
        print(f"$rankFusion REJECTED ({err.code}): {err}")
        print("Set retrieval.fusion: app in config/free.yaml.")
        return 1
    print("$rankFusion accepted. Results when searching as smoke-owner-a:")
    for doc in docs:
        print(f"  {doc['owner_id']:<14} {doc['fusion']['value']:.4f}  {doc['text']}")
    leaked = [doc for doc in docs if doc["owner_id"] != "smoke-owner-a"]
    print("Owner filter OK" if not leaked else f"LEAK: {len(leaked)} chunks of another owner")
    return 1 if leaked or not docs else 0


async def main() -> int:
    """Run the smoke test, always deleting the test chunks."""
    settings = load_settings()
    mongo = MongoConnection(settings.mongodb_uri(), settings.mongodb)
    try:
        return await run(settings, mongo)
    finally:
        await mongo.db[settings.chunks_collection].delete_many({"owner_id": {"$in": list(OWNERS)}})
        await mongo.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
