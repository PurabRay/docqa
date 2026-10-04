"""Every VectorStore adapter must pass this suite.

memory runs as a unit test; mongo-server and mongo-app need atlas-local (integration).
Pattern from atlas-rag-multitenant's tenant-isolation test: owner B's chunks are made
MORE similar to the query than owner A's, and must still never reach owner A.
"""

import uuid

import pytest

from docqa.adapters.vectorstore.memory_store import MemoryVectorStore
from docqa.adapters.vectorstore.mongo_store import MongoVectorStore
from docqa.domain.models import AccessFilter, Chunk, EmbeddedChunk, HybridQuery, chunk_id

STORES = [
    pytest.param("memory", marks=pytest.mark.unit),
    pytest.param("server", marks=pytest.mark.integration, id="mongo-server"),
    pytest.param("app", marks=pytest.mark.integration, id="mongo-app"),
]
QUERY_TEXT = "refund window"


def unit_vector(dim: int, index: int, nudge: float = 0.0) -> list[float]:
    vector = [0.0] * dim
    vector[index] = 1.0
    vector[0] += nudge  # a little similarity to the query direction (index 0)
    return vector


def make_chunks(owner, doc_id, count, vector, text, embed_model) -> list[EmbeddedChunk]:
    return [
        EmbeddedChunk(
            chunk=Chunk(
                id=chunk_id(doc_id, i),
                doc_id=doc_id,
                owner_id=owner,
                filename=f"{doc_id}.pdf",
                text=f"{text} (part {i})",
                page_start=1,
                page_end=1,
                content_hash=str(i),
            ),
            embedding=vector,
            embed_model=embed_model,
        )
        for i in range(count)
    ]


@pytest.fixture(params=STORES)
async def store(request, settings):
    retrieval = settings.retrieval
    if request.param == "memory":
        yield MemoryVectorStore(retrieval)
        return
    container = request.getfixturevalue("initialized")
    yield MongoVectorStore(
        container.mongo.db,
        settings.mongodb,
        retrieval.model_copy(update={"fusion": request.param}),
        settings.chunks_collection,
        settings.embedding.dense_model,
        list(settings.search_indexes.values()),
    )


@pytest.fixture
def tenants(settings):
    """Unique owner and document ids, so tests never see each other's data."""
    tag = uuid.uuid4().hex[:8]
    return {name: f"{name}-{tag}" for name in ("owner_a", "owner_b", "doc_a1", "doc_a2", "doc_b1")}


async def write(store, items):
    await store.upsert(items)
    for doc_id, owner in {(i.chunk.doc_id, i.chunk.owner_id) for i in items}:
        expected = sum(1 for i in items if i.chunk.doc_id == doc_id)
        await store.wait_until_searchable(doc_id, owner, expected, timeout_s=60)


@pytest.fixture
async def seeded(store, settings, tenants):
    dim, model, t = settings.embedding.dim, settings.embedding.dense_model, tenants
    a1 = make_chunks(t["owner_a"], t["doc_a1"], 12, unit_vector(dim, 1, 0.3), "policy text", model)
    a2 = make_chunks(t["owner_a"], t["doc_a2"], 3, unit_vector(dim, 2, 0.2), "refund notes", model)
    # Owner B: exactly the query vector and the query words. Must never reach owner A.
    b1 = make_chunks(
        t["owner_b"], t["doc_b1"], 25, unit_vector(dim, 0), "refund window refund window", model
    )
    await write(store, a1 + a2 + b1)
    query = HybridQuery(text=QUERY_TEXT, vector=unit_vector(dim, 0))
    return store, query, t, a1 + a2


@pytest.mark.parametrize("k", [1, 5, 20])
async def test_another_owners_chunks_never_return(seeded, k):
    store, query, t, _ = seeded
    # Owner A even names owner B's document id: the owner filter must still win.
    access = AccessFilter(owner_id=t["owner_a"], doc_ids=[t["doc_a1"], t["doc_a2"], t["doc_b1"]])
    results = await store.search(query, access, limit=k)
    assert results, "owner A should still get their own chunks"
    assert len(results) <= k
    assert {r.chunk.owner_id for r in results} == {t["owner_a"]}


async def test_one_document_search_stays_in_that_document(seeded):
    store, query, t, _ = seeded
    results = await store.search(
        query, AccessFilter(owner_id=t["owner_a"], doc_ids=[t["doc_a2"]]), 20
    )
    assert results and {r.chunk.doc_id for r in results} == {t["doc_a2"]}


async def test_results_carry_fused_scores_and_ranks(seeded):
    store, query, t, _ = seeded
    results = await store.search(
        query, AccessFilter(owner_id=t["owner_a"], doc_ids=[t["doc_a2"]]), 20
    )
    scores = [r.fused_score for r in results]
    assert scores == sorted(scores, reverse=True) and scores[0] > 0
    assert all(r.vector_rank or r.text_rank for r in results)


async def test_upsert_is_idempotent(seeded):
    store, query, t, own_chunks = seeded
    access = AccessFilter(owner_id=t["owner_a"], doc_ids=[t["doc_a1"], t["doc_a2"]])
    before = sorted(r.chunk.id for r in await store.search(query, access, 20))
    await write(store, own_chunks)
    assert sorted(r.chunk.id for r in await store.search(query, access, 20)) == before


async def test_delete_removes_only_that_document(seeded):
    store, query, t, _ = seeded
    assert await store.delete_document(t["doc_a1"], t["owner_a"]) == 12
    assert await store.delete_document(t["doc_b1"], t["owner_a"]) == 0  # not A's to delete
    remaining = await store.search(
        query, AccessFilter(owner_id=t["owner_a"], doc_ids=[t["doc_a2"]]), 20
    )
    assert len(remaining) == 3
    still_b = await store.search(
        query, AccessFilter(owner_id=t["owner_b"], doc_ids=[t["doc_b1"]]), 20
    )
    assert still_b
