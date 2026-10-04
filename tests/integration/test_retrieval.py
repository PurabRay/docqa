"""Hybrid retrieval and re-ranking on text.pdf with the real models and atlas-local."""

import asyncio
import time
from pathlib import Path

import pytest
import pytest_asyncio

from docqa.adapters.rerank.cross_encoder import FastEmbedReranker
from docqa.adapters.vectorstore.mongo_store import MongoVectorStore
from docqa.bootstrap import build_container, init_database
from docqa.domain.models import AccessFilter, IngestionStatus
from docqa.retrieval.hybrid_retriever import HybridRetriever
from tests.fixtures.make_fixtures import KNOWN_SENTENCES

pytestmark = pytest.mark.asyncio(loop_scope="module")

FIXTURES = Path(__file__).parents[1] / "fixtures"
OWNER = "retrieval-owner"
CLAUSE_QUESTION = "What does clause 14.3(b) say?"
PARAPHRASE = "How long do customers have to send something back and get their money returned?"
REFUND = KNOWN_SENTENCES[3]
RERANK_BUDGET_MS = 400  # PRD latency budget for re-ranking 20 pairs


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def ingested(settings, tmp_path_factory, test_database):
    upload_dir = str(tmp_path_factory.mktemp("uploads"))
    app_settings = settings.model_copy(
        update={"ingestion": settings.ingestion.model_copy(update={"upload_dir": upload_dir})}
    )
    container = build_container(app_settings)
    await init_database(container)
    doc = await container.documents.upload(OWNER, "text.pdf", (FIXTURES / "text.pdf").read_bytes())
    await container.worker.run_once()  # the queued ingestion job
    assert (await container.repository.get(doc.id, OWNER)).status is IngestionStatus.READY
    yield container, AccessFilter(owner_id=OWNER, doc_ids=[doc.id])
    await container.close()


def retriever(container, fusion: str) -> HybridRetriever:
    settings = container.settings
    store = MongoVectorStore(
        container.mongo.db,
        settings.mongodb,
        settings.retrieval.model_copy(update={"fusion": fusion}),
        settings.chunks_collection,
        settings.embedding.dense_model,
        list(settings.search_indexes.values()),
    )
    return HybridRetriever(container.embedder, store, settings.retrieval.fused_k)


@pytest.mark.parametrize("fusion", ["server", "app"])
async def test_clause_number_is_found_by_the_text_branch(ingested, fusion):
    container, access = ingested
    results = await retriever(container, fusion).retrieve(CLAUSE_QUESTION, access)
    clause = next(r for r in results if "14.3(b)" in r.chunk.text)
    assert clause.text_rank == 1
    assert results[0].chunk.id == clause.chunk.id


@pytest.mark.parametrize("fusion", ["server", "app"])
async def test_paraphrase_is_found_by_the_vector_branch(ingested, fusion):
    container, access = ingested
    results = await retriever(container, fusion).retrieve(PARAPHRASE, access)
    refund = next(r for r in results if REFUND in r.chunk.text)
    assert refund.vector_rank is not None and refund.vector_rank <= 3


@pytest.mark.slow
async def test_reranker_puts_the_relevant_passage_first(ingested):
    container, access = ingested
    settings = container.settings
    candidates = await retriever(container, "server").retrieve(PARAPHRASE, access)
    reranker = FastEmbedReranker(
        settings.retrieval.rerank_model, settings.retrieval.rerank_max_tokens
    )

    top = await asyncio.to_thread(reranker.rerank, PARAPHRASE, candidates, settings.retrieval.top_k)
    assert REFUND in top[0].chunk.text

    twenty = (candidates * 20)[:20]  # warm model; time exactly 20 pairs
    start = time.perf_counter()
    await asyncio.to_thread(reranker.rerank, PARAPHRASE, twenty, 20)
    elapsed_ms = (time.perf_counter() - start) * 1000
    print(f"\nre-rank 20 pairs: {elapsed_ms:.0f} ms (budget {RERANK_BUDGET_MS} ms)")
    assert elapsed_ms < RERANK_BUDGET_MS
