"""QueryService with fake ports: event order, cache, abstention, degraded path."""

import json

import pytest

from docqa.adapters.cache.ttl_answer_cache import TTLAnswerCache
from docqa.adapters.llm.stub import StubLLM
from docqa.adapters.rerank.noop import NoopReranker
from docqa.adapters.tracing.noop_tracer import NoopTracer
from docqa.domain.errors import (
    AllProvidersUnavailableError,
    DocumentNotReadyError,
    RateLimitedError,
)
from docqa.domain.models import (
    Chunk,
    DocumentRecord,
    IngestionStatus,
    QueryRequest,
    RetrievedChunk,
)
from docqa.generation.answer_generator import AnswerGenerator
from docqa.generation.prompt_builder import PromptBuilder
from docqa.generation.prompt_registry import PromptRegistry
from docqa.guardrails.input_guard import InputGuard
from docqa.guardrails.rate_limiter import RateLimiter
from docqa.ingestion.sanitizer import compile_patterns
from docqa.ingestion.text_split import token_counter
from docqa.retrieval.query_rewriter import QueryRewriter
from docqa.services.query_service import QueryDeps, QueryService
from docqa.settings import load_settings

SETTINGS = load_settings(env_file=None)
PROMPTS = PromptRegistry(SETTINGS.prompts_dir)
REFUND = "The refund window is 30 days."
REQUEST = QueryRequest(
    owner_id="u1", session_id="s1", question="What is the refund window?", doc_ids=["d1"]
)


def chunk(cid="c1", text=REFUND, score=0.5):
    c = Chunk(
        id=cid,
        doc_id="d1",
        owner_id="u1",
        filename="text.pdf",
        text=text,
        page_start=3,
        page_end=3,
        content_hash="h",
    )
    return RetrievedChunk(chunk=c, fused_score=score)


class FakeRepo:
    def __init__(self, history=()):
        self.turns = list(history)

    async def get(self, doc_id, owner_id):
        return DocumentRecord(
            id=doc_id,
            owner_id=owner_id,
            filename="text.pdf",
            sha256="a" * 64,
            page_count=12,
            status=IngestionStatus.READY,
            embed_model="m",
        )

    async def last_turns(self, session_id, n=3):
        return self.turns[-n:]

    async def append_turn(self, turn):
        self.turns.append(turn)


class FakeRetriever:
    def __init__(self, results):
        self.results, self.calls = results, 0

    async def retrieve(self, question, access):
        self.calls += 1
        return self.results


class Unavailable:
    name = "down"

    async def complete(self, *a, **k):
        raise AllProvidersUnavailableError()

    async def stream(self, *a, **k):
        raise AllProvidersUnavailableError()
        yield  # makes this an async generator


def reply(chunk_id="c1", quote=REFUND, sufficient=True):
    return json.dumps(
        {
            "answer": "Customers have 30 days.",
            "sufficient_context": sufficient,
            "citations": [{"chunk_id": chunk_id, "quote": quote}],
        }
    )


def service(results=None, llm=None, repo=None, threshold=0.0):
    results = results if results is not None else [chunk()]
    llm = llm or StubLLM(reply(), piece_size=6)
    retriever = FakeRetriever(list(results))
    deps = QueryDeps(
        repo=repo or FakeRepo(),
        cache=TTLAnswerCache(3600),
        tracer=NoopTracer(),
        input_guard=InputGuard(500, compile_patterns(SETTINGS.ingestion.injection_patterns)),
        rate_limiter=RateLimiter(10),
        rewriter=QueryRewriter(StubLLM("rewritten?"), PROMPTS.get("rewrite", "v1"), 50),
        retriever=retriever,
        reranker=NoopReranker(),
        prompt_builder=PromptBuilder(6000, token_counter("cl100k_base")),
        template=PROMPTS.get("answer", "v1"),
        generator=AnswerGenerator(llm, 600),
        retrieval=SETTINGS.retrieval.model_copy(update={"abstain_threshold": threshold}),
        generation=SETTINGS.generation,
        tracing=SETTINGS.tracing,
        model_key="m1,m2",
        models={"stub": "stub-model"},
        trace_meta={"config_hash": "x"},
    )
    return QueryService(deps), retriever, llm


async def collect(svc, req=REQUEST):
    return [e async for e in svc.answer(req)]


async def test_event_order_meta_tokens_citations_done():
    svc, _, _ = service()
    events = await collect(svc)
    types = [e.type for e in events]
    assert types[0] == "meta" and types[-1] == "done" and types[-2] == "citations"
    assert set(types[1:-2]) == {"token"}
    assert "".join(e.text for e in events if e.type == "token") == "Customers have 30 days."
    assert [(c.doc_name, c.page) for c in events[-2].citations] == [("text.pdf", 3)]
    assert events[-1].provider == "stub" and not events[-1].cached


async def test_second_identical_query_is_a_cache_hit_and_skips_retrieval():
    svc, retriever, _ = service()
    await collect(svc)
    events = await collect(svc)
    assert retriever.calls == 1 and events[-1].cached
    assert [e.type for e in events] == ["meta", "token", "citations", "done"]


async def test_low_relevance_abstains_without_calling_the_llm():
    llm = StubLLM(reply())
    svc, _, _ = service(results=[chunk(score=0.01)], llm=llm, threshold=0.5)
    events = await collect(svc)
    assert [e.type for e in events] == ["meta", "abstain", "done"]
    assert events[1].reason == "low_relevance" and llm.calls == []


async def test_model_says_insufficient_context():
    svc, _, _ = service(llm=StubLLM(reply(sufficient=False)))
    events = await collect(svc)
    assert any(e.type == "abstain" and e.reason == "insufficient_context" for e in events)


async def test_all_providers_down_degrades_to_top_three_passages():
    results = [chunk(f"c{i}") for i in range(5)]
    svc, _, _ = service(results=results, llm=Unavailable())
    events = await collect(svc)
    assert [e.type for e in events] == ["meta", "degraded", "done"]
    assert [p.chunk.id for p in events[1].passages] == ["c0", "c1", "c2"]


async def test_follow_up_reports_the_rewritten_question():
    from docqa.domain.models import Turn

    history = [Turn(session_id="s1", role="user", content_scrubbed="refunds?")]
    svc, _, _ = service(repo=FakeRepo(history))
    events = await collect(svc)
    assert events[0].rewritten_question == "rewritten?"


async def test_turns_are_stored_scrubbed():
    repo = FakeRepo()
    svc, _, _ = service(repo=repo)
    req = REQUEST.model_copy(update={"question": "Refund window? Reply to jane.doe@example.com"})
    await collect(svc, req)
    assert [t.role for t in repo.turns] == ["user", "assistant"]
    assert "jane.doe@example.com" not in repo.turns[0].content_scrubbed
    assert repo.turns[0].created_at < repo.turns[1].created_at


async def test_check_rejects_rate_limit_and_unready_documents():
    svc, _, _ = service()
    for _ in range(10):
        await svc.check(REQUEST)
    with pytest.raises(RateLimitedError):
        await svc.check(REQUEST)

    class NotReady(FakeRepo):
        async def get(self, doc_id, owner_id):
            return None

    svc, _, _ = service(repo=NotReady())
    with pytest.raises(DocumentNotReadyError):
        await svc.check(REQUEST)
