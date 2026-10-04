"""Query use case (docs/DESIGN.md, "Query use case"), one named step per line.

guard -> cache -> last turns -> rewrite -> retrieve -> re-rank -> abstain? -> prompt ->
stream -> validate -> cache -> append turns -> done. Every step is a span.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import timedelta

from docqa.config_schema import GenerationConfig, RetrievalConfig, TracingConfig
from docqa.domain.errors import AllProvidersUnavailableError, DocQAError, DocumentNotReadyError
from docqa.domain.models import (
    Abstained,
    AccessFilter,
    Answer,
    Citations,
    Completed,
    Degraded,
    Error,
    IngestionStatus,
    Meta,
    QueryEvent,
    QueryRequest,
    RetrievedChunk,
    Token,
    Turn,
)
from docqa.generation.answer_generator import AnswerGenerator, Final
from docqa.generation.citation_validator import ValidatedAnswer, validate_citations
from docqa.generation.prompt_builder import PromptBuilder
from docqa.generation.prompt_registry import PromptTemplate
from docqa.guardrails.input_guard import InputGuard
from docqa.guardrails.pii import scrub
from docqa.guardrails.rate_limiter import RateLimiter
from docqa.ports.cache import AnswerCache
from docqa.ports.llm import Message
from docqa.ports.repository import DocumentRepository
from docqa.ports.reranker import Reranker
from docqa.ports.tracer import SpanHandle, Tracer
from docqa.retrieval.abstention import best_score, should_abstain
from docqa.retrieval.hybrid_retriever import HybridRetriever
from docqa.retrieval.query_rewriter import QueryRewriter

WHITESPACE = re.compile(r"\s+")
TOKENS_PER_MILLION = 1_000_000
TURN_SPACING = timedelta(milliseconds=1)


def hash_id(value: str) -> str:
    """Short one-way id for traces (never the raw owner id)."""
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def cache_key(req: QueryRequest, prompt_version: str, model: str) -> str:
    """hash(owner_id, sorted doc_ids, normalised question, prompt_version, model) (PRD)."""
    question = WHITESPACE.sub(" ", req.question.strip().lower())
    parts = [req.owner_id, ",".join(sorted(req.doc_ids)), question, prompt_version, model]
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()


@dataclass
class QueryDeps:
    """Everything QueryService needs; built by bootstrap.py."""

    repo: DocumentRepository
    cache: AnswerCache
    tracer: Tracer
    input_guard: InputGuard
    rate_limiter: RateLimiter
    rewriter: QueryRewriter
    retriever: HybridRetriever
    reranker: Reranker
    prompt_builder: PromptBuilder
    template: PromptTemplate
    generator: AnswerGenerator
    retrieval: RetrievalConfig
    generation: GenerationConfig
    tracing: TracingConfig
    model_key: str  # the router's model order (cache key "model", decision of 2026-10-04)
    models: dict[str, str]  # provider name -> model id, for LLM spans
    trace_meta: dict[str, str]  # config_hash, fusion mode, ...


class QueryService:
    """Answers one question as a stream of QueryEvents."""

    def __init__(self, deps: QueryDeps) -> None:
        self.d = deps

    async def check(self, req: QueryRequest) -> None:
        """Checks that must fail BEFORE streaming starts (HTTP 400/429). Call first.

        Raises:
            InputRejectedError, RateLimitedError, DocumentNotReadyError.
        """
        self.d.input_guard.check(req.question)
        self.d.rate_limiter.check(req.session_id)
        for doc_id in req.doc_ids:
            doc = await self.d.repo.get(doc_id, req.owner_id)
            if doc is None or doc.status is not IngestionStatus.READY:
                raise DocumentNotReadyError()

    async def answer(self, req: QueryRequest) -> AsyncIterator[QueryEvent]:
        """Stream meta, tokens, citations (or abstain/degraded), then done."""
        start = time.perf_counter()
        user = hash_id(req.owner_id)
        with self.d.tracer.trace(
            "query", user=user, prompt=self.d.template.version, **self.d.trace_meta
        ) as trace:
            outcome = "answer"  # what scripts/check_alerts.py counts: answer|abstain|degraded|error
            try:
                async for event in self._run(req, trace.trace_id, start, trace):
                    if event.type in ("abstain", "degraded"):
                        outcome = event.type
                    elif isinstance(event, Completed) and event.cached:
                        outcome = "cached"
                    yield event
            except DocQAError as err:
                outcome = "error"
                trace.set(error=err.code)
                yield Error(code=err.code, message=err.message)
            trace.set(outcome=outcome, latency_ms=_ms(start))

    async def _run(
        self, req: QueryRequest, trace_id: str, start: float, trace: SpanHandle
    ) -> AsyncIterator[QueryEvent]:
        trace.set(injection_flag=self.d.input_guard.check(req.question))
        key = cache_key(req, self.d.template.version, self.d.model_key)
        if cached := await self.d.cache.get(key):
            yield Meta(trace_id=trace_id)
            yield Token(text=cached.text)
            yield Citations(citations=cached.citations)
            yield Completed(
                answer=cached, provider=cached.provider, latency_ms=_ms(start), cached=True
            )
            return
        history = await self.d.repo.last_turns(req.session_id, n=self.d.generation.history_turns)
        with self.d.tracer.span("rewrite", turns=len(history)):
            question = await self.d.rewriter.rewrite(req.question, history)
        yield Meta(
            trace_id=trace_id, rewritten_question=question if question != req.question else None
        )
        top = await self._retrieve(question, req)
        if should_abstain(top, self.d.retrieval.abstain_threshold):
            yield Abstained(reason="low_relevance")
            await self._append_turns(req, "[abstained: low relevance]", trace_id)
            yield Completed(latency_ms=_ms(start))
            return
        messages = self.d.prompt_builder.build(self.d.template, top, question)
        final = None
        try:
            async for generated in self._generate(messages):
                if isinstance(generated, Token):
                    yield generated
                else:
                    final = generated
        except AllProvidersUnavailableError:
            yield Degraded(passages=top[: self.d.generation.degraded_passages])
            yield Completed(latency_ms=_ms(start))
            return
        assert final is not None  # the generator always ends with exactly one Final
        validated = validate_citations(final.parsed, top)
        async for event in self._finish(req, key, final, validated, trace_id, start, trace):
            yield event

    async def _retrieve(self, question: str, req: QueryRequest) -> list[RetrievedChunk]:
        access = AccessFilter(owner_id=req.owner_id, doc_ids=req.doc_ids)
        candidates = await self.d.retriever.retrieve(question, access)
        with self.d.tracer.span("rerank", candidates=len(candidates)) as span:
            top = await asyncio.to_thread(
                self.d.reranker.rerank, question, candidates, self.d.retrieval.top_k
            )
            span.set(top_score=best_score(top))
        return top

    async def _generate(self, messages: list[Message]) -> AsyncIterator[Token | Final]:
        with self.d.tracer.span("generate", prompt=self.d.template.version) as span:
            async for event in self.d.generator.generate(messages):
                if isinstance(event, Final):
                    span.set(**self._usage(event))
                yield event

    def _usage(self, final: Final) -> dict[str, str | int | float]:
        t = self.d.tracing
        cost = (
            final.input_tokens * t.cost_per_million_input
            + final.output_tokens * t.cost_per_million_output
        )
        return {
            "provider": final.provider,
            "model": self.d.models.get(final.provider, final.provider),
            "input_tokens": final.input_tokens,
            "output_tokens": final.output_tokens,
            "cost_usd": cost / TOKENS_PER_MILLION,
            "parse_style": final.style,
        }

    async def _finish(
        self,
        req: QueryRequest,
        key: str,
        final: Final,
        validated: ValidatedAnswer,
        trace_id: str,
        start: float,
        trace: SpanHandle,
    ) -> AsyncIterator[QueryEvent]:
        yield Citations(citations=validated.citations, dropped=validated.dropped)
        if not validated.sufficient_context:
            yield Abstained(reason="insufficient_context")
        answer = Answer(
            text=validated.text,
            citations=validated.citations,
            abstained=not validated.sufficient_context,
            provider=final.provider,
            prompt_version=self.d.template.version,
        )
        if validated.sufficient_context:
            await self.d.cache.set(key, answer, req.doc_ids)
        await self._append_turns(req, answer.text, trace_id)
        # For scripts/online_judge.py: the cited quotes, PII-scrubbed like everything traced.
        quotes = await asyncio.to_thread(scrub, "\n".join(c.quote for c in validated.citations))
        trace.set(quotes=quotes, citations=len(validated.citations), dropped=validated.dropped)
        yield Completed(
            answer=answer,
            provider=final.provider,
            latency_ms=_ms(start),
            input_tokens=final.input_tokens,
            output_tokens=final.output_tokens,
        )

    async def _append_turns(self, req: QueryRequest, answer_text: str, trace_id: str) -> None:
        """Store both turns, PII-scrubbed (scrubbing runs in a thread: Presidio is CPU-bound)."""
        question, answer = await asyncio.gather(
            asyncio.to_thread(scrub, req.question), asyncio.to_thread(scrub, answer_text)
        )
        asked = Turn(
            session_id=req.session_id, role="user", content_scrubbed=question, trace_id=trace_id
        )
        # BSON keeps milliseconds only: space the two turns so their order is stable.
        answered = Turn(
            session_id=req.session_id,
            role="assistant",
            content_scrubbed=answer,
            trace_id=trace_id,
            created_at=asked.created_at + TURN_SPACING,
        )
        await self.d.repo.append_turn(asked)
        await self.d.repo.append_turn(answered)


def _ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)
