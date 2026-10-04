"""Composition root: the one place that builds concrete objects and wires them together.

Nothing else in DocQA constructs clients or adapters. The API lifespan calls
``build_container``, then ``Container.start`` and finally ``Container.close``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel

from docqa.adapters.cache.mongo_ttl_cache import MongoTTLCache
from docqa.adapters.cache.ttl_answer_cache import TTLAnswerCache
from docqa.adapters.embedding.fastembed_dense import FastEmbedDenseEmbedder
from docqa.adapters.llm.circuit_breaker import CircuitBreaker
from docqa.adapters.llm.openai_compatible import OpenAICompatibleLLM
from docqa.adapters.llm.router import FallbackLLMRouter
from docqa.adapters.mongo.client import MongoConnection
from docqa.adapters.mongo.health import MongoHealthProbe
from docqa.adapters.mongo.indexes import IndexAction, initialize_database
from docqa.adapters.mongo.storage_meter import MongoStorageMeter
from docqa.adapters.rerank.cross_encoder import FastEmbedReranker
from docqa.adapters.storage.mongo_repository import MongoDocumentRepository
from docqa.adapters.tracing.langfuse_tracer import LangfuseTracer
from docqa.adapters.tracing.noop_tracer import NoopTracer
from docqa.adapters.vectorstore.mongo_store import MongoVectorStore
from docqa.generation.answer_generator import AnswerGenerator
from docqa.generation.prompt_builder import PromptBuilder
from docqa.generation.prompt_registry import PromptRegistry
from docqa.generation.schemas import LLMAnswer
from docqa.guardrails.input_guard import InputGuard
from docqa.guardrails.rate_limiter import RateLimiter
from docqa.guardrails.storage_guard import StorageGuard
from docqa.ingestion.sanitizer import compile_patterns
from docqa.ingestion.text_split import token_counter
from docqa.ingestion.worker import IngestionWorker
from docqa.ports.cache import AnswerCache
from docqa.ports.embedder import DenseEmbedder
from docqa.ports.health import HealthProbe
from docqa.ports.llm import LLMClient
from docqa.ports.repository import DocumentRepository
from docqa.ports.tracer import Tracer
from docqa.retrieval.hybrid_retriever import HybridRetriever
from docqa.retrieval.query_rewriter import QueryRewriter
from docqa.services.document_service import DocumentService
from docqa.services.feedback_service import FeedbackService
from docqa.services.ingestion_service import IngestionService
from docqa.services.query_service import QueryDeps, QueryService
from docqa.settings import Settings

log = logging.getLogger(__name__)


@dataclass
class Container:
    """Every long-lived object the app needs."""

    settings: Settings
    mongo: MongoConnection
    repository: DocumentRepository
    health: HealthProbe
    embedder: DenseEmbedder
    store: MongoVectorStore
    ingestion: IngestionService
    worker: IngestionWorker
    documents: DocumentService
    cache: AnswerCache
    tracer: Tracer
    breakers: dict[str, CircuitBreaker]
    query_deps: QueryDeps
    query: QueryService
    feedback: FeedbackService
    _worker_task: asyncio.Task[None] | None = field(default=None, repr=False)

    def start(self) -> None:
        """Start the background ingestion worker (needs a running event loop)."""
        self._worker_task = asyncio.create_task(self.worker.run_forever())

    def llm_states(self) -> dict[str, str]:
        """Circuit-breaker state per provider, for /health."""
        return {name: breaker.state for name, breaker in self.breakers.items()}

    async def close(self) -> None:
        """Stop the worker, flush traces and release connections."""
        if self._worker_task is not None:
            self._worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker_task
        flush = getattr(self.tracer, "flush", None)
        if flush is not None:
            flush()
        await self.mongo.close()


def build_container(
    settings: Settings,
    *,
    answer_llm: LLMClient | None = None,
    text_llm: LLMClient | None = None,
    tracer: Tracer | None = None,
) -> Container:
    """Build everything from settings. Opens no network connection yet.

    ``answer_llm``, ``text_llm`` and ``tracer`` replace the configured ones (tests, load
    tests with StubLLM).

    Raises:
        ConfigurationError: If MONGODB_URI (or the configured uri_env) is missing.
    """
    mongo = MongoConnection(settings.mongodb_uri(), settings.mongodb)
    repository = MongoDocumentRepository(mongo.db)
    store = MongoVectorStore(
        mongo.db,
        settings.chunks_collection,
        settings.retrieval_cfg(),
        settings.embedding.dense_model,
    )
    embedder = FastEmbedDenseEmbedder(settings.embedding)
    patterns = compile_patterns(settings.ingestion.injection_patterns)
    ingestion = IngestionService(
        repository, store, embedder, settings.chunking, patterns, settings.mongodb.sync_timeout_s
    )
    worker = IngestionWorker(ingestion, settings.ingestion.queue_size)
    cache = build_cache(settings, mongo)
    tracer = tracer or build_tracer(settings)
    breakers = build_breakers(settings)
    deps = build_query_deps(
        settings, repository, store, embedder, cache, tracer, breakers, answer_llm, text_llm
    )
    index_names = [settings.mongodb.vector_index, settings.mongodb.text_index]
    return Container(
        settings=settings,
        mongo=mongo,
        repository=repository,
        health=MongoHealthProbe(mongo, settings.chunks_collection, index_names),
        embedder=embedder,
        store=store,
        ingestion=ingestion,
        worker=worker,
        documents=build_documents(settings, mongo, repository, store, worker, cache),
        cache=cache,
        tracer=tracer,
        breakers=breakers,
        query_deps=deps,
        query=QueryService(deps),
        feedback=FeedbackService(repository, tracer),
    )


def build_documents(
    settings: Settings,
    mongo: MongoConnection,
    repository: DocumentRepository,
    store: MongoVectorStore,
    worker: IngestionWorker,
    cache: AnswerCache,
) -> DocumentService:
    """Upload / list / get / delete, with the storage guard in front of uploads."""
    m = settings.mongodb
    guard = StorageGuard(MongoStorageMeter(mongo.db), m.storage_cap_mb, m.max_documents)
    return DocumentService(
        repository,
        store,
        guard,
        worker.submit,
        settings.limits,
        Path(settings.ingestion.upload_dir),
        settings.embedding.dense_model,
        cache,
    )


def build_query_deps(
    settings: Settings,
    repository: DocumentRepository,
    store: MongoVectorStore,
    embedder: DenseEmbedder,
    cache: AnswerCache,
    tracer: Tracer,
    breakers: dict[str, CircuitBreaker],
    answer_llm: LLMClient | None,
    text_llm: LLMClient | None,
) -> QueryDeps:
    """Everything the query use case needs."""
    prompts = PromptRegistry(settings.prompts_dir)
    answer_llm = answer_llm or build_llm_router(settings, breakers, response_schema=LLMAnswer)
    text_llm = text_llm or build_llm_router(settings, breakers)
    gen, retrieval, llm = settings.generation, settings.retrieval, settings.llm
    models = {name: llm.providers[name].model for name in llm.router}
    return QueryDeps(
        repo=repository,
        cache=cache,
        tracer=tracer,
        input_guard=InputGuard(
            settings.limits.max_question_chars,
            compile_patterns(settings.ingestion.injection_patterns),
        ),
        rate_limiter=RateLimiter(settings.limits.questions_per_minute),
        rewriter=QueryRewriter(
            text_llm, prompts.get("rewrite", settings.prompts.rewrite), gen.max_rewrite_tokens
        ),
        retriever=HybridRetriever(embedder, store, retrieval.fused_k, tracer),
        reranker=FastEmbedReranker(retrieval.rerank_model, retrieval.rerank_max_tokens),
        prompt_builder=PromptBuilder(
            settings.limits.max_context_tokens, token_counter(settings.chunking.tokenizer)
        ),
        template=prompts.get("answer", settings.prompts.answer),
        generator=AnswerGenerator(answer_llm, gen.max_answer_tokens),
        retrieval=retrieval,
        generation=gen,
        tracing=settings.tracing,
        model_key=",".join(models[name] for name in llm.router),
        models=models,
        trace_meta={"config_hash": settings.config_hash(), "fusion": retrieval.fusion},
    )


def build_cache(settings: Settings, mongo: MongoConnection) -> AnswerCache:
    """cache.backend: memory (one process) or mongo (several replicas)."""
    if settings.cache.backend == "mongo":
        return MongoTTLCache(mongo.db, settings.cache.ttl_seconds)
    return TTLAnswerCache(settings.cache.ttl_seconds)


def build_tracer(settings: Settings) -> Tracer:
    """Langfuse when configured and keys are set; otherwise the no-op tracer."""
    public = settings.optional_secret("LANGFUSE_PUBLIC_KEY")
    secret = settings.optional_secret("LANGFUSE_SECRET_KEY")
    if settings.tracing.backend != "langfuse" or not (public and secret):
        if settings.tracing.backend == "langfuse":
            log.warning("tracing.disabled reason=LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY not set")
        return NoopTracer()
    from langfuse import Langfuse  # imported only when tracing is on

    host = settings.optional_secret("LANGFUSE_HOST")
    return LangfuseTracer(Langfuse(public_key=public, secret_key=secret, host=host))


def build_breakers(settings: Settings) -> dict[str, CircuitBreaker]:
    """One breaker per configured provider, shared by every router."""
    cb = settings.llm.circuit_breaker
    return {
        name: CircuitBreaker(name, cb.failures_to_open, cb.open_seconds)
        for name in settings.llm.router
    }


def build_llm_router(
    settings: Settings,
    breakers: dict[str, CircuitBreaker],
    response_schema: type[BaseModel] | None = None,
) -> FallbackLLMRouter:
    """One client per provider in llm.router order; providers missing their key are left out.

    ``response_schema`` is what streamed replies must follow (LLMAnswer for answers).
    """
    llm, clients = settings.llm, []
    for name in llm.router:
        provider = llm.providers[name]
        key = settings.optional_secret(provider.api_key_env) if provider.api_key_env else None
        if provider.api_key_env and key is None:
            log.warning(
                "llm.provider_skipped name=%s reason=%s not set", name, provider.api_key_env
            )
            continue
        clients.append(OpenAICompatibleLLM(name, provider, key, response_schema))
    return FallbackLLMRouter(clients, breakers, llm.retry_on_minute_429, llm.retry_wait_s)


async def init_database(container: Container) -> dict[str, IndexAction]:
    """Create every collection and index from config and wait for search indexes."""
    settings = container.settings
    actions = await initialize_database(
        container.mongo.db,
        chunks_collection=settings.chunks_collection,
        btree=settings.btree_indexes,
        search_indexes=list(settings.search_indexes.values()),
        timeout_s=settings.mongodb.index_ready_timeout_s,
        poll_interval_s=settings.mongodb.index_poll_interval_s,
    )
    if isinstance(container.cache, MongoTTLCache):
        await container.cache.ensure_indexes()
    return actions
