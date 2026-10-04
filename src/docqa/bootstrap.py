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
from docqa.adapters.vectorstore.mongo_store import MongoVectorStore
from docqa.generation.answer_generator import AnswerGenerator
from docqa.generation.prompt_registry import PromptRegistry, PromptTemplate
from docqa.guardrails.storage_guard import StorageGuard
from docqa.ingestion.sanitizer import compile_patterns
from docqa.ingestion.text_split import TokenCounter, token_counter
from docqa.ingestion.worker import IngestionWorker
from docqa.ports.embedder import DenseEmbedder
from docqa.ports.health import HealthProbe
from docqa.ports.llm import LLMClient
from docqa.ports.repository import DocumentRepository
from docqa.ports.reranker import Reranker
from docqa.retrieval.hybrid_retriever import HybridRetriever
from docqa.retrieval.query_rewriter import QueryRewriter
from docqa.services.document_service import DocumentService
from docqa.services.ingestion_service import IngestionService
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
    _worker_task: asyncio.Task[None] | None = field(default=None, repr=False)

    def start(self) -> None:
        """Start the background ingestion worker (needs a running event loop)."""
        self._worker_task = asyncio.create_task(self.worker.run_forever())

    async def close(self) -> None:
        """Stop the worker and release connections."""
        if self._worker_task is not None:
            self._worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker_task
        await self.mongo.close()


def build_container(settings: Settings) -> Container:
    """Build the container from settings. Opens no network connection yet.

    Raises:
        ConfigurationError: If MONGODB_URI (or the configured uri_env) is missing.
    """
    mongo = MongoConnection(settings.mongodb_uri(), settings.mongodb)
    repository = MongoDocumentRepository(mongo.db)
    store = MongoVectorStore(
        mongo.db,
        settings.mongodb,
        settings.retrieval,
        settings.chunks_collection,
        settings.embedding.dense_model,
        list(settings.search_indexes.values()),
    )
    embedder = FastEmbedDenseEmbedder(settings.embedding)
    ingestion = IngestionService(
        repository,
        store,
        embedder,
        settings.chunking,
        compile_patterns(settings.ingestion.injection_patterns),
        settings.mongodb.sync_timeout_s,
    )
    worker = IngestionWorker(ingestion, settings.ingestion.queue_size)
    guard = StorageGuard(
        MongoStorageMeter(mongo.db), settings.mongodb.storage_cap_mb, settings.mongodb.max_documents
    )
    documents = DocumentService(
        repository,
        store,
        guard,
        worker.submit,
        settings.limits,
        Path(settings.ingestion.upload_dir),
        settings.embedding.dense_model,
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
        documents=documents,
    )


@dataclass
class QueryStack:
    """The objects that answer a question (the query service arrives in prompt 7)."""

    retriever: HybridRetriever
    reranker: Reranker
    llm: LLMClient
    generator: AnswerGenerator
    rewriter: QueryRewriter
    answer_prompt: PromptTemplate
    count_tokens: TokenCounter


def build_llm_router(settings: Settings) -> FallbackLLMRouter:
    """One client per provider in llm.router order; providers missing their key are left out."""
    llm, clients = settings.llm, []
    for name in llm.router:
        provider = llm.providers[name]
        key = settings.optional_secret(provider.api_key_env) if provider.api_key_env else None
        if provider.api_key_env and key is None:
            log.warning(
                "llm.provider_skipped name=%s reason=%s not set", name, provider.api_key_env
            )
            continue
        clients.append(OpenAICompatibleLLM(name, provider, key))
    cb = llm.circuit_breaker
    breakers = {
        c.name: CircuitBreaker(c.name, cb.failures_to_open, cb.open_seconds) for c in clients
    }
    return FallbackLLMRouter(clients, breakers, llm.retry_on_minute_429, llm.retry_wait_s)


def build_query_stack(settings: Settings, container: Container) -> QueryStack:
    """Build retrieval, re-ranking and generation on top of the container."""
    retrieval, generation = settings.retrieval, settings.generation
    prompts = PromptRegistry(settings.prompts_dir)
    llm = build_llm_router(settings)
    return QueryStack(
        retriever=HybridRetriever(container.embedder, container.store, retrieval.fused_k),
        reranker=FastEmbedReranker(retrieval.rerank_model, retrieval.rerank_max_tokens),
        llm=llm,
        generator=AnswerGenerator(llm, generation.max_answer_tokens),
        rewriter=QueryRewriter(
            llm, prompts.get("rewrite", settings.prompts.rewrite), generation.max_rewrite_tokens
        ),
        answer_prompt=prompts.get("answer", settings.prompts.answer),
        count_tokens=token_counter(settings.chunking.tokenizer),
    )


async def init_database(container: Container) -> dict[str, IndexAction]:
    """Create every collection and index from config and wait for search indexes."""
    settings = container.settings
    return await initialize_database(
        container.mongo.db,
        chunks_collection=settings.chunks_collection,
        btree=settings.btree_indexes,
        search_indexes=list(settings.search_indexes.values()),
        timeout_s=settings.mongodb.index_ready_timeout_s,
        poll_interval_s=settings.mongodb.index_poll_interval_s,
    )
