"""Typed shape of config/*.yaml. Pure data: loading happens in ``docqa.settings``.

Every section forbids unknown keys, so a typo in a YAML file fails at startup
instead of silently falling back to a default.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Section(BaseModel):
    """Base for config sections: immutable, no unknown keys."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ProviderConfig(Section):
    """One OpenAI-compatible LLM endpoint."""

    base_url: str
    model: str
    api_key_env: str | None = None  # None for keyless providers such as Ollama
    timeout_s: float = Field(gt=0)
    json_schema: bool  # supports response_format json_schema; else JSON mode
    json_mode: bool = True  # supports response_format json_object; else plain text
    extra_body: dict[str, Any] = {}


class CircuitBreakerConfig(Section):
    """When to stop calling a failing provider, and for how long."""

    failures_to_open: int = Field(ge=1)
    open_seconds: float = Field(gt=0)


class LLMConfig(Section):
    """Providers and the order the router tries them in."""

    router: list[str] = Field(min_length=1)
    providers: dict[str, ProviderConfig]
    circuit_breaker: CircuitBreakerConfig
    retry_on_minute_429: int = Field(ge=0)
    retry_wait_s: float = Field(ge=0)

    @model_validator(mode="after")
    def _router_names_exist(self) -> LLMConfig:
        missing = [name for name in self.router if name not in self.providers]
        if missing:
            raise ValueError(f"llm.router names unknown providers: {missing}")
        return self


class MongoConfig(Section):
    """Database connection, collection names and free-cluster budgets."""

    uri_env: str
    database: str
    chunks_collection: str
    vector_index: str
    text_index: str
    bulk_batch: int = Field(ge=1)
    sync_timeout_s: float = Field(gt=0)
    sync_poll_interval_s: float = Field(gt=0)
    max_pool_size: int = Field(ge=1)
    server_selection_timeout_ms: int = Field(ge=1)
    index_ready_timeout_s: float = Field(gt=0)
    index_poll_interval_s: float = Field(gt=0)
    storage_cap_mb: int = Field(ge=1)
    max_documents: int = Field(ge=1)


class EmbeddingConfig(Section):
    """The pinned dense embedding model."""

    dense_model: str
    dim: int = Field(ge=1)
    storage: Literal["float32_binary"]
    batch_size: int = Field(ge=1)
    document_prefix: str
    query_prefix: str


class ChunkingConfig(Section):
    """How documents are split into chunks."""

    max_tokens: int = Field(ge=1)
    overlap_tokens: int = Field(ge=0)
    respect_headings: bool
    tokenizer: str


class IngestionConfig(Section):
    """Upload storage, queue size and injection patterns."""

    upload_dir: str
    queue_size: int = Field(ge=1)
    injection_patterns: list[str]


class FusionWeights(Section):
    """Per-branch weights for $rankFusion."""

    vector: float = Field(ge=0)
    text: float = Field(ge=0)


class RetrievalConfig(Section):
    """Hybrid search, re-ranking and abstention parameters."""

    fusion: Literal["server", "app"]
    vector_k: int = Field(ge=1)
    text_k: int = Field(ge=1)
    num_candidates: int = Field(ge=1)
    fused_k: int = Field(ge=1)
    weights: FusionWeights
    rerank_model: str
    top_k: int = Field(ge=1)
    abstain_threshold: float
    rerank_max_tokens: int = Field(ge=1)


class GenerationConfig(Section):
    """Answer and rewrite budgets, and the abstention calibration target."""

    max_answer_tokens: int = Field(ge=1)
    max_rewrite_tokens: int = Field(ge=1)
    max_false_abstention_rate: float = Field(ge=0, le=1)


class PromptsConfig(Section):
    """Active prompt versions (prompts/<name>/<version>.yaml)."""

    answer: str
    rewrite: str


class LimitsConfig(Section):
    """Input and cost guardrails."""

    max_pdf_mb: int = Field(ge=1)
    max_pages: int = Field(ge=1)
    min_chars_per_page: int = Field(ge=0)
    max_question_chars: int = Field(ge=1)
    max_context_tokens: int = Field(ge=1)
    questions_per_minute: int = Field(ge=1)


class CacheConfig(Section):
    """Answer cache backend."""

    backend: Literal["memory", "mongo"]
    ttl_seconds: int = Field(ge=1)


class TracingConfig(Section):
    """Tracing backend and online-judge sampling."""

    backend: Literal["langfuse", "noop"]
    sample_judge_rate: float = Field(ge=0, le=1)


class AppConfig(Section):
    """Everything in config/base.yaml after the profile is merged in."""

    llm: LLMConfig
    mongodb: MongoConfig
    embedding: EmbeddingConfig
    chunking: ChunkingConfig
    ingestion: IngestionConfig
    retrieval: RetrievalConfig
    generation: GenerationConfig
    prompts: PromptsConfig
    limits: LimitsConfig
    cache: CacheConfig
    tracing: TracingConfig


class BTreeIndex(Section):
    """One B-tree index from config/indexes/btree.yaml."""

    keys: dict[str, Literal[1, -1]] = Field(min_length=1)
    unique: bool = False


class SearchIndex(Section):
    """One Atlas Search or Vector Search index from config/indexes/*.json."""

    name: str
    type: Literal["search", "vectorSearch"]
    definition: dict[str, Any]


class RetrievalCfg(Section):
    """What the vector store and its pipelines need, gathered from several sections.

    This is the ``cfg: RetrievalCfg`` of hybrid_pipeline and MongoVectorStore in
    docs/DESIGN.md. Build it with Settings.retrieval_cfg().
    """

    fusion: Literal["server", "app"]
    vector_index: str
    text_index: str
    num_candidates: int
    vector_k: int
    text_k: int
    fused_k: int
    w_vector: float
    w_text: float
    bulk_batch: int
    sync_poll_interval_s: float
    index_ready_timeout_s: float
    index_poll_interval_s: float
    search_indexes: list[SearchIndex]
