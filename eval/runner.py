"""Run golden questions through QueryService and record what each step produced."""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from docqa.api.deps import owner_id
from docqa.bootstrap import Container
from docqa.domain.models import AccessFilter, IngestionStatus, QueryRequest, RetrievedChunk
from docqa.services.query_service import QueryService
from eval.metrics.retrieval import ChunkRef
from eval.schema import GoldenRecord


class QuestionResult(BaseModel):
    """Everything the metrics need about one question."""

    record: GoldenRecord
    answer: str = ""
    abstained: bool = False
    degraded: bool = False
    citations: list[tuple[str, int, str]] = []  # (doc_name, page, quote)
    top: list[ChunkRef] = []  # chunks sent to the LLM
    fused: list[ChunkRef] = []  # fused list before re-ranking
    contexts: list[str] = []
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    provider: str | None = None


def ref(item: RetrievedChunk) -> ChunkRef:
    """A chunk's document and page range."""
    return ChunkRef(
        doc_name=item.chunk.filename, page_start=item.chunk.page_start, page_end=item.chunk.page_end
    )


class Recorder:
    """Wraps the retriever and re-ranker to keep the latest fused and top lists."""

    def __init__(self, retriever: Any, reranker: Any) -> None:
        self._retriever, self._reranker = retriever, reranker
        self.fused: list[RetrievedChunk] = []
        self.top: list[RetrievedChunk] = []

    async def retrieve(self, question: str, access: AccessFilter) -> list[RetrievedChunk]:
        """Retrieve and remember the fused list."""
        self.fused = await self._retriever.retrieve(question, access)
        return self.fused

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        """Re-rank and remember the chunks sent to the LLM."""
        self.top = self._reranker.rerank(query, chunks, top_k)
        return self.top


async def ingest_corpus(container: Container, corpus: Path, session_id: str) -> list[str]:
    """Upload every PDF in ``corpus`` once (skips files already ingested); return doc ids."""
    owner = owner_id(session_id)
    known = {d.sha256: d for d in await container.documents.list(owner)}
    for pdf in sorted(corpus.glob("*.pdf")):
        data = pdf.read_bytes()
        if hashlib.sha256(data).hexdigest() not in known:
            await container.documents.upload(owner, pdf.name, data)
            await container.worker.run_once()
    return [
        d.id for d in await container.documents.list(owner) if d.status is IngestionStatus.READY
    ]


async def run_questions(
    container: Container, records: list[GoldenRecord], doc_ids: list[str], session_id: str
) -> list[QuestionResult]:
    """Ask every record (follow-ups replay their history first in a session of their own)."""
    recorder = Recorder(container.query_deps.retriever, container.query_deps.reranker)
    service = QueryService(
        dataclasses.replace(container.query_deps, retriever=recorder, reranker=recorder)
    )
    owner, results = owner_id(session_id), []
    for record in records:
        session = f"{session_id}-{record.id}"
        for earlier in record.history:
            await _ask(
                service,
                QueryRequest(owner_id=owner, session_id=session, question=earlier, doc_ids=doc_ids),
            )
        request = QueryRequest(
            owner_id=owner, session_id=session, question=record.question, doc_ids=doc_ids
        )
        result = await _ask(service, request, QuestionResult(record=record))
        result.fused, result.top = [ref(c) for c in recorder.fused], [ref(c) for c in recorder.top]
        result.contexts = [c.chunk.text for c in recorder.top]
        results.append(result)
    return results


async def _ask(
    service: QueryService, request: QueryRequest, result: QuestionResult | None = None
) -> QuestionResult:
    result = result or QuestionResult(
        record=GoldenRecord(
            id="history", question=request.question, gold_answer="", category="single", split="dev"
        )
    )
    async for event in service.answer(request):
        if event.type == "token":
            result.answer += event.text
        elif event.type == "citations":
            result.citations = [(c.doc_name, c.page, c.quote) for c in event.citations]
        elif event.type == "abstain":
            result.abstained = True
        elif event.type == "degraded":
            result.degraded = True
        elif event.type == "done":
            result.latency_ms, result.provider = event.latency_ms, event.provider
            result.input_tokens, result.output_tokens = event.input_tokens, event.output_tokens
    return result
