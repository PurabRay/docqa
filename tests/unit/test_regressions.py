"""Regression tests for the fixes of 2026-10-04 (stuck statuses, offline tokens, strict quotes)."""

from pathlib import Path

import pytest
import requests

from docqa.domain.errors import DatabaseUnavailableError, InvalidUploadError
from docqa.domain.models import Chunk, IngestionStatus, IngestJob, RetrievedChunk
from docqa.generation.citation_validator import validate_citations
from docqa.generation.schemas import CitedClaim, LLMAnswer
from docqa.ingestion import extractor
from docqa.ingestion.text_split import token_counter
from docqa.ingestion.worker import IngestionWorker
from tests.unit.test_ingestion_flow import job, service

FIXTURES = Path(__file__).parents[1] / "fixtures"


def broken_pdf_library(*args, **kwargs):
    raise IndexError("list index out of range")  # what PyMuPDF code paths raise on odd files


def test_pdf_library_errors_become_invalid_upload(monkeypatch):
    monkeypatch.setattr(extractor.pymupdf4llm, "to_markdown", broken_pdf_library)
    with pytest.raises(InvalidUploadError):
        extractor.extract_pages(FIXTURES / "text.pdf", "d1")


async def test_extraction_crash_marks_the_document_failed(monkeypatch):
    monkeypatch.setattr(extractor.pymupdf4llm, "to_markdown", broken_pdf_library)
    svc, repo, _ = service()
    await svc.ingest(job())
    assert repo.statuses[-1][0] is IngestionStatus.FAILED
    assert "could not be read" in repo.statuses[-1][1]


class CrashingRunner:
    def __init__(self, fail_marking=False):
        self.failed: list[tuple[str, str]] = []
        self.fail_marking = fail_marking

    async def ingest(self, job: IngestJob) -> None:
        raise RuntimeError("bug")

    async def mark_failed(self, job: IngestJob, message: str) -> None:
        if self.fail_marking:
            raise DatabaseUnavailableError()
        self.failed.append((job.doc_id, message))


async def test_worker_marks_a_crashed_job_failed():
    runner = CrashingRunner()
    worker = IngestionWorker(runner, queue_size=2)
    worker.submit(job())
    await worker.run_once()
    from docqa.ingestion.worker import CRASH_MESSAGE

    assert runner.failed == [("d1", CRASH_MESSAGE)]


async def test_worker_survives_when_marking_failed_also_fails():
    worker = IngestionWorker(CrashingRunner(fail_marking=True), queue_size=2)
    worker.submit(job())
    await worker.run_once()  # must not raise


def test_token_counting_needs_no_network(monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("token counting tried to use the network")

    monkeypatch.setattr(requests, "get", no_network)
    token_counter.cache_clear()
    assert token_counter("cl100k_base")("Under clause 14.3(b), employees accrue 25 days.") == 15


def test_quotes_must_match_markdown_exactly():
    chunk = Chunk(
        id="c1",
        doc_id="d",
        owner_id="u",
        filename="f.pdf",
        text="**Refunds** take 30 days.",
        page_start=1,
        page_end=1,
        content_hash="h",
    )
    answer = LLMAnswer(
        answer="...",
        sufficient_context=True,
        citations=[CitedClaim(chunk_id="c1", quote="Refunds take 30 days")],
    )
    result = validate_citations(answer, [RetrievedChunk(chunk=chunk)])
    assert result.dropped == 1 and not result.verified
