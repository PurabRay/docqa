"""IngestionService and IngestionWorker with fake ports (no database, no model)."""

from pathlib import Path

import pytest

from docqa.domain.errors import IndexSyncTimeoutError, RateLimitedError
from docqa.domain.models import IngestionStatus, IngestJob
from docqa.ingestion.sanitizer import compile_patterns
from docqa.ingestion.worker import IngestionWorker
from docqa.services.ingestion_service import IngestionService
from docqa.settings import load_settings

FIXTURES = Path(__file__).parents[1] / "fixtures"
SETTINGS = load_settings(env_file=None)


class FakeRepo:
    def __init__(self):
        self.statuses: list[tuple[IngestionStatus, str | None, int | None]] = []

    async def set_status(self, doc_id, status, error=None, chunk_count=None):
        self.statuses.append((status, error, chunk_count))


class FakeStore:
    def __init__(self, sync_error=None):
        self.upserted = []
        self.sync_error = sync_error

    async def upsert(self, chunks):
        self.upserted.extend(chunks)

    async def wait_until_searchable(self, doc_id, owner_id, expected, timeout_s=30):
        if self.sync_error:
            raise self.sync_error


class FakeEmbedder:
    model_id = "fake-model"
    dim = 4

    def embed_documents(self, texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


def service(store=None, repo=None):
    repo = repo or FakeRepo()
    store = store or FakeStore()
    svc = IngestionService(
        repo,
        store,
        FakeEmbedder(),
        SETTINGS.chunking,
        compile_patterns(SETTINGS.ingestion.injection_patterns),
        sync_timeout_s=1,
    )
    return svc, repo, store


def job(name="text.pdf"):
    return IngestJob(doc_id="d1", owner_id="u1", filename=name, path=str(FIXTURES / name))


async def test_statuses_go_extracting_indexing_syncing_ready():
    svc, repo, store = service()
    await svc.ingest(job())

    statuses = [status for status, _, _ in repo.statuses]
    assert statuses == ["extracting", "indexing", "syncing", "ready"]
    assert repo.statuses[-1][2] == len(store.upserted) > 0
    assert {c.embed_model for c in store.upserted} == {"fake-model"}


async def test_sync_timeout_marks_the_document_failed_with_the_message():
    svc, repo, _ = service(store=FakeStore(sync_error=IndexSyncTimeoutError("too slow")))
    await svc.ingest(job())
    assert repo.statuses[-1] == (IngestionStatus.FAILED, "too slow", None)


async def test_injection_page_produces_flagged_chunks():
    svc, _, store = service()
    await svc.ingest(job("injection.pdf"))
    assert [c.chunk.flagged_injection for c in store.upserted] == [False, True]


class RecordingRunner:
    def __init__(self, fail_first=False):
        self.seen = []
        self.fail_first = fail_first

    async def ingest(self, job):
        self.seen.append(job.doc_id)
        if self.fail_first and len(self.seen) == 1:
            raise RuntimeError("bug")

    async def mark_failed(self, job, message):
        self.seen.append(f"failed:{job.doc_id}")


async def test_worker_runs_jobs_in_order_and_survives_a_crash():
    runner = RecordingRunner(fail_first=True)
    worker = IngestionWorker(runner, queue_size=5)
    for doc_id in ["a", "b"]:
        worker.submit(job().model_copy(update={"doc_id": doc_id}))
    await worker.run_once()
    await worker.run_once()
    assert runner.seen == ["a", "failed:a", "b"]


def test_full_queue_is_rate_limited():
    worker = IngestionWorker(RecordingRunner(), queue_size=1)
    worker.submit(job())
    with pytest.raises(RateLimitedError):
        worker.submit(job())
