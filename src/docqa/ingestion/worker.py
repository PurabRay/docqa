"""In-process ingestion queue, one document at a time.

One at a time keeps writes well under the free cluster's 100 ops/s.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol

from docqa.domain.errors import DocQAError, RateLimitedError
from docqa.domain.models import IngestJob

log = logging.getLogger(__name__)

CRASH_MESSAGE = "Ingestion failed unexpectedly; please upload the file again."


class JobRunner(Protocol):
    """Runs one job (IngestionService)."""

    async def ingest(self, job: IngestJob) -> None:
        """Ingest one document."""
        ...

    async def mark_failed(self, job: IngestJob, message: str) -> None:
        """Set the document's status to failed with ``message``."""
        ...


class IngestionWorker:
    """An asyncio queue drained by one background task."""

    def __init__(self, runner: JobRunner, queue_size: int) -> None:
        self._runner = runner
        self._queue: asyncio.Queue[IngestJob] = asyncio.Queue(maxsize=queue_size)

    def submit(self, job: IngestJob) -> None:
        """Queue a job.

        Raises:
            RateLimitedError: If the queue is full.
        """
        try:
            self._queue.put_nowait(job)
        except asyncio.QueueFull as err:
            raise RateLimitedError("Too many documents are waiting; try again soon.") from err

    async def run_forever(self) -> None:
        """Process jobs until cancelled."""
        while True:
            await self.run_once()

    async def run_once(self) -> None:
        """Wait for one job and run it."""
        job = await self._queue.get()
        try:
            await self._runner.ingest(job)
        except Exception as err:
            # Process boundary: the runner already marks DocQAErrors as failed. Anything
            # else is a bug. Mark the document failed (never leave it stuck in
            # "extracting"), log only the error type (no document text), keep going.
            log.error("ingest.crashed doc_id=%s error=%s", job.doc_id, type(err).__name__)
            await self._mark_failed(job)
        finally:
            self._queue.task_done()

    async def _mark_failed(self, job: IngestJob) -> None:
        try:
            await self._runner.mark_failed(job, CRASH_MESSAGE)
        except DocQAError as err:  # e.g. the database is down too
            log.error("ingest.mark_failed_failed doc_id=%s code=%s", job.doc_id, err.code)

    async def join(self) -> None:
        """Wait until every queued job is done (used by tests)."""
        await self._queue.join()
