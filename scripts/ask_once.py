"""Ask one question end to end through QueryService and print the events.

Usage:
    uv run python scripts/ask_once.py "What is the refund window?" --session-id demo
    uv run python scripts/ask_once.py "..." --session-id demo --pdf tests/fixtures/text.pdf

--pdf ingests the file first (skipped if that session already has it).
Needs MONGODB_URI and at least one LLM key (or Ollama).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import sys
from pathlib import Path

from docqa.api.deps import owner_id
from docqa.bootstrap import Container, build_container, init_database
from docqa.domain.errors import DocQAError
from docqa.domain.models import (
    Abstained,
    Citations,
    Completed,
    Degraded,
    Error,
    IngestionStatus,
    Meta,
    QueryRequest,
    Token,
)
from docqa.settings import load_settings


async def ingest_if_needed(container: Container, owner: str, pdf: Path) -> None:
    """Upload and ingest ``pdf`` unless the owner already has it."""
    sha = hashlib.sha256(pdf.read_bytes()).hexdigest()
    if any(d.sha256 == sha for d in await container.documents.list(owner)):
        return
    await container.documents.upload(owner, pdf.name, pdf.read_bytes())
    await container.worker.run_once()
    print(f"ingested {pdf.name}")


async def ask(question: str, session_id: str, pdf: Path | None) -> int:
    """Answer and print; return a process exit code."""
    container = build_container(load_settings())
    owner = owner_id(session_id)
    try:
        await init_database(container)
        if pdf:
            await ingest_if_needed(container, owner, pdf)
        docs = [
            d.id for d in await container.documents.list(owner) if d.status is IngestionStatus.READY
        ]
        if not docs:
            print("No ready documents for this session; pass --pdf.")
            return 1
        request = QueryRequest(
            owner_id=owner, session_id=session_id, question=question, doc_ids=docs
        )
        await container.query.check(request)
        async for event in container.query.answer(request):
            print_event(event)
        return 0
    except DocQAError as err:
        print(f"\nfailed: [{err.code}] {err.message}", file=sys.stderr)
        return 1
    finally:
        await container.close()


def print_event(event: object) -> None:
    """One line per event; tokens print inline as they stream."""
    if isinstance(event, Token):
        print(event.text, end="", flush=True)
    elif isinstance(event, Meta):
        print(f"[trace {event.trace_id}] rewritten={event.rewritten_question}")
    elif isinstance(event, Citations):
        print(f"\n\ncitations ({event.dropped} dropped):")
        for c in event.citations:
            print(f'  [{c.doc_name}, p. {c.page}] "{c.quote}"')
    elif isinstance(event, Abstained):
        print(f"\nABSTAIN ({event.reason}): I couldn't find this in your documents.")
    elif isinstance(event, Degraded):
        print(f"\nDEGRADED: no model available; top {len(event.passages)} passages returned.")
    elif isinstance(event, Error):
        print(f"\nERROR [{event.code}] {event.message}")
    elif isinstance(event, Completed):
        print(
            f"done: provider={event.provider} latency={event.latency_ms} ms cached={event.cached}"
        )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Ask DocQA one question.")
    parser.add_argument("question")
    parser.add_argument("--session-id", default="ask-once")
    parser.add_argument("--pdf", type=Path)
    args = parser.parse_args()
    sys.exit(asyncio.run(ask(args.question, args.session_id, args.pdf)))


if __name__ == "__main__":
    main()
