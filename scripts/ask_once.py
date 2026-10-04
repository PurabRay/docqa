"""Ask one question end to end: retrieve -> re-rank -> generate -> validate.

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
from docqa.bootstrap import Container, build_container, build_query_stack, init_database
from docqa.domain.errors import DocQAError
from docqa.domain.models import AccessFilter, IngestionStatus, RetrievedChunk
from docqa.generation.answer_generator import GeneratedAnswer, GeneratedToken
from docqa.generation.citation_validator import validate_citations
from docqa.generation.prompt_builder import build_messages
from docqa.retrieval.abstention import best_score, should_abstain
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
    settings = load_settings()
    container = build_container(settings)
    stack = build_query_stack(settings, container)
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
        candidates = await stack.retriever.retrieve(
            question, AccessFilter(owner_id=owner, doc_ids=docs)
        )
        top = await asyncio.to_thread(
            stack.reranker.rerank, question, candidates, settings.retrieval.top_k
        )
        print(f"retrieved {len(candidates)}, best re-rank score {best_score(top)}")
        if should_abstain(top, settings.retrieval.abstain_threshold):
            print("ABSTAIN: I couldn't find this in your documents.")
            return 0
        messages = build_messages(
            stack.answer_prompt,
            top,
            question,
            settings.limits.max_context_tokens,
            stack.count_tokens,
        )
        async for event in stack.generator.generate(messages):
            if isinstance(event, GeneratedToken):
                print(event.text, end="", flush=True)
            elif isinstance(event, GeneratedAnswer):
                print_result(event, top)
        return 0
    except DocQAError as err:
        print(f"\nfailed: [{err.code}] {err.message}", file=sys.stderr)
        return 1
    finally:
        await container.close()


def print_result(event: GeneratedAnswer, top: list[RetrievedChunk]) -> None:
    """Print provider, parse style and the citations that survived validation."""
    validated = validate_citations(event.answer, top)
    print(f"\n\nprovider={event.provider} style={event.style}")
    print(f"sufficient_context={validated.sufficient_context}")
    if event.warning:
        print(f"warning: {event.warning}")
    for c in validated.citations:
        print(f'  [{c.doc_name}, p. {c.page}] "{c.quote}"')
    print(f"verified={validated.verified} dropped={validated.dropped}")


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
