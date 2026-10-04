"""Pick the abstention threshold from (question, is_answerable) records.

The threshold is the highest best-score cut that abstains on at most
``generation.max_false_abstention_rate`` of the answerable questions.

Usage:
    uv run python eval/calibrate_threshold.py records.jsonl --session-id S
Records: one JSON object per line, {"question": str, "is_answerable": bool}.
The documents searched are every ready document of that session's owner.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from pathlib import Path

from pydantic import BaseModel


class Calibration(BaseModel):
    """The chosen threshold and what it does on the records."""

    threshold: float
    false_abstention_rate: float  # answerable questions we would refuse
    abstention_rate_unanswerable: float  # unanswerable questions we would refuse


def choose_threshold(scored: list[tuple[float, bool]], max_false_rate: float) -> Calibration:
    """Choose the threshold from (best_score, is_answerable) pairs (abstain if score < t)."""
    answerable = sorted(score for score, ok in scored if ok)
    unanswerable = [score for score, ok in scored if not ok]
    if not answerable:
        raise ValueError("Need at least one answerable question to calibrate.")
    allowed_misses = math.floor(max_false_rate * len(answerable))
    threshold = answerable[allowed_misses]
    return Calibration(
        threshold=threshold,
        false_abstention_rate=sum(s < threshold for s in answerable) / len(answerable),
        abstention_rate_unanswerable=(
            sum(s < threshold for s in unanswerable) / len(unanswerable) if unanswerable else 0.0
        ),
    )


async def best_scores(
    records: list[dict[str, object]], session_id: str
) -> list[tuple[float, bool]]:
    """Run retrieve -> re-rank for each question and keep its best score."""
    from docqa.api.deps import owner_id
    from docqa.bootstrap import build_container, build_query_stack
    from docqa.domain.models import AccessFilter, IngestionStatus
    from docqa.retrieval.abstention import best_score
    from docqa.settings import load_settings

    settings = load_settings()
    container = build_container(settings)
    stack = build_query_stack(settings, container)
    owner = owner_id(session_id)
    docs = [
        d.id for d in await container.documents.list(owner) if d.status is IngestionStatus.READY
    ]
    access = AccessFilter(owner_id=owner, doc_ids=docs)
    scored = []
    for record in records:
        question = str(record["question"])
        candidates = await stack.retriever.retrieve(question, access)
        top = await asyncio.to_thread(
            stack.reranker.rerank, question, candidates, settings.retrieval.top_k
        )
        scored.append((best_score(top) or float("-inf"), bool(record["is_answerable"])))
    await container.close()
    return scored


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("records", type=Path)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--max-false-rate", type=float, default=0.10)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.records.read_text().splitlines() if line.strip()]
    result = choose_threshold(
        asyncio.run(best_scores(records, args.session_id)), args.max_false_rate
    )
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
