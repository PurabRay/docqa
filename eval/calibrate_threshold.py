"""Pick the abstention threshold from (question, is_answerable) records.

The threshold is the highest best-score cut that abstains on at most
``generation.max_false_abstention_rate`` of the answerable questions.

Usage:
    uv run python -m eval.calibrate_threshold            # dev split of eval/golden.jsonl
    uv run python -m eval.calibrate_threshold records.jsonl --session-id S
Records: one JSON object per line, {"question": str, "is_answerable": bool}.
Runs on the eval database; run eval/run_eval.py once first so the corpus is ingested.
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
    from docqa.bootstrap import build_container
    from docqa.domain.models import AccessFilter, IngestionStatus
    from docqa.retrieval.abstention import best_score
    from docqa.settings import load_settings

    base = load_settings()
    settings = load_settings(overrides={"mongodb": {"database": base.eval.database}})
    container = build_container(settings)
    stack = container.query_deps
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


def dev_split_records(golden: Path) -> list[dict[str, object]]:
    """The golden set's dev split as {question, is_answerable} records (frozen stays untouched)."""
    from eval.schema import GoldenRecord

    lines = [line for line in golden.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [GoldenRecord.model_validate(json.loads(line)) for line in lines]
    return [
        {"question": r.question, "is_answerable": r.answerable} for r in records if r.split == "dev"
    ]


def main() -> None:
    """CLI entry point. Default: the dev split of eval/golden.jsonl, on the eval corpus."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "records", type=Path, nargs="?", help="JSONL records; default: golden dev split"
    )
    parser.add_argument(
        "--session-id", default="eval-corpus", help="owner of the searched documents"
    )
    parser.add_argument("--max-false-rate", type=float, default=0.10)
    args = parser.parse_args()
    if args.records:
        lines = args.records.read_text().splitlines()
        records = [json.loads(line) for line in lines if line.strip()]
    else:
        records = dev_split_records(Path(__file__).parent / "golden.jsonl")
    result = choose_threshold(
        asyncio.run(best_scores(records, args.session_id)), args.max_false_rate
    )
    print(result.model_dump_json(indent=2))
    print("Copy the threshold into config (retrieval.abstain_threshold) and commit it.")


if __name__ == "__main__":
    main()
