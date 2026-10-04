"""Run the golden set end to end and write a report. Refuses to run until golden.jsonl validates.

Usage:
    uv run python -m eval.run_eval --config free --split dev --out eval/reports
Runs on atlas-local only (MONGODB_URI), in the eval.database database.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from docqa.bootstrap import build_breakers, build_container, build_llm_router, init_database
from docqa.generation.schemas import LLMAnswer
from docqa.settings import Settings, load_settings
from eval.llm_cache import CachingLLM
from eval.metrics.citations import corpus_page_text
from eval.metrics.judge import build_judge
from eval.metrics.ragas_runner import RagasRow, run_ragas
from eval.report import build_report, to_markdown, write_report
from eval.runner import QuestionResult, ingest_corpus, run_questions
from eval.validate_golden import EVAL_DIR, load, validate

ATLAS_HOST_MARKER = "mongodb.net"  # never evaluate on the Atlas free cluster


def eval_settings(profile: str) -> Settings:
    """Settings for the profile, pointed at the eval database."""
    base = load_settings(profile)
    return load_settings(profile, overrides={"mongodb": {"database": base.eval.database}})


async def judge_all(results: list[QuestionResult], settings: Settings) -> dict[str, int | None]:
    """1-5 correctness per question id (empty when no judge key is set)."""
    judge = build_judge(settings, settings.eval.judge_prompt)
    if judge is None:
        print("judge: skipped (MISTRAL_API_KEY not set); correctness will be missing")
        return {}
    return {
        r.record.id: await judge.correctness(r.record.question, r.record.gold_answer, r.answer)
        for r in results
    }


async def run(settings: Settings, split: str, out: Path) -> Path:
    """Ingest, ask, score, report."""
    golden = EVAL_DIR / "golden.jsonl"
    records = [r for r in load(golden) if r.split == split]
    breakers = build_breakers(settings)
    salt = f"{settings.prompts.answer}|{settings.llm.router}|{settings.config_hash()}"
    answer_llm = CachingLLM(
        build_llm_router(settings, breakers, LLMAnswer),
        Path(settings.eval.cache_dir) / "generations.json",
        salt,
    )
    container = build_container(settings, answer_llm=answer_llm)
    try:
        await init_database(container)
        doc_ids = await ingest_corpus(
            container, Path(settings.eval.corpus_dir), settings.eval.session_id
        )
        results = await run_questions(container, records, doc_ids, settings.eval.session_id)
    finally:
        await container.close()
    scores = await judge_all(results, settings)
    ragas = run_ragas(
        [
            RagasRow(
                question=r.record.question,
                answer=r.answer,
                contexts=r.contexts,
                reference=r.record.gold_answer,
            )
            for r in results
            if r.record.answerable
        ],
        settings,
    )
    report = build_report(
        results, scores, ragas, corpus_page_text(Path(settings.eval.corpus_dir)), settings, split
    )
    path = write_report(report, out)
    print(to_markdown(report))
    return path


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run the golden-set evaluation.")
    parser.add_argument("--config", default="free", help="profile: free, local or an ablation name")
    parser.add_argument("--split", choices=["dev", "frozen"], default="dev")
    parser.add_argument("--out", type=Path, default=EVAL_DIR / "reports")
    args = parser.parse_args()
    errors = validate(EVAL_DIR / "golden.jsonl", EVAL_DIR / "corpus")
    if errors:
        print("Refusing to report numbers: the golden set does not validate.")
        for error in errors:
            print(f"  {error}")
        return 1
    settings = eval_settings(args.config)
    if ATLAS_HOST_MARKER in settings.mongodb_uri():
        print("Refusing to run the eval on the Atlas free cluster; use atlas-local.")
        return 1
    print(f"report written to {asyncio.run(run(settings, args.split, args.out))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
