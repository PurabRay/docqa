"""Turn per-question results into the JSON + Markdown eval report."""

from __future__ import annotations

import json
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from docqa.settings import Settings
from eval.metrics.abstention import abstention_scores
from eval.metrics.citations import PageText, citation_accuracy
from eval.metrics.retrieval import recall_at_k, reciprocal_rank
from eval.runner import QuestionResult

TOKENS_PER_MILLION = 1_000_000


def mean(values: list[float]) -> float | None:
    """Average, or None for an empty list."""
    return sum(values) / len(values) if values else None


def percentile(values: list[float], pct: int) -> float | None:
    """The ``pct``-th percentile (1-99)."""
    if not values:
        return None
    return values[0] if len(values) == 1 else statistics.quantiles(values, n=100)[pct - 1]


def summarise(
    results: list[QuestionResult],
    scores: dict[str, int | None],
    page_text: PageText,
    settings: Settings,
) -> dict[str, Any]:
    """The headline metrics for a set of results."""
    e = settings.eval
    answerable = [r for r in results if r.record.answerable]
    gold = [(r, r.record.gold_doc or "", r.record.gold_pages) for r in answerable]
    citations = [c for r in results for c in r.citations]
    judged = [scores.get(r.record.id) for r in results]
    abstain = abstention_scores([(r.record.answerable, r.abstained) for r in results])
    return {
        "n": len(results),
        "recall_at_5": mean([recall_at_k(r.top, doc, pages, e.recall_k) for r, doc, pages in gold]),
        "mrr_at_20": mean(
            [reciprocal_rank(r.fused, doc, pages, e.mrr_k) for r, doc, pages in gold]
        ),
        "citation_accuracy": citation_accuracy(citations, page_text),
        "abstention_accuracy": abstain.abstention_accuracy,
        "false_abstention": abstain.false_abstention,
        # A missing judge score counts as incorrect, so judge failures never inflate the number.
        "correctness": mean(
            [1.0 if s is not None and s >= e.correct_score else 0.0 for s in judged]
        )
        if any(s is not None for s in judged)
        else None,
    }


def build_report(
    results: list[QuestionResult],
    scores: dict[str, int | None],
    ragas: dict[str, float | None],
    page_text: PageText,
    settings: Settings,
    split: str,
) -> dict[str, Any]:
    """Overall + per-category metrics, latency percentiles, tokens and equivalent cost."""
    t = settings.tracing
    latencies = sorted(float(r.latency_ms) for r in results)
    costs = [
        (r.input_tokens * t.cost_per_million_input + r.output_tokens * t.cost_per_million_output)
        / TOKENS_PER_MILLION
        for r in results
    ]
    categories = sorted({r.record.category for r in results})
    return {
        "split": split,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config_hash": settings.config_hash(),
        "prompt_version": settings.prompts.answer,
        "overall": summarise(results, scores, page_text, settings) | ragas,
        "per_category": {
            c: summarise(
                [r for r in results if r.record.category == c], scores, page_text, settings
            )
            for c in categories
        },
        "latency_ms": {f"p{p}": percentile(latencies, p) for p in (50, 95, 99)},
        "tokens": {
            "input_mean": mean([float(r.input_tokens) for r in results]),
            "output_mean": mean([float(r.output_tokens) for r in results]),
        },
        "cost_per_query_usd": mean(costs),
    }


def fmt(value: Any) -> str:
    """Numbers to 3 decimals; None as an em dash."""
    return "—" if value is None else f"{value:.3f}" if isinstance(value, float) else str(value)


def to_markdown(report: dict[str, Any]) -> str:
    """A human-readable version of the report."""
    lines = [
        f"# Eval report — {report['split']} split",
        "",
        f"config_hash `{report['config_hash'][:12]}`, prompt `{report['prompt_version']}`, "
        f"{report['created_at']}",
        "",
        "| metric | value |",
        "|---|---|",
    ]
    lines += [f"| {k} | {fmt(v)} |" for k, v in report["overall"].items()]
    lines += ["", "| latency | ms |", "|---|---|"] + [
        f"| {k} | {fmt(v)} |" for k, v in report["latency_ms"].items()
    ]
    lines += [
        "",
        f"Equivalent paid cost per query: ${fmt(report['cost_per_query_usd'])}",
        "",
        "| category | n | recall@5 | correctness | citation acc. | abstention acc. |",
        "|---|---|---|---|---|---|",
    ]
    for name, m in report["per_category"].items():
        lines.append(
            f"| {name} | {m['n']} | {fmt(m['recall_at_5'])} | {fmt(m['correctness'])} | "
            f"{fmt(m['citation_accuracy'])} | {fmt(m['abstention_accuracy'])} |"
        )
    return "\n".join(lines) + "\n"


def write_report(report: dict[str, Any], out_dir: Path) -> Path:
    """Write <split>-<timestamp>.json/.md and latest-<split>.json; return the JSON path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{report['split']}-{report['created_at'].replace(':', '').replace('+0000', 'Z')}"
    path = out_dir / f"{stem}.json"
    for target in (path, out_dir / f"latest-{report['split']}.json"):
        target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / f"{stem}.md").write_text(to_markdown(report), encoding="utf-8")
    (out_dir / f"latest-{report['split']}.md").write_text(to_markdown(report), encoding="utf-8")
    return path
