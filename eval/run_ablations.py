"""Run the dev split for the default profile and every config/ablations/*.yaml; one table.

Usage: uv run python -m eval.run_ablations [--only vector_only rerank_off]
Writes eval/reports/ablations.json and .md (atlas-local only, like run_eval).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from eval.report import fmt
from eval.run_eval import ATLAS_HOST_MARKER, eval_settings, run
from eval.validate_golden import EVAL_DIR, validate

ABLATIONS_DIR = EVAL_DIR.parent / "config" / "ablations"
COLUMNS = ["recall_at_5", "faithfulness", "correctness"]


def row(name: str, report: dict[str, Any]) -> dict[str, Any]:
    """The comparison columns of one report."""
    overall = report["overall"]
    return {
        "profile": name,
        **{c: overall.get(c) for c in COLUMNS},
        "p95_ms": report["latency_ms"]["p95"],
        "cost_per_query_usd": report["cost_per_query_usd"],
        "config_hash": report["config_hash"][:12],
    }


def to_markdown(rows: list[dict[str, Any]]) -> str:
    """One table, one row per profile."""
    header = ["profile", *COLUMNS, "p95_ms", "cost_per_query_usd", "config_hash"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(fmt(r[h]) for h in header) + " |" for r in rows]
    return "\n".join(lines) + "\n"


async def run_all(names: list[str], out: Path) -> list[dict[str, Any]]:
    """Default profile first, then each ablation, all on the dev split."""
    rows = []
    for name in ["free", *names]:
        settings = eval_settings(name)
        if ATLAS_HOST_MARKER in settings.mongodb_uri():
            raise SystemExit("Ablations run on atlas-local only.")
        report_path = await run(settings, "dev", out / name)
        rows.append(row(name, json.loads(report_path.read_text(encoding="utf-8"))))
    return rows


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run ablations on the dev split.")
    parser.add_argument("--only", nargs="*", help="ablation names (default: all)")
    args = parser.parse_args()
    if errors := validate(EVAL_DIR / "golden.jsonl", EVAL_DIR / "corpus"):
        print("Refusing to run: the golden set does not validate.", *errors, sep="\n  ")
        return 1
    names = args.only or sorted(p.stem for p in ABLATIONS_DIR.glob("*.yaml"))
    out = EVAL_DIR / "reports"
    rows = asyncio.run(run_all(names, out / "ablations"))
    (out / "ablations.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (out / "ablations.md").write_text(to_markdown(rows), encoding="utf-8")
    print(to_markdown(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
