"""Fill the README's number tables from report files; never hand-type a number.

Reads eval/reports/latest-frozen.json (eval), eval/reports/ablations.json and
eval/reports/load-stub.json / load-real.json, and rewrites the text between
<!-- numbers:start --> / <!-- numbers:end --> and <!-- ablations:start --> / <!-- ablations:end -->.
A missing report is shown as "not measured yet" with the command that produces it.

Usage: uv run python -m scripts.render_readme_tables [--check]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[1]
REPORTS = ROOT / "eval" / "reports"
README = ROOT / "README.md"
NOT_MEASURED = "not measured yet — run `{command}`"


def load(name: str) -> dict[str, Any] | None:
    """A report file, or None if it does not exist yet."""
    path = REPORTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def num(value: Any, digits: int = 3) -> str:
    """Format a number; None as an em dash."""
    return (
        "—" if value is None else f"{value:.{digits}f}" if isinstance(value, float) else str(value)
    )


def numbers_table() -> str:
    """Latency, cost, quality, throughput, ops/s and storage."""
    lines = ["| metric | value | source |", "|---|---|---|"]
    frozen = load("latest-frozen.json")
    if frozen:
        o, lat = frozen["overall"], frozen["latency_ms"]
        source = f"frozen split, config `{frozen['config_hash'][:12]}`"
        values = [
            ("Latency p50 / p99 (ms)", f"{num(lat['p50'], 0)} / {num(lat['p99'], 0)}"),
            ("Equivalent paid cost per request (USD)", num(frozen["cost_per_query_usd"], 5)),
            ("Answer correctness", num(o.get("correctness"))),
            ("Faithfulness", num(o.get("faithfulness"))),
            ("Recall@5", num(o.get("recall_at_5"))),
            ("Citation accuracy", num(o.get("citation_accuracy"))),
            ("Abstention accuracy", num(o.get("abstention_accuracy"))),
            ("False abstention", num(o.get("false_abstention"))),
            ("Storage per document (MB)", num(frozen.get("storage", {}).get("mb_per_document"))),
        ]
        lines += [f"| {label} | {value} | {source} |" for label, value in values]
    else:
        missing = NOT_MEASURED.format(command="make eval")
        lines.append(f"| eval scores, latency, cost, storage | {missing} | — |")
    for label, name, command in (
        ("stub", "load-stub.json", "make load"),
        ("real provider", "load-real.json", "locust ... --mode real"),
    ):
        report = load(name)
        if report is None:
            lines.append(f"| Throughput, {label} | {NOT_MEASURED.format(command=command)} | — |")
            continue
        errors, peak = num(report["error_rate"] * 100, 2), num(report["peak_ops_per_s"], 0)
        detail = f"error rate {errors}%, peak {peak} ops/s"
        lines.append(
            f"| Throughput, {label} ({report['users']} users, req/s) | "
            f"{num(report['requests_per_s'], 2)} | {detail} of {report['ops_cap']} |"
        )
    return "\n".join(lines)


def ablations_table() -> str:
    """The ablation comparison (eval/run_ablations.py)."""
    path = REPORTS / "ablations.json"
    rows: list[dict[str, Any]] = (
        json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
    )
    if not rows:
        return NOT_MEASURED.format(command="uv run python -m eval.run_ablations")
    header = [
        "profile",
        "recall_at_5",
        "faithfulness",
        "correctness",
        "p95_ms",
        "cost_per_query_usd",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(num(r[h]) for h in header) + " |" for r in rows])


def render(text: str) -> str:
    """Replace both marked blocks."""
    for name, body in (("numbers", numbers_table()), ("ablations", ablations_table())):
        start, end = f"<!-- {name}:start -->", f"<!-- {name}:end -->"
        before, rest = text.split(start, 1)
        _, after = rest.split(end, 1)
        text = f"{before}{start}\n{body}\n{end}{after}"
    return text


def main() -> int:
    """Rewrite README.md, or with --check fail if it is out of date."""
    parser = argparse.ArgumentParser(description="Render README number tables from reports.")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    current = README.read_text(encoding="utf-8")
    rendered = render(current)
    if args.check:
        print("README up to date" if rendered == current else "README numbers are stale")
        return 0 if rendered == current else 1
    README.write_text(rendered, encoding="utf-8")
    print("README tables rendered")
    return 0


if __name__ == "__main__":
    sys.exit(main())
