"""CI eval gate: compare a report with eval/reports/baseline.json (PRD: CI/CD).

Fails if faithfulness or Recall@5 drop by more than the configured margin, or citation
accuracy falls below the floor. A missing metric counts as a failure.

Usage:
    uv run python -m eval.gate eval/reports/latest-dev.json [--baseline FILE]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from docqa.config_schema import GateConfig
from docqa.settings import load_settings

EVAL_DIR = Path(__file__).parent


def gate_failures(report: dict[str, Any], baseline: dict[str, Any], cfg: GateConfig) -> list[str]:
    """Every rule the report breaks; empty means the gate passes."""
    now, before, failures = report["overall"], baseline["overall"], []
    for metric, margin in (
        ("faithfulness", cfg.max_faithfulness_drop),
        ("recall_at_5", cfg.max_recall_drop),
    ):
        if now.get(metric) is None or before.get(metric) is None:
            failures.append(
                f"{metric} missing (report {now.get(metric)}, baseline {before.get(metric)})"
            )
        elif before[metric] - now[metric] > margin:
            failures.append(f"{metric} dropped {before[metric] - now[metric]:.3f} (> {margin})")
    accuracy = now.get("citation_accuracy")
    if accuracy is None or accuracy < cfg.min_citation_accuracy:
        failures.append(f"citation_accuracy {accuracy} < {cfg.min_citation_accuracy}")
    return failures


def main() -> int:
    """CLI: print the verdict; exit 1 if the gate fails."""
    parser = argparse.ArgumentParser(description="Eval gate.")
    parser.add_argument("report", type=Path)
    parser.add_argument("--baseline", type=Path, default=EVAL_DIR / "reports" / "baseline.json")
    args = parser.parse_args()
    if not args.baseline.is_file():
        print(f"No baseline at {args.baseline}; commit one from a reviewed run first.")
        return 1
    report, baseline = (
        json.loads(p.read_text(encoding="utf-8")) for p in (args.report, args.baseline)
    )
    failures = gate_failures(report, baseline, load_settings(env_file=None).eval.gate)
    for failure in failures:
        print(f"GATE FAIL: {failure}")
    print("gate passed" if not failures else "gate failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
