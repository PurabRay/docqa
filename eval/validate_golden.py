"""Check eval/golden.jsonl before any eval number is reported.

40 records, categories 16/6/6/4/8, 28 dev / 12 frozen, no duplicate ids or questions,
and every gold_quote really printed on one of its gold_pages in eval/corpus/<gold_doc>.

Usage: uv run python -m eval.validate_golden [path] [--corpus eval/corpus]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pymupdf
from pydantic import ValidationError

from eval.schema import CATEGORY_COUNTS, SPLIT_COUNTS, GoldenRecord

EVAL_DIR = Path(__file__).parent
WHITESPACE = re.compile(r"\s+")


def load(path: Path) -> list[GoldenRecord]:
    """Parse one record per non-empty line."""
    return [
        GoldenRecord.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def normalise(text: str) -> str:
    """Single spaces; the quote must otherwise appear exactly."""
    return WHITESPACE.sub(" ", text).strip()


def count_errors(records: list[GoldenRecord]) -> list[str]:
    """Totals, category mix, split sizes and duplicates."""
    errors = []
    expected_total = sum(CATEGORY_COUNTS.values())
    if len(records) != expected_total:
        errors.append(f"{len(records)} records, expected {expected_total}")
    for name, expected, counts in (
        ("category", CATEGORY_COUNTS, Counter(r.category for r in records)),
        ("split", SPLIT_COUNTS, Counter(r.split for r in records)),
    ):
        if dict(counts) != expected:
            errors.append(f"{name} counts {dict(counts)}, expected {expected}")
    for field in ("id", "question"):
        dupes = [v for v, n in Counter(getattr(r, field) for r in records).items() if n > 1]
        if dupes:
            errors.append(f"duplicate {field}s: {dupes}")
    return errors


def quote_errors(records: list[GoldenRecord], corpus: Path) -> list[str]:
    """Every answerable record's quote must be on one of its gold pages."""
    errors = []
    for r in records:
        if not r.answerable:
            continue
        if not (r.gold_doc and r.gold_pages and r.gold_quote):
            errors.append(f"{r.id}: answerable records need gold_doc, gold_pages and gold_quote")
            continue
        path = corpus / r.gold_doc
        if not path.is_file():
            errors.append(f"{r.id}: {path} not found")
            continue
        with pymupdf.open(path) as doc:
            pages = [
                normalise(doc[p - 1].get_text()) for p in r.gold_pages if 1 <= p <= doc.page_count
            ]
        if not any(normalise(r.gold_quote) in page for page in pages):
            errors.append(f"{r.id}: gold_quote not found on pages {r.gold_pages} of {r.gold_doc}")
    return errors


def validate(path: Path, corpus: Path) -> list[str]:
    """All problems with the golden file; empty means valid."""
    if not path.is_file():
        return [f"{path} does not exist (write it by hand; see eval/golden.template.jsonl)"]
    try:
        records = load(path)
    except (ValidationError, json.JSONDecodeError) as err:
        return [f"invalid record: {err}"]
    return count_errors(records) + quote_errors(records, corpus)


def main() -> int:
    """CLI: print problems, exit 1 if any."""
    parser = argparse.ArgumentParser(description="Validate the golden set.")
    parser.add_argument("path", type=Path, nargs="?", default=EVAL_DIR / "golden.jsonl")
    parser.add_argument("--corpus", type=Path, default=EVAL_DIR / "corpus")
    args = parser.parse_args()
    errors = validate(args.path, args.corpus)
    for error in errors:
        print(f"INVALID: {error}")
    print("golden set is valid" if not errors else f"{len(errors)} problem(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
