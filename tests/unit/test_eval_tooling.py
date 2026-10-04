"""Eval metrics, golden-set validation and the gate, on hand-made cases (no golden data)."""

import json
import math

import pytest

from docqa.settings import load_settings
from eval.gate import gate_failures
from eval.llm_cache import CachingLLM
from eval.metrics.abstention import abstention_scores
from eval.metrics.citations import citation_accuracy, quote_on_page
from eval.metrics.judge import parse_json
from eval.metrics.ragas_runner import nan_as_failure
from eval.metrics.retrieval import ChunkRef, recall_at_k, reciprocal_rank
from eval.schema import CATEGORY_COUNTS, SPLIT_COUNTS
from eval.validate_golden import validate
from tests.fixtures.make_fixtures import HERE as FIXTURES
from tests.fixtures.make_fixtures import KNOWN_SENTENCES

GATE = load_settings(env_file=None).eval.gate


def ref(doc, start, end=None):
    return ChunkRef(doc_name=doc, page_start=start, page_end=end or start)


def test_recall_at_5_hit_and_miss():
    top = [ref("a.pdf", 1), ref("a.pdf", 7, 8), ref("b.pdf", 3)]
    assert recall_at_k(top, "a.pdf", [8], k=5) == 1.0  # inside a page range
    assert recall_at_k(top, "b.pdf", [4], k=5) == 0.0
    assert recall_at_k(top, "b.pdf", [3], k=2) == 0.0  # beyond k


def test_mrr_uses_the_first_gold_rank():
    fused = [ref("a.pdf", 1), ref("a.pdf", 2), ref("a.pdf", 3)]
    assert reciprocal_rank(fused, "a.pdf", [2, 3], k=20) == 0.5
    assert reciprocal_rank(fused, "a.pdf", [9], k=20) == 0.0


def test_citation_accuracy_exact_vs_paraphrased_quote():
    pages = {("r.pdf", 3): "The refund   window is 30 days\nfrom delivery."}
    text = lambda doc, page: pages.get((doc, page), "")  # noqa: E731
    exact = ("r.pdf", 3, "refund window is 30 days from delivery")
    paraphrase = ("r.pdf", 3, "refunds are allowed for a month")
    wrong_page = ("r.pdf", 4, "refund window is 30 days")
    assert quote_on_page(exact[2], pages[("r.pdf", 3)])
    assert citation_accuracy([exact, paraphrase, wrong_page], text) == pytest.approx(1 / 3)
    assert citation_accuracy([], text) is None


def test_abstention_counts():
    results = [
        (False, True),
        (False, True),
        (False, False),
        (True, False),
        (True, True),
        (True, False),
        (True, False),
    ]
    scores = abstention_scores(results)
    assert scores.abstention_accuracy == pytest.approx(2 / 3)
    assert scores.false_abstention == pytest.approx(1 / 4)


def test_nan_and_missing_scores_count_as_failure():
    assert nan_as_failure([0.9, math.nan, None]) == [0.9, 0.0, 0.0]


def test_judge_reply_parsing():
    assert parse_json('```json\n{"score": 4, "reason": "ok"}\n```') == {"score": 4, "reason": "ok"}
    assert parse_json("I think it is a 4") is None


# ---------------------------------------------------------------- golden validation


def records(quote=KNOWN_SENTENCES[3], page=3):
    """A structurally valid 40-record set over tests/fixtures/text.pdf (test data, not golden)."""
    rows, n = [], 0
    for category, count in CATEGORY_COUNTS.items():
        for _ in range(count):
            n += 1
            answerable = category != "unanswerable"
            rows.append(
                {
                    "id": f"T{n}",
                    "question": f"test question {n}",
                    "gold_answer": "x" if answerable else "",
                    "gold_doc": "text.pdf" if answerable else None,
                    "gold_pages": [page] if answerable else [],
                    "gold_quote": quote if answerable else None,
                    "category": category,
                    "split": "dev" if n <= SPLIT_COUNTS["dev"] else "frozen",
                }
            )
    return rows


def write(tmp_path, rows):
    path = tmp_path / "golden.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return path


def test_a_valid_set_passes(tmp_path):
    assert validate(write(tmp_path, records()), FIXTURES) == []


def test_wrong_count_is_caught(tmp_path):
    errors = validate(write(tmp_path, records()[:-1]), FIXTURES)
    assert any("39 records" in e for e in errors) and any("category counts" in e for e in errors)


def test_off_page_quote_is_caught(tmp_path):
    errors = validate(write(tmp_path, records(page=4)), FIXTURES)
    assert any("gold_quote not found on pages [4]" in e for e in errors)


def test_duplicates_are_caught(tmp_path):
    rows = records()
    rows[1]["question"] = rows[0]["question"]
    assert any("duplicate questions" in e for e in validate(write(tmp_path, rows), FIXTURES))


def test_missing_file_is_reported(tmp_path):
    assert "does not exist" in validate(tmp_path / "golden.jsonl", FIXTURES)[0]


# ---------------------------------------------------------------- gate


def report(faith=0.92, recall=0.88, citations=0.95):
    return {
        "overall": {"faithfulness": faith, "recall_at_5": recall, "citation_accuracy": citations}
    }


def test_gate_passes_within_margins():
    assert gate_failures(report(faith=0.90, recall=0.86), report(), GATE) == []


@pytest.mark.parametrize(
    "now, rule",
    [
        (report(faith=0.88), "faithfulness dropped"),
        (report(recall=0.84), "recall_at_5 dropped"),
        (report(citations=0.89), "citation_accuracy"),
        (report(faith=None), "faithfulness missing"),
    ],
)
def test_gate_fails(now, rule):
    assert any(rule in f for f in gate_failures(now, report(), GATE))


# ---------------------------------------------------------------- generator cache


async def test_cached_generations_replay_without_calling_the_model(tmp_path):
    from docqa.adapters.llm.stub import StubLLM
    from docqa.ports.llm import Message

    inner = StubLLM('{"answer": "x"}')
    messages = [Message(role="user", content="q")]
    first = CachingLLM(inner, tmp_path / "g.json", salt="v1|m|h")
    assert "".join([d.text async for d in first.stream(messages)]) == '{"answer": "x"}'
    second = CachingLLM(inner, tmp_path / "g.json", salt="v1|m|h")
    assert "".join([d.text async for d in second.stream(messages)]) == '{"answer": "x"}'
    assert len(inner.calls) == 1
    other = CachingLLM(inner, tmp_path / "g.json", salt="v2|m|h")  # new prompt version: miss
    assert other.key(messages) != first.key(messages)
