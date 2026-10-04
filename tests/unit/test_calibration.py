import json
from pathlib import Path

import pytest

from eval.calibrate_threshold import choose_threshold

SYNTHETIC = (
    Path(__file__).parents[2] / "eval" / "fixtures" / "synthetic_calibration_NOT_GOLDEN.jsonl"
)


def test_hand_made_scores():
    # 10 answerable: allowing 10% false abstentions means at most 1 may fall below t.
    scored = [(float(s), True) for s in range(1, 11)] + [(0.5, False), (1.5, False), (9.0, False)]
    result = choose_threshold(scored, max_false_rate=0.10)
    assert result.threshold == 2.0
    assert result.false_abstention_rate == pytest.approx(0.1)
    assert result.abstention_rate_unanswerable == pytest.approx(2 / 3)


def test_zero_tolerance_keeps_every_answerable_question():
    result = choose_threshold([(3.0, True), (1.0, True), (0.0, False)], max_false_rate=0.0)
    assert result.threshold == 1.0 and result.false_abstention_rate == 0.0


def test_needs_an_answerable_question():
    with pytest.raises(ValueError):
        choose_threshold([(1.0, False)], 0.1)


def test_synthetic_fixture_file():
    records = [json.loads(line) for line in SYNTHETIC.read_text().splitlines()]
    scored = [(r["best_score"], r["is_answerable"]) for r in records if "question" in r]
    result = choose_threshold(scored, max_false_rate=0.10)
    assert result.threshold == 0.4  # only the -2.5 answerable question is refused
    assert result.abstention_rate_unanswerable == pytest.approx(0.75)
