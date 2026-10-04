"""Abstention accuracy on unanswerable questions and false abstention on answerable ones."""

from __future__ import annotations

from pydantic import BaseModel


class AbstentionScores(BaseModel):
    """PRD targets: abstention accuracy >= 0.80, false abstention <= 0.10."""

    abstention_accuracy: float | None  # abstained / unanswerable
    false_abstention: float | None  # abstained / answerable


def abstention_scores(results: list[tuple[bool, bool]]) -> AbstentionScores:
    """From (answerable, abstained) pairs, one per question."""
    unanswerable = [abstained for answerable, abstained in results if not answerable]
    answerable = [abstained for ok, abstained in results if ok]
    return AbstentionScores(
        abstention_accuracy=sum(unanswerable) / len(unanswerable) if unanswerable else None,
        false_abstention=sum(answerable) / len(answerable) if answerable else None,
    )
