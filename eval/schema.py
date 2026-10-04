"""One hand-written golden record (docs/PRD.md, "Offline evaluation: the golden set")."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Category = Literal["single", "table", "multi", "followup", "unanswerable"]

# Required mix of the 40 records (PRD table) and the dev/frozen split.
CATEGORY_COUNTS: dict[str, int] = {
    "single": 16,
    "table": 6,
    "multi": 6,
    "followup": 4,
    "unanswerable": 8,
}
SPLIT_COUNTS: dict[str, int] = {"dev": 28, "frozen": 12}


class GoldenRecord(BaseModel):
    """A question, its gold answer and the evidence that proves it."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    gold_answer: str  # empty for unanswerable questions
    gold_doc: str | None = None  # file name in eval/corpus; None for unanswerable
    gold_pages: list[int] = []  # 1-based
    gold_quote: str | None = None  # verbatim text on one of gold_pages
    category: Category
    split: Literal["dev", "frozen"]
    history: list[str] = []  # earlier user turns, for follow-ups

    @property
    def answerable(self) -> bool:
        """True unless the question is designed to have no answer in the corpus."""
        return self.category != "unanswerable"
