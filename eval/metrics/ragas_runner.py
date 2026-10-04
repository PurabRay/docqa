"""RAGAS faithfulness and context precision/recall, judged by Mistral. NaN counts as failure.

ragas 0.3.1's llm_factory takes no client, so the Mistral endpoint is wrapped through
LangChain's ChatOpenAI (same OpenAI-compatible endpoint as the judge).
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel

from docqa.settings import Settings


class RagasRow(BaseModel):
    """One answered question for RAGAS."""

    question: str
    answer: str
    contexts: list[str]
    reference: str


def nan_as_failure(values: list[float | None]) -> list[float]:
    """NaN or missing scores become 0.0, so a broken judge lowers the score (and fails the gate)."""
    return [0.0 if v is None or math.isnan(v) else float(v) for v in values]


def mean(values: list[float]) -> float | None:
    """Average, or None for an empty list."""
    return sum(values) / len(values) if values else None


def run_ragas(rows: list[RagasRow], settings: Settings) -> dict[str, float | None]:
    """faithfulness, context_precision, context_recall over ``rows`` (None without a key)."""
    cfg = settings.eval.judge
    key = settings.optional_secret(cfg.api_key_env) if cfg.api_key_env else None
    if not rows or key is None:
        return {"faithfulness": None, "context_precision": None, "context_recall": None}
    from langchain_openai import ChatOpenAI
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall

    llm = LangchainLLMWrapper(
        ChatOpenAI(model=cfg.model, base_url=cfg.base_url, api_key=key, temperature=0)
    )
    samples = [
        SingleTurnSample(
            user_input=r.question,
            response=r.answer,
            retrieved_contexts=r.contexts,
            reference=r.reference,
        )
        for r in rows
    ]
    metrics: list[Any] = [
        Faithfulness(llm=llm),
        LLMContextPrecisionWithReference(llm=llm),
        LLMContextRecall(llm=llm),
    ]
    frame = evaluate(EvaluationDataset(samples=samples), metrics=metrics).to_pandas()
    names = {
        "faithfulness": "faithfulness",
        "context_precision": "llm_context_precision_with_reference",
        "context_recall": "context_recall",
    }
    return {
        key_: mean(nan_as_failure(list(frame[col]))) if col in frame else None
        for key_, col in names.items()
    }
