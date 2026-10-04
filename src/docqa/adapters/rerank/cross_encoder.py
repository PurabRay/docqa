"""FastEmbedReranker: MiniLM cross-encoder (ONNX, CPU). Scores are raw logits.

CPU-bound: callers run rerank() with asyncio.to_thread.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Protocol

from docqa.domain.models import RetrievedChunk

log = logging.getLogger(__name__)

# Rough characters-per-token for English, only used to decide whether to warn.
CHARS_PER_TOKEN = 4


class CrossEncoderModel(Protocol):
    """The part of fastembed's TextCrossEncoder we use; tests pass a fake."""

    def rerank(self, query: str, documents: Iterable[str], batch_size: int = 64) -> Iterable[float]:
        """Return one relevance score per document."""
        ...


class FastEmbedReranker:
    """Implements the Reranker port.

    Args:
        model_name: e.g. "Xenova/ms-marco-MiniLM-L-6-v2".
        max_tokens: The model's pair limit; longer pairs are truncated by the model.
        model: Optional ready model; by default fastembed loads it on first use.
    """

    def __init__(
        self, model_name: str, max_tokens: int, model: CrossEncoderModel | None = None
    ) -> None:
        self._model_name = model_name
        self._max_chars = max_tokens * CHARS_PER_TOKEN
        self._model = model
        self._warned = False

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        """Score every (query, chunk) pair and keep the best ``top_k``."""
        if not chunks:
            return []
        texts = [c.chunk.text for c in chunks]
        self._warn_if_truncated(query, texts)
        scores = list(self._load().rerank(query, texts))
        scored = [
            c.model_copy(update={"rerank_score": float(s)})
            for c, s in zip(chunks, scores, strict=True)
        ]
        return sorted(scored, key=lambda c: c.rerank_score or 0.0, reverse=True)[:top_k]

    def _warn_if_truncated(self, query: str, texts: list[str]) -> None:
        if not self._warned and any(len(query) + len(t) > self._max_chars for t in texts):
            log.warning(
                "rerank.truncated: some (query, chunk) pairs exceed the model's token limit"
            )
            self._warned = True

    def _load(self) -> CrossEncoderModel:
        if self._model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._model = TextCrossEncoder(self._model_name)
        return self._model
