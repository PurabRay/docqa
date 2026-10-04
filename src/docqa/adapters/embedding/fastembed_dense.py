"""FastEmbedDenseEmbedder: ONNX embeddings on CPU (default nomic-embed-text-v1.5).

nomic expects task prefixes ("search_document: " / "search_query: "), set in config.
Vectors are truncated to ``embedding.dim`` (Matryoshka) and L2-normalised.
Calls are CPU-bound: callers run them with ``asyncio.to_thread``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike

from docqa.config_schema import EmbeddingConfig
from docqa.domain.errors import ConfigurationError


class EmbeddingModel(Protocol):
    """The part of fastembed.TextEmbedding we use; tests pass a fake."""

    def embed(self, documents: Iterable[str], batch_size: int = 256) -> Iterable[ArrayLike]:
        """Yield one vector per document."""
        ...


def truncate_and_normalise(vector: ArrayLike, dim: int) -> list[float]:
    """Keep the first ``dim`` values and scale to unit length."""
    values = np.asarray(vector, dtype=np.float32)
    if values.shape[0] < dim:
        raise ConfigurationError(
            f"The model returns {values.shape[0]} dims, config asks for {dim}."
        )
    values = values[:dim]
    norm = float(np.linalg.norm(values))
    return [float(x) for x in (values / norm if norm else values)]


class FastEmbedDenseEmbedder:
    """Implements the DenseEmbedder port.

    Args:
        cfg: The ``embedding`` config section.
        model: Optional ready model; by default fastembed loads it on first use.
    """

    def __init__(self, cfg: EmbeddingConfig, model: EmbeddingModel | None = None) -> None:
        self.model_id = cfg.dense_model
        self.dim = cfg.dim
        self._cfg = cfg
        self._model = model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed chunk texts for indexing."""
        return self._embed([self._cfg.document_prefix + text for text in texts])

    def embed_query(self, text: str) -> list[float]:
        """Embed a question for search."""
        return self._embed([self._cfg.query_prefix + text])[0]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._load().embed(texts, batch_size=self._cfg.batch_size)
        return [truncate_and_normalise(vector, self.dim) for vector in vectors]

    def _load(self) -> EmbeddingModel:
        if self._model is None:
            from fastembed import TextEmbedding  # heavy import, only when first needed

            self._model = TextEmbedding(self.model_id)
        return self._model
