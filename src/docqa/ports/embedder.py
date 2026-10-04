"""Dense embedder port: text in, fixed-size float vectors out."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class DenseEmbedder(Protocol):
    """An embedding model pinned to one id and one dimension. Calls are CPU-bound."""

    model_id: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed chunks for indexing."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a question for search."""
        ...
