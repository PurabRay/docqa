import math

import pytest

from docqa.adapters.embedding.fastembed_dense import FastEmbedDenseEmbedder, truncate_and_normalise
from docqa.domain.errors import ConfigurationError
from docqa.settings import load_settings


class FakeModel:
    """Returns [3, 4, 1, 1, ...] for every text and records what it was asked."""

    def __init__(self, size: int = 768) -> None:
        self.size = size
        self.calls: list[tuple[list[str], int]] = []

    def embed(self, documents, batch_size=256):
        documents = list(documents)
        self.calls.append((documents, batch_size))
        return [[3.0, 4.0] + [0.0] * (self.size - 2) for _ in documents]


def embedder(dim=768, model=None):
    cfg = load_settings(env_file=None).embedding.model_copy(update={"dim": dim})
    model = model or FakeModel()
    return FastEmbedDenseEmbedder(cfg, model), model


def test_documents_get_the_document_prefix_and_configured_batch_size():
    emb, model = embedder()
    emb.embed_documents(["alpha", "beta"])
    assert model.calls == [(["search_document: alpha", "search_document: beta"], 64)]


def test_queries_get_the_query_prefix():
    emb, model = embedder()
    emb.embed_query("what is the refund window?")
    assert model.calls[0][0] == ["search_query: what is the refund window?"]


def test_vectors_are_unit_length_and_configured_size():
    emb, _ = embedder(dim=512)
    [vector] = emb.embed_documents(["x"])
    assert len(vector) == 512 and emb.dim == 512
    assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0, rel_tol=1e-6)
    assert vector[:2] == pytest.approx([0.6, 0.8])


def test_truncation_happens_before_normalising():
    assert truncate_and_normalise([1.0, 0.0, 100.0], 2) == [1.0, 0.0]


def test_zero_vector_stays_zero():
    assert truncate_and_normalise([0.0, 0.0], 2) == [0.0, 0.0]


def test_dimension_larger_than_the_model_is_a_config_error():
    emb, _ = embedder(dim=1024)
    with pytest.raises(ConfigurationError, match="1024"):
        emb.embed_documents(["x"])
