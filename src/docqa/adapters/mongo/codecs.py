"""Convert domain models to BSON documents and back. The only place that knows BSON.

Conventions:
    * a model's ``id`` is stored as ``_id``;
    * enums are stored as their string value;
    * vectors are packed as BSON binary float32 (``Binary.from_vector``), about
      3 KB for 768 dims instead of ~8 KB as an array of doubles.
"""

from __future__ import annotations

from typing import Any

from bson.binary import Binary, BinaryVectorDtype
from pydantic import BaseModel

from docqa.domain.models import (
    Chunk,
    DocumentRecord,
    EmbeddedChunk,
    Feedback,
    RetrievedChunk,
    Turn,
)

Document = dict[str, Any]


def vector_to_bson(vector: list[float]) -> Binary:
    """Pack a float vector as BSON binary float32."""
    return Binary.from_vector(vector, BinaryVectorDtype.FLOAT32)


def vector_from_bson(value: Binary) -> list[float]:
    """Unpack a BSON binary float32 vector."""
    return [float(x) for x in value.as_vector().data]


def _to_bson(model: BaseModel) -> Document:
    doc = model.model_dump()
    doc["_id"] = doc.pop("id")
    return doc


def _from_bson(doc: Document) -> Document:
    fields = dict(doc)
    fields["id"] = fields.pop("_id")
    return fields


# ---------------------------------------------------------------- documents


def document_to_bson(record: DocumentRecord) -> Document:
    """Encode a DocumentRecord for the ``documents`` collection."""
    doc = _to_bson(record)
    doc["status"] = record.status.value
    return doc


def document_from_bson(doc: Document) -> DocumentRecord:
    """Decode a ``documents`` collection document."""
    return DocumentRecord.model_validate(_from_bson(doc))


# ---------------------------------------------------------------- chunks


def chunk_to_bson(item: EmbeddedChunk) -> Document:
    """Encode an EmbeddedChunk for the ``chunks_<model>`` collection."""
    doc = _to_bson(item.chunk)
    doc["embedding"] = vector_to_bson(item.embedding)
    doc["embed_model"] = item.embed_model
    return doc


# Fields on a chunk document (or search result) that are not part of the Chunk model.
NON_CHUNK_FIELDS = ("embedding", "embed_model", "fusion", "score")


def chunk_from_bson(doc: Document) -> Chunk:
    """Decode a chunk document; vectors and search metadata are ignored if present."""
    fields = _from_bson(doc)
    for name in NON_CHUNK_FIELDS:
        fields.pop(name, None)
    return Chunk.model_validate(fields)


def retrieved_from_bson(
    doc: Document, fused_score: float, vector_rank: int | None, text_rank: int | None
) -> RetrievedChunk:
    """Decode a search result with its fusion score and branch ranks."""
    return RetrievedChunk(
        chunk=chunk_from_bson(doc),
        fused_score=fused_score,
        vector_rank=vector_rank,
        text_rank=text_rank,
    )


def embedded_chunk_from_bson(doc: Document) -> EmbeddedChunk:
    """Decode a chunk document including its vector."""
    return EmbeddedChunk(
        chunk=chunk_from_bson(doc),
        embedding=vector_from_bson(doc["embedding"]),
        embed_model=doc["embed_model"],
    )


# ---------------------------------------------------------------- turns and feedback


def turn_to_bson(turn: Turn) -> Document:
    """Encode a chat turn for the ``turns`` collection."""
    return _to_bson(turn)


def turn_from_bson(doc: Document) -> Turn:
    """Decode a ``turns`` collection document."""
    return Turn.model_validate(_from_bson(doc))


def feedback_to_bson(feedback: Feedback) -> Document:
    """Encode feedback for the ``feedback`` collection."""
    return _to_bson(feedback)


def feedback_from_bson(doc: Document) -> Feedback:
    """Decode a ``feedback`` collection document."""
    return Feedback.model_validate(_from_bson(doc))
