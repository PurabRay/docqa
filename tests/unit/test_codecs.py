import random
import struct
from datetime import UTC, datetime

import pytest
from bson import BSON
from bson.binary import Binary
from bson.codec_options import CodecOptions

from docqa.adapters.mongo import codecs
from docqa.domain.models import (
    Chunk,
    DocumentRecord,
    EmbeddedChunk,
    Feedback,
    IngestionStatus,
    Turn,
    chunk_id,
)

# BSON datetimes have millisecond precision, so tests use whole-second timestamps.
NOW = datetime(2026, 10, 4, 12, 0, 0, tzinfo=UTC)


def through_bson(doc):
    """Encode and decode like a real database round trip (tz_aware, as the client)."""
    return BSON.decode(BSON.encode(doc), CodecOptions(tz_aware=True))


def as_float32(values):
    return list(struct.unpack(f"{len(values)}f", struct.pack(f"{len(values)}f", *values)))


def make_chunk(dim: int) -> EmbeddedChunk:
    chunk = Chunk(
        id=chunk_id("d1", 0),
        doc_id="d1",
        owner_id="u1",
        filename="a.pdf",
        text="Revenue grew 12%.",
        page_start=4,
        page_end=5,
        section="Results",
        content_hash="h",
    )
    vector = [random.uniform(-1, 1) for _ in range(dim)]
    return EmbeddedChunk(chunk=chunk, embedding=vector, embed_model="nomic")


@pytest.mark.parametrize("dim", [512, 768])
def test_chunk_round_trip_keeps_float32_vector(dim):
    item = make_chunk(dim)
    doc = codecs.chunk_to_bson(item)

    assert doc["_id"] == item.chunk.id and "id" not in doc
    assert isinstance(doc["embedding"], Binary)
    decoded = codecs.embedded_chunk_from_bson(through_bson(doc))
    assert decoded.chunk == item.chunk
    assert decoded.embedding == as_float32(item.embedding)


def test_packed_vector_is_four_bytes_per_dimension():
    doc = codecs.chunk_to_bson(make_chunk(768))
    assert len(doc["embedding"]) == 768 * 4 + 2  # +2 header bytes: dtype and padding


def test_chunk_from_bson_ignores_vector_fields():
    item = make_chunk(8)
    assert codecs.chunk_from_bson(codecs.chunk_to_bson(item)) == item.chunk


def test_document_round_trip_stores_status_as_string():
    record = DocumentRecord(
        owner_id="u1",
        filename="a.pdf",
        sha256="a" * 64,
        page_count=10,
        status=IngestionStatus.SYNCING,
        embed_model="nomic",
        created_at=NOW,
        updated_at=NOW,
    )
    doc = codecs.document_to_bson(record)
    assert type(doc["status"]) is str and doc["status"] == "syncing"
    assert codecs.document_from_bson(through_bson(doc)) == record


def test_turn_round_trip():
    turn = Turn(session_id="s1", role="user", content_scrubbed="hi <EMAIL>", created_at=NOW)
    assert codecs.turn_from_bson(through_bson(codecs.turn_to_bson(turn))) == turn


def test_feedback_round_trip():
    feedback = Feedback(trace_id="t1", session_id="s1", rating=-1, comment="no", created_at=NOW)
    assert codecs.feedback_from_bson(through_bson(codecs.feedback_to_bson(feedback))) == feedback
