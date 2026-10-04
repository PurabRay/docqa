import uuid
from datetime import UTC, datetime, timedelta

import pytest
from pymongo.errors import WriteError

from docqa.domain.errors import DocumentNotFoundError, DuplicateDocumentError
from docqa.domain.models import DocumentRecord, Feedback, IngestionStatus, Turn

T0 = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)


def record(owner: str, sha_char: str = "a", minutes: int = 0) -> DocumentRecord:
    at = T0 + timedelta(minutes=minutes)
    return DocumentRecord(
        owner_id=owner,
        filename="a.pdf",
        sha256=sha_char * 64,
        page_count=3,
        embed_model="nomic",
        created_at=at,
        updated_at=at,
    )


@pytest.fixture
def owner() -> str:
    return f"user-{uuid.uuid4().hex[:6]}"


async def test_create_get_and_list(initialized, owner):
    repo = initialized.repository
    old, new = record(owner, "a", minutes=0), record(owner, "b", minutes=5)
    await repo.create(old)
    await repo.create(new)

    assert await repo.get(old.id, owner) == old
    assert await repo.get(old.id, "someone-else") is None
    assert [d.id for d in await repo.list_for_owner(owner)] == [new.id, old.id]


async def test_duplicate_owner_and_sha256_raises(initialized, owner):
    await initialized.repository.create(record(owner, "c"))
    with pytest.raises(DuplicateDocumentError):
        await initialized.repository.create(record(owner, "c"))


async def test_same_file_for_another_owner_is_fine(initialized, owner):
    await initialized.repository.create(record(owner, "d"))
    await initialized.repository.create(record(owner + "-2", "d"))


async def test_set_status(initialized, owner):
    repo = initialized.repository
    doc = record(owner, "e")
    await repo.create(doc)
    await repo.set_status(doc.id, IngestionStatus.FAILED, error="no text layer")

    stored = await repo.get(doc.id, owner)
    assert stored.status is IngestionStatus.FAILED and stored.error == "no text layer"
    with pytest.raises(DocumentNotFoundError):
        await repo.set_status("missing", IngestionStatus.READY)


async def test_validator_rejects_unknown_status(initialized, owner):
    with pytest.raises(WriteError):
        await initialized.mongo.db["documents"].insert_one(
            {"owner_id": owner, "sha256": "f" * 64, "status": "bogus"}
        )


async def test_last_turns_returns_newest_n_oldest_first(initialized):
    repo = initialized.repository
    session = f"s-{uuid.uuid4().hex[:6]}"
    for i in range(5):
        role = "user" if i % 2 == 0 else "assistant"
        turn = Turn(
            session_id=session,
            role=role,
            content_scrubbed=f"m{i}",
            created_at=T0 + timedelta(seconds=i),
        )
        await repo.append_turn(turn)

    turns = await repo.last_turns(session, n=3)
    assert [t.content_scrubbed for t in turns] == ["m2", "m3", "m4"]


async def test_add_feedback(initialized):
    fb = Feedback(trace_id=f"t-{uuid.uuid4().hex[:6]}", session_id="s", rating=1, created_at=T0)
    await initialized.repository.add_feedback(fb)
    stored = await initialized.mongo.db["feedback"].find_one({"trace_id": fb.trace_id})
    assert stored["rating"] == 1
