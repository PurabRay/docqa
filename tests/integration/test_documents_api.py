"""Upload -> ready -> search -> delete through the real API, worker and atlas-local.

Uses the real FastEmbed model (downloaded on first run). Tests in this module share
one app and run in order.
"""

import asyncio
import time
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from bson.binary import Binary

from docqa.api.app import create_app
from docqa.api.deps import owner_id
from docqa.bootstrap import init_database
from docqa.domain.models import IngestJob
from docqa.ingestion.chunker import chunk_pages
from docqa.ingestion.extractor import extract_pages
from tests.fixtures.make_fixtures import KNOWN_SENTENCES, many_pages_pdf

pytestmark = pytest.mark.asyncio(loop_scope="module")

FIXTURES = Path(__file__).parents[1] / "fixtures"
SESSION = "integration-session"
HEADERS = {"X-Session-Id": SESSION}
OWNER = owner_id(SESSION)
FIRST_RUN_TIMEOUT_S = 600  # includes the one-time model download
READY_TARGET_S = 60  # PRD: 100-page PDF upload -> ready


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def api(settings, tmp_path_factory):
    upload_dir = str(tmp_path_factory.mktemp("uploads"))
    app_settings = settings.model_copy(
        update={"ingestion": settings.ingestion.model_copy(update={"upload_dir": upload_dir})}
    )
    app = create_app(app_settings)
    async with app.router.lifespan_context(app):
        container = app.state.container
        await init_database(container)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client, container


async def upload(client, path: Path):
    files = {"file": (path.name, path.read_bytes(), "application/pdf")}
    return await client.post("/documents", files=files, headers=HEADERS)


async def wait_ready(client, doc_id: str, timeout_s: float) -> float:
    start = time.perf_counter()
    while time.perf_counter() - start < timeout_s:
        body = (await client.get(f"/documents/{doc_id}", headers=HEADERS)).json()
        if body["status"] in ("ready", "failed"):
            assert body["status"] == "ready", body
            return time.perf_counter() - start
        await asyncio.sleep(0.5)
    raise AssertionError(f"{doc_id} not ready after {timeout_s}s")


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def text_doc(api):
    client, _ = api
    response = await upload(client, FIXTURES / "text.pdf")
    assert response.status_code == 202, response.json()
    doc_id = response.json()["doc_id"]
    seconds = await wait_ready(client, doc_id, FIRST_RUN_TIMEOUT_S)
    print(f"\ntext.pdf upload -> ready in {seconds:.1f}s (includes model load)")
    return doc_id


def chunks(container):
    return container.mongo.db[container.settings.chunks_collection]


async def test_chunk_count_matches_the_chunker(api, text_doc):
    client, container = api
    pages = extract_pages(FIXTURES / "text.pdf", text_doc)
    expected = chunk_pages(
        pages, container.settings.chunking, doc_id=text_doc, owner_id=OWNER, filename="text.pdf"
    )
    assert await chunks(container).count_documents({"doc_id": text_doc}) == len(expected)
    body = (await client.get(f"/documents/{text_doc}", headers=HEADERS)).json()
    assert body["chunk_count"] == len(expected)


async def test_vectors_are_binary_float32_of_the_configured_size(api, text_doc):
    _, container = api
    doc = await chunks(container).find_one({"doc_id": text_doc})
    assert isinstance(doc["embedding"], Binary) and doc["embedding"].subtype == 9
    assert len(doc["embedding"].as_vector().data) == container.settings.embedding.dim
    stats = await container.mongo.db.command("collStats", container.settings.chunks_collection)
    print(f"\nbytes per chunk (avgObjSize): {stats['avgObjSize']}")
    assert stats["avgObjSize"] < 8000  # an array of 768 doubles alone would be ~8.4 KB


async def test_filtered_search_finds_a_known_sentence(api, text_doc):
    _, container = api
    pipeline = [
        {
            "$search": {
                "index": container.settings.mongodb.text_index,
                "compound": {
                    "must": [{"phrase": {"query": "refund window is 30 days", "path": "text"}}],
                    "filter": [
                        {"equals": {"path": "owner_id", "value": OWNER}},
                        {"in": {"path": "doc_id", "value": [text_doc]}},
                    ],
                },
            }
        },
        {"$project": {"page_start": 1, "text": 1}},
    ]
    hits = await (await chunks(container).aggregate(pipeline)).to_list()
    assert hits and hits[0]["page_start"] <= 3 and KNOWN_SENTENCES[3] in hits[0]["text"]


async def test_reupload_is_a_conflict(api, text_doc):
    client, _ = api
    response = await upload(client, FIXTURES / "text.pdf")
    assert response.status_code == 409 and response.json()["code"] == "duplicate_document"


async def test_scanned_pdf_is_rejected(api):
    client, _ = api
    response = await upload(client, FIXTURES / "scanned.pdf")
    assert response.status_code == 422 and response.json()["code"] == "no_text_layer"


async def test_forced_retry_creates_no_duplicates(api, text_doc, tmp_path):
    _, container = api
    before = await chunks(container).count_documents({"doc_id": text_doc})
    job = IngestJob(
        doc_id=text_doc, owner_id=OWNER, filename="text.pdf", path=str(FIXTURES / "text.pdf")
    )
    await container.ingestion.ingest(job)
    assert await chunks(container).count_documents({"doc_id": text_doc}) == before


@pytest.mark.slow
async def test_100_page_pdf_is_ready_within_target(api, text_doc, tmp_path):
    client, _ = api
    path = tmp_path / "report100.pdf"
    many_pages_pdf(path, 100)
    response = await upload(client, path)
    assert response.status_code == 202, response.json()
    seconds = await wait_ready(client, response.json()["doc_id"], READY_TARGET_S * 3)
    print(f"\n100-page PDF upload -> ready in {seconds:.1f}s (target {READY_TARGET_S}s)")
    assert seconds < READY_TARGET_S


async def test_delete_removes_chunks_and_marks_deleted(api, text_doc):
    client, container = api
    assert (await client.delete(f"/documents/{text_doc}", headers=HEADERS)).status_code == 204
    assert await chunks(container).count_documents({"doc_id": text_doc}) == 0
    record = await container.repository.get(text_doc, OWNER)
    assert record.status == "deleted"
    listed = (await client.get("/documents", headers=HEADERS)).json()
    assert text_doc not in [d["doc_id"] for d in listed]
    assert (await client.delete(f"/documents/{text_doc}", headers=HEADERS)).status_code == 404
