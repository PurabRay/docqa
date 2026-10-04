"""End to end over HTTP: upload -> ready -> POST /query (SSE) -> cache -> feedback.

Real FastAPI app, worker, embeddings, re-ranker and atlas-local; the LLM is a stub
that cites the retrieved chunk holding the known sentence.
"""

import asyncio
import json
import re
from pathlib import Path

import httpx
import pytest
import pytest_asyncio

from docqa.adapters.llm.stub import StubLLM
from docqa.api.app import create_app
from docqa.bootstrap import build_container, init_database
from docqa.ports.llm import LLMDelta
from tests.fixtures.make_fixtures import KNOWN_SENTENCES

pytestmark = pytest.mark.asyncio(loop_scope="module")

FIXTURES = Path(__file__).parents[1] / "fixtures"
SESSION = "e2e-session"
REFUND = KNOWN_SENTENCES[3]
BLOCK = re.compile(r"\[BEGIN DOCUMENT id=(\S+) doc=\S+ page=\S+?\]\n(.*?)\n\[END DOCUMENT\]", re.S)
REWRITTEN = "What is the refund window for international orders?"


class RewriteStub:
    """Rewrites only the international-orders follow-up; echoes any other question."""

    name = "rewrite-stub"

    async def complete(self, messages, **kwargs):
        question = messages[-1].content.rsplit("Follow-up question: ", 1)[-1].strip()
        text = REWRITTEN if "international" in question else question
        return await StubLLM(text, name=self.name).complete(messages)


class CitingStub:
    """Answers from the prompt like a well-behaved model: cites the refund chunk if present."""

    name = "citing-stub"

    async def stream(self, messages, *, max_tokens=512):
        blocks = BLOCK.findall(messages[-1].content)
        hit = next((cid for cid, text in blocks if REFUND in text), None)
        reply = {
            "answer": "Customers have 30 days.",
            "sufficient_context": hit is not None,
            "citations": [{"chunk_id": hit, "quote": REFUND}] if hit else [],
        }
        async for delta in StubLLM(json.dumps(reply), name=self.name).stream(messages):
            yield LLMDelta(**{**delta.model_dump(), "provider": self.name})


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def client(settings, tmp_path_factory, test_database):
    upload_dir = str(tmp_path_factory.mktemp("uploads"))
    app_settings = settings.model_copy(
        update={"ingestion": settings.ingestion.model_copy(update={"upload_dir": upload_dir})}
    )
    build = lambda s: build_container(s, answer_llm=CitingStub(), text_llm=RewriteStub())  # noqa: E731
    app = create_app(app_settings, build=build)
    async with app.router.lifespan_context(app):
        await init_database(app.state.container)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://e2e", timeout=60
        ) as c:
            yield c, app.state.container


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def doc_id(client):
    http, _ = client
    files = {"file": ("text.pdf", (FIXTURES / "text.pdf").read_bytes(), "application/pdf")}
    response = await http.post("/documents", files=files, headers={"X-Session-Id": SESSION})
    assert response.status_code == 202, response.json()
    doc = response.json()["doc_id"]
    for _ in range(600):
        status = (await http.get(f"/documents/{doc}", headers={"X-Session-Id": SESSION})).json()[
            "status"
        ]
        if status in ("ready", "failed"):
            assert status == "ready"
            return doc
        await asyncio.sleep(0.5)
    raise AssertionError("document never became ready")


def parse_sse(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.replace("\r\n", "\n").split("\n\n"):
        lines = dict(
            line.split(": ", 1)
            for line in block.splitlines()
            if ": " in line and not line.startswith(":")
        )
        if "event" in lines:
            events.append((lines["event"], json.loads(lines.get("data", "{}"))))
    return events


async def ask(http, doc, question, session=SESSION):
    response = await http.post(
        "/query", json={"session_id": session, "question": question, "doc_ids": [doc]}
    )
    assert response.status_code == 200, response.text
    return parse_sse(response.text)


async def test_query_streams_meta_tokens_citations_done(client, doc_id):
    http, _ = client
    events = await ask(http, doc_id, "What is the refund window?")
    names = [name for name, _ in events]
    assert names[0] == "meta" and names[-1] == "done" and "token" in names
    citations = dict(events)["citations"]["citations"]
    assert [(c["doc_name"], c["page"]) for c in citations] == [("text.pdf", 3)]
    assert dict(events)["done"]["cached"] is False


async def test_repeat_query_is_cached(client, doc_id):
    http, _ = client
    await ask(http, doc_id, "How long is the refund window?")
    events = await ask(http, doc_id, "How long is the refund window?")
    assert dict(events)["done"]["cached"] is True


async def test_absent_topic_abstains(client, doc_id):
    http, _ = client
    events = await ask(http, doc_id, "What is the boiling point of tungsten on Mars?")
    assert "abstain" in [name for name, _ in events]


async def test_feedback_is_stored(client, doc_id):
    http, container = client
    trace_id = dict(await ask(http, doc_id, "What is the refund window?"))["meta"]["trace_id"]
    response = await http.post(
        "/feedback",
        json={"trace_id": trace_id, "session_id": SESSION, "rating": -1, "comment": "wrong page"},
    )
    assert response.status_code == 204
    stored = await container.mongo.db["feedback"].find_one({"trace_id": trace_id})
    assert stored["rating"] == -1


async def test_follow_up_reports_the_rewritten_question(client, doc_id):
    http, _ = client
    await ask(http, doc_id, "What is the refund window?")
    events = await ask(http, doc_id, "And for international orders?")
    assert dict(events)["meta"]["rewritten_question"] == REWRITTEN


async def test_bad_requests_are_http_errors_not_streams(client, doc_id):
    http, _ = client
    too_long = await http.post(
        "/query", json={"session_id": "e2e-bad", "question": "x" * 501, "doc_ids": [doc_id]}
    )
    assert too_long.status_code == 400 and too_long.json()["code"] == "input_rejected"
    unknown = await http.post(
        "/query", json={"session_id": "e2e-bad", "question": "hi", "doc_ids": ["nope"]}
    )
    assert unknown.status_code == 400 and unknown.json()["code"] == "document_not_ready"
