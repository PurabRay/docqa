"""Every PRD use case over HTTP, then delete (FR-12). Real pipeline, stub LLM, atlas-local.

The stub cites every prompt chunk that contains one of the known facts, like a model
that answers only from the documents.
"""

import asyncio
import json
import re
import time
from pathlib import Path

import httpx
import pytest
import pytest_asyncio

from docqa.adapters.llm.stub import StubLLM
from docqa.api.app import create_app
from docqa.bootstrap import build_container, init_database
from docqa.ports.llm import LLMDelta
from tests.e2e.test_query_api import RewriteStub, parse_sse
from tests.fixtures.make_fixtures import KNOWN_SENTENCES

pytestmark = pytest.mark.asyncio(loop_scope="module")

FIXTURES = Path(__file__).parents[1] / "fixtures"
SESSION = "full-flow"
HEADERS = {"X-Session-Id": SESSION}
# fact -> the word a question must contain for that fact to answer it
FACTS = {KNOWN_SENTENCES[3]: "refund", KNOWN_SENTENCES[11]: "leave", "|Research|63|": "research"}
BLOCK = re.compile(r"\[BEGIN DOCUMENT id=(\S+) doc=\S+ page=\S+?\]\n(.*?)\n\[END DOCUMENT\]", re.S)
DELETE_DEADLINE_S = 60  # FR-12


class FactStub:
    """Cites the known facts the question asks about and the prompt contains; else insufficient."""

    name = "fact-stub"

    async def stream(self, messages, *, max_tokens=512):
        prompt = messages[-1].content
        question = prompt.rsplit("Question:", 1)[-1].lower()
        cites = [
            {"chunk_id": cid, "quote": fact}
            for fact, word in FACTS.items()
            if word in question
            for cid, text in BLOCK.findall(prompt)
            if fact in text
        ][:5]
        reply = {
            "answer": "From the documents.",
            "sufficient_context": bool(cites),
            "citations": cites,
        }
        async for delta in StubLLM(json.dumps(reply), name=self.name).stream(messages):
            yield LLMDelta(**{**delta.model_dump(), "provider": self.name})


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def api(settings, tmp_path_factory, test_database):
    upload_dir = str(tmp_path_factory.mktemp("full-flow"))
    # The abstain threshold is not calibrated yet (needs the golden set), so this app never
    # abstains on low relevance; unanswerable questions abstain via insufficient_context.
    app_settings = settings.model_copy(
        update={
            "ingestion": settings.ingestion.model_copy(update={"upload_dir": upload_dir}),
            "retrieval": settings.retrieval.model_copy(update={"abstain_threshold": -1e9}),
        }
    )
    app = create_app(
        app_settings,
        build=lambda s: build_container(s, answer_llm=FactStub(), text_llm=RewriteStub()),
    )
    async with app.router.lifespan_context(app):
        await init_database(app.state.container)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://e2e", timeout=60
        ) as c:
            docs = {}
            for name in ("text.pdf", "tables.pdf"):
                files = {"file": (name, (FIXTURES / name).read_bytes(), "application/pdf")}
                docs[name] = (await c.post("/documents", files=files, headers=HEADERS)).json()[
                    "doc_id"
                ]
            for doc_id in docs.values():
                for _ in range(600):
                    if (await c.get(f"/documents/{doc_id}", headers=HEADERS)).json()[
                        "status"
                    ] == "ready":
                        break
                    await asyncio.sleep(0.5)
            yield c, app.state.container, docs


async def ask(c, question, doc_ids):
    response = await c.post(
        "/query", json={"session_id": SESSION, "question": question, "doc_ids": doc_ids}
    )
    return response, parse_sse(response.text) if response.status_code == 200 else []


def cited(events):
    return {
        (c["doc_name"], c["page"]) for c in dict(events).get("citations", {}).get("citations", [])
    }


async def test_factual_question_cites_the_right_page(api):
    c, _, docs = api
    _, events = await ask(c, "What is the refund window?", [docs["text.pdf"]])
    assert ("text.pdf", 3) in cited(events)


async def test_table_lookup_cites_the_table_page(api):
    c, _, docs = api
    _, events = await ask(c, "How many staff does the Research team have?", [docs["tables.pdf"]])
    assert ("tables.pdf", 1) in cited(events)


async def test_multi_document_question_cites_both(api):
    c, _, docs = api
    _, events = await ask(c, "Refund window and research team headcount?", list(docs.values()))
    assert {name for name, _ in cited(events)} == {"text.pdf", "tables.pdf"}


async def test_follow_up_is_rewritten(api):
    c, _, docs = api
    await ask(c, "What is the refund window?", [docs["text.pdf"]])
    _, events = await ask(c, "And for international orders?", [docs["text.pdf"]])
    assert dict(events)["meta"]["rewritten_question"]


async def test_unanswerable_question_abstains(api):
    c, _, docs = api
    _, events = await ask(c, "Who won the 1998 chess olympiad?", [docs["text.pdf"]])
    assert "abstain" in [name for name, _ in events]


async def test_delete_removes_chunks_cache_and_access_within_60s(api):
    c, container, docs = api
    doc_id = docs["text.pdf"]
    await ask(c, "How much paid leave do employees accrue?", [doc_id])  # cache an answer
    start = time.monotonic()
    assert (await c.delete(f"/documents/{doc_id}", headers=HEADERS)).status_code == 204
    chunks = container.mongo.db[container.settings.chunks_collection]
    assert await chunks.count_documents({"doc_id": doc_id}) == 0
    assert await container.cache.purge_document(doc_id) == 0  # DELETE already purged it
    response, _ = await ask(c, "How much paid leave do employees accrue?", [doc_id])
    assert response.status_code == 400 and response.json()["code"] == "document_not_ready"
    assert time.monotonic() - start < DELETE_DEADLINE_S
