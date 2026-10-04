"""ui/api_client.py against the real API (StubLLM, atlas-local): every call the UI makes."""

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from docqa.adapters.llm.stub import StubLLM
from docqa.api.app import create_app
from docqa.bootstrap import build_container, init_database
from tests.e2e.test_query_api import CitingStub
from tests.fixtures.make_fixtures import KNOWN_SENTENCES
from ui.api_client import ApiClient, ApiError

TEXT_PDF = Path(__file__).parents[1] / "fixtures" / "text.pdf"


@pytest.fixture(scope="module")
def api(settings, tmp_path_factory, test_database):
    upload_dir = str(tmp_path_factory.mktemp("ui-uploads"))
    app_settings = settings.model_copy(
        update={"ingestion": settings.ingestion.model_copy(update={"upload_dir": upload_dir})}
    )
    app = create_app(
        app_settings,
        build=lambda s: build_container(s, answer_llm=CitingStub(), text_llm=StubLLM("q?")),
    )
    with TestClient(app) as http:  # runs the lifespan (worker, Mongo client) in its own loop
        http.portal.call(init_database, app.state.container)
        yield ApiClient(http, session_id="ui-e2e")


def wait_ready(api, doc_id):
    for _ in range(600):
        status = next(d["status"] for d in api.list_documents() if d["doc_id"] == doc_id)
        if status in ("ready", "failed"):
            return status
        time.sleep(0.5)
    raise AssertionError("not ready")


def test_full_ui_flow(api):
    doc_id = api.upload("text.pdf", TEXT_PDF.read_bytes())["doc_id"]
    assert wait_ready(api, doc_id) == "ready"
    with pytest.raises(ApiError) as duplicate:
        api.upload("text.pdf", TEXT_PDF.read_bytes())
    assert duplicate.value.status == 409 and "already uploaded" in duplicate.value.message

    events = list(api.query("What is the refund window?", [doc_id]))
    names = [name for name, _ in events]
    assert names[0] == "meta" and names[-1] == "done" and "token" in names
    citation = dict(events)["citations"]["citations"][0]
    assert citation["page"] == 3

    found = api.highlights(doc_id, citation["page"], citation["quote"])
    assert found["rects"], "the cited quote should be highlighted on its page"
    assert api.file(doc_id).startswith(b"%PDF-")

    api.feedback(dict(events)["meta"]["trace_id"], -1, "wrong page")
    api.delete(doc_id)
    assert doc_id not in [d["doc_id"] for d in api.list_documents()]


def test_highlights_for_a_known_quote(api):
    doc_id = api.upload("text-copy.pdf", TEXT_PDF.read_bytes() + b"\n%copy")["doc_id"]
    assert wait_ready(api, doc_id) == "ready"
    assert api.highlights(doc_id, 3, KNOWN_SENTENCES[3])["rects"]
    assert api.highlights(doc_id, 4, KNOWN_SENTENCES[3])["rects"] == []
