"""Failure modes from the PRD risks table: database down, every LLM down, storage cap,
missing search index."""

import asyncio
import time
import uuid
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from docqa.api.app import create_app
from docqa.bootstrap import build_container, init_database
from docqa.settings import Secrets, Settings
from tests.unit.test_query_service import Unavailable

FIXTURES = Path(__file__).parents[1] / "fixtures"
HEADERS = {"X-Session-Id": "failures"}
FAIL_FAST_S = 5  # PRD: /query fails fast when MongoDB is unreachable


def variant(settings: Settings, tmp_path: Path, **mongodb) -> Settings:
    """Same settings with a fresh database name and mongodb overrides."""
    mongo = settings.mongodb.model_copy(
        update={"database": f"docqa_fail_{uuid.uuid4().hex[:6]}", **mongodb}
    )
    ingestion = settings.ingestion.model_copy(update={"upload_dir": str(tmp_path)})
    return settings.model_copy(update={"mongodb": mongo, "ingestion": ingestion})


async def client_for(app):
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://e2e", timeout=60
    )


async def test_mongo_down_health_503_and_query_fails_fast(settings, tmp_path):
    down = variant(settings, tmp_path, server_selection_timeout_ms=300).model_copy(
        update={
            "secrets": Secrets(
                _env_file=None,
                mongodb_uri=SecretStr("mongodb://127.0.0.1:1/?directConnection=true"),
            )
        }
    )
    app = create_app(down)
    async with app.router.lifespan_context(app), await client_for(app) as c:
        assert (await c.get("/health")).status_code == 503
        start = time.monotonic()
        response = await c.post(
            "/query", json={"session_id": "s", "question": "hi", "doc_ids": ["d"]}
        )
        assert response.status_code == 503 and response.json()["code"] == "database_unavailable"
        assert time.monotonic() - start < FAIL_FAST_S


async def test_missing_search_index_makes_health_503(settings, tmp_path, test_database):
    app = create_app(variant(settings, tmp_path))  # fresh database, init_database never ran
    async with app.router.lifespan_context(app), await client_for(app) as c:
        response = await c.get("/health")
        assert response.status_code == 503
        assert set(response.json()["search_indexes"].values()) == {"MISSING"}


@pytest.mark.asyncio(loop_scope="function")
async def test_all_llms_down_degrades_and_storage_cap_returns_507(
    settings, tmp_path, test_database
):
    capped = variant(settings, tmp_path, max_documents=1)
    app = create_app(capped, build=lambda s: build_container(s, answer_llm=Unavailable()))
    async with app.router.lifespan_context(app), await client_for(app) as c:
        await init_database(app.state.container)
        files = {"file": ("text.pdf", (FIXTURES / "text.pdf").read_bytes(), "application/pdf")}
        doc_id = (await c.post("/documents", files=files, headers=HEADERS)).json()["doc_id"]
        for _ in range(600):
            if (await c.get(f"/documents/{doc_id}", headers=HEADERS)).json()["status"] == "ready":
                break
            await asyncio.sleep(0.5)
        body = (
            await c.post(
                "/query",
                json={
                    "session_id": "failures",
                    "question": "What is the refund window?",
                    "doc_ids": [doc_id],
                },
            )
        ).text
        assert "event: degraded" in body and "passages" in body

        more = {"file": ("tables.pdf", (FIXTURES / "tables.pdf").read_bytes(), "application/pdf")}
        response = await c.post("/documents", files=more, headers=HEADERS)
        assert response.status_code == 507 and response.json()["code"] == "storage_cap_reached"
