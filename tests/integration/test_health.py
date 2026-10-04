from fastapi.testclient import TestClient

from docqa.api.app import create_app


async def test_health_is_200_after_init_db(initialized, settings):
    # TestClient runs the app (and its own Mongo client) in a separate event loop.
    with TestClient(create_app(settings)) as client:
        response = client.get("/health")

    assert response.status_code == 200, response.json()
    body = response.json()
    assert body["mongodb"] == "ok"
    assert body["search_indexes"] == {"chunks_vector": "READY", "chunks_text": "READY"}
