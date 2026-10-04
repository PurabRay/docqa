"""GET /health with a fake container, so no database is needed."""

from fastapi.testclient import TestClient

from docqa.api.app import create_app
from docqa.ports.health import DatabaseHealth
from docqa.settings import load_settings


class FakeProbe:
    def __init__(self, health: DatabaseHealth) -> None:
        self.health = health

    async def check(self) -> DatabaseHealth:
        return self.health


class FakeContainer:
    def __init__(self, health: DatabaseHealth) -> None:
        self.settings = load_settings(env_file=None)
        self.health = FakeProbe(health)

    def start(self) -> None:
        pass

    async def close(self) -> None:
        pass


def get_health(health: DatabaseHealth):
    app = create_app(build=lambda _settings: FakeContainer(health))
    with TestClient(app) as client:
        return client.get("/health")


def test_healthy_when_reachable_and_indexes_ready():
    response = get_health(
        DatabaseHealth(
            reachable=True, search_indexes={"chunks_vector": "READY", "chunks_text": "READY"}
        )
    )
    assert response.status_code == 200
    body = response.json()
    assert body["mongodb"] == "ok" and len(body["config_hash"]) == 64


def test_503_when_an_index_is_not_ready():
    response = get_health(
        DatabaseHealth(
            reachable=True, search_indexes={"chunks_vector": "BUILDING", "chunks_text": "READY"}
        )
    )
    assert response.status_code == 503
    assert response.json()["search_indexes"]["chunks_vector"] == "BUILDING"


def test_503_when_database_unreachable():
    response = get_health(DatabaseHealth(reachable=False))
    assert response.status_code == 503
    assert response.json()["mongodb"] == "unreachable"
