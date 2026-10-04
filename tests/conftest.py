"""Shared test setup.

* Each test is marked by its folder, so `pytest -m unit` runs tests/unit and so on.
* Mongo fixtures (settings, container, initialized) give integration and contract
  tests a throwaway database on the atlas-local container. They connect only when a
  test asks for them. MONGODB_URI defaults to the local container; Atlas URIs are
  refused so tests never hit the free cluster.
"""

import os
import uuid
from pathlib import Path

import pytest

from docqa.bootstrap import Container, build_container, init_database
from docqa.settings import Settings, load_settings

FOLDER_MARKERS = {"unit", "integration", "e2e", "contract", "live"}
LOCAL_URI = "mongodb://localhost:27017/?directConnection=true"


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        folder = Path(item.fspath).parent.name
        if folder in FOLDER_MARKERS:
            item.add_marker(folder)


@pytest.fixture(scope="session")
def settings():
    os.environ.setdefault("MONGODB_URI", LOCAL_URI)
    if "mongodb.net" in os.environ["MONGODB_URI"]:
        pytest.exit("Integration tests must run on atlas-local, not on Atlas.", returncode=2)
    database = f"docqa_test_{uuid.uuid4().hex[:8]}"
    return load_settings(overrides={"mongodb": {"database": database}}, env_file=None)


@pytest.fixture(scope="session")
def test_database(settings: Settings):
    """Drops the throwaway database after the run; only Mongo tests pull this in."""
    yield settings.mongodb.database
    from pymongo import MongoClient  # sync client is fine here: outside any event loop

    with MongoClient(settings.mongodb_uri()) as client:
        client.drop_database(settings.mongodb.database)


@pytest.fixture
async def container(settings: Settings, test_database: str):
    container = build_container(settings)
    yield container
    await container.close()


@pytest.fixture
async def initialized(container: Container) -> Container:
    """A container whose database has every collection and index READY."""
    await init_database(container)
    return container
