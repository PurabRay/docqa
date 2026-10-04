"""Integration fixtures: a throwaway database on the atlas-local container.

Start the container first with `make up`. MONGODB_URI defaults to the local
container; Atlas URIs are refused so tests never hit the free cluster.
"""

import os
import uuid

import pytest

from docqa.bootstrap import Container, build_container, init_database
from docqa.settings import Settings, load_settings

LOCAL_URI = "mongodb://localhost:27017/?directConnection=true"


@pytest.fixture(scope="session")
def settings() -> Settings:
    os.environ.setdefault("MONGODB_URI", LOCAL_URI)
    if "mongodb.net" in os.environ["MONGODB_URI"]:
        pytest.exit("Integration tests must run on atlas-local, not on Atlas.", returncode=2)
    database = f"docqa_test_{uuid.uuid4().hex[:8]}"
    return load_settings(overrides={"mongodb": {"database": database}}, env_file=None)


@pytest.fixture
async def container(settings: Settings):
    container = build_container(settings)
    yield container
    await container.close()


@pytest.fixture
async def initialized(container: Container) -> Container:
    """A container whose database has every collection and index READY."""
    await init_database(container)
    return container


@pytest.fixture(scope="session", autouse=True)
def drop_test_database(settings: Settings):
    yield
    from pymongo import MongoClient  # sync client is fine here: outside any event loop

    with MongoClient(settings.mongodb_uri()) as client:
        client.drop_database(settings.mongodb.database)
