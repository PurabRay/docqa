"""Create collections, B-tree indexes and search indexes from config. Safe to re-run.

Search indexes are built asynchronously by MongoDB, so after creating or
updating them we poll ``$listSearchIndexes`` until each one is READY.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterable, Mapping
from typing import Any, Literal

from pymongo.asynchronous.collection import AsyncCollection
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.operations import SearchIndexModel

from docqa.adapters.mongo.client import Document, translate_errors
from docqa.adapters.mongo.names import CHUNKS_ALIAS, DOCUMENTS, FEEDBACK, SESSIONS, TURNS
from docqa.config_schema import BTreeIndex, SearchIndex
from docqa.domain.errors import SearchIndexNotReadyError
from docqa.domain.models import IngestionStatus

IndexAction = Literal["created", "updated", "unchanged"]


def documents_validator() -> Document:
    """JSON Schema that rejects a document record with an unknown status."""
    return {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["owner_id", "sha256", "status"],
            "properties": {"status": {"enum": [status.value for status in IngestionStatus]}},
        }
    }


def definition_matches(desired: Any, actual: Any) -> bool:
    """True if every key in ``desired`` has the same value in ``actual``.

    MongoDB may add defaults to a stored definition, so extra keys in
    ``actual`` are ignored; lists must match element by element.
    """
    if isinstance(desired, Mapping):
        return isinstance(actual, Mapping) and all(
            key in actual and definition_matches(value, actual[key])
            for key, value in desired.items()
        )
    if isinstance(desired, list):
        return (
            isinstance(actual, list)
            and len(desired) == len(actual)
            and all(definition_matches(d, a) for d, a in zip(desired, actual, strict=True))
        )
    return bool(desired == actual)


async def ensure_collections(db: AsyncDatabase[Document], chunks_collection: str) -> None:
    """Create any missing collection and (re)apply the documents status validator."""
    with translate_errors():
        existing = set(await db.list_collection_names())
        for name in (DOCUMENTS, SESSIONS, TURNS, FEEDBACK, chunks_collection):
            if name not in existing:
                await db.create_collection(name)
        await db.command("collMod", DOCUMENTS, validator=documents_validator())


async def ensure_btree_indexes(
    db: AsyncDatabase[Document], specs: Mapping[str, list[BTreeIndex]], chunks_collection: str
) -> None:
    """Create the B-tree indexes from config/indexes/btree.yaml (a no-op if present)."""
    with translate_errors():
        for name, indexes in specs.items():
            collection = db[chunks_collection if name == CHUNKS_ALIAS else name]
            for index in indexes:
                await collection.create_index(list(index.keys.items()), unique=index.unique)


async def list_search_indexes(collection: AsyncCollection[Document]) -> dict[str, Document]:
    """Return the collection's search indexes keyed by name."""
    with translate_errors():
        cursor = await collection.list_search_indexes()
        return {index["name"]: dict(index) for index in await cursor.to_list()}


async def ensure_search_indexes(
    collection: AsyncCollection[Document], indexes: Iterable[SearchIndex]
) -> dict[str, IndexAction]:
    """Create missing search indexes and update changed ones; leave equal ones alone."""
    existing = await list_search_indexes(collection)
    actions: dict[str, IndexAction] = {}
    with translate_errors():
        for index in indexes:
            current = existing.get(index.name)
            if current is None:
                model = SearchIndexModel(
                    definition=index.definition, name=index.name, type=index.type
                )
                await collection.create_search_index(model)
                actions[index.name] = "created"
            elif definition_matches(index.definition, current.get("latestDefinition")):
                actions[index.name] = "unchanged"
            else:
                await collection.update_search_index(index.name, index.definition)
                actions[index.name] = "updated"
    return actions


def _pending(indexes: Iterable[SearchIndex], existing: Mapping[str, Document]) -> dict[str, str]:
    """Map each index that is not READY with the desired definition to its status."""
    pending = {}
    for index in indexes:
        current = existing.get(index.name)
        if current is None:
            pending[index.name] = "MISSING"
        elif current.get("status") != "READY" or not definition_matches(
            index.definition, current.get("latestDefinition")
        ):
            pending[index.name] = str(current.get("status", "UNKNOWN"))
    return pending


async def wait_until_ready(
    collection: AsyncCollection[Document],
    indexes: list[SearchIndex],
    timeout_s: float,
    poll_interval_s: float,
) -> None:
    """Poll until every search index is READY.

    Raises:
        SearchIndexNotReadyError: If an index FAILED or the timeout passed.
    """
    deadline = time.monotonic() + timeout_s
    while pending := _pending(indexes, await list_search_indexes(collection)):
        if "FAILED" in pending.values():
            raise SearchIndexNotReadyError(f"Search index build FAILED: {pending}")
        if time.monotonic() > deadline:
            raise SearchIndexNotReadyError(
                f"Search indexes not READY after {timeout_s:.0f}s: {pending}"
            )
        await asyncio.sleep(poll_interval_s)


async def initialize_database(
    db: AsyncDatabase[Document],
    *,
    chunks_collection: str,
    btree: Mapping[str, list[BTreeIndex]],
    search_indexes: list[SearchIndex],
    timeout_s: float,
    poll_interval_s: float,
) -> dict[str, IndexAction]:
    """Build every collection and index DocQA needs and wait for search indexes.

    Returns:
        What happened to each search index: created, updated or unchanged.
    """
    await ensure_collections(db, chunks_collection)
    await ensure_btree_indexes(db, btree, chunks_collection)
    chunks = db[chunks_collection]
    actions = await ensure_search_indexes(chunks, search_indexes)
    await wait_until_ready(chunks, search_indexes, timeout_s, poll_interval_s)
    return actions
