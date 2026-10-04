from docqa.bootstrap import init_database


async def test_init_db_builds_everything_and_indexes_reach_ready(container):
    await init_database(container)

    db = container.mongo.db
    settings = container.settings
    names = set(await db.list_collection_names())
    assert {"documents", "sessions", "turns", "feedback", settings.chunks_collection} <= names

    doc_indexes = await db["documents"].index_information()
    unique = [i for i in doc_indexes.values() if i.get("unique")]
    assert [i["key"] for i in unique] == [[("owner_id", 1), ("sha256", 1)]]

    health = await container.health.check()
    assert health.search_indexes == {"chunks_vector": "READY", "chunks_text": "READY"}


async def test_init_db_twice_changes_nothing(container):
    await init_database(container)
    db = container.mongo.db
    before = await db["documents"].index_information()

    actions = await init_database(container)

    assert actions == {"chunks_text": "unchanged", "chunks_vector": "unchanged"}
    assert await db["documents"].index_information() == before
