"""Create DocQA's collections, B-tree indexes and search indexes, then wait for READY.

Idempotent: running it twice changes nothing. Usage:

    uv run python scripts/init_db.py                   # uses MONGODB_URI
    MONGODB_URI=$MONGODB_URI_ATLAS make initdb         # same indexes on the Atlas free cluster
"""

from __future__ import annotations

import asyncio
import sys

from docqa.bootstrap import build_container, init_database
from docqa.domain.errors import DocQAError
from docqa.settings import load_settings


async def main() -> int:
    """Run the initialisation and print one line per search index."""
    settings = load_settings()
    container = build_container(settings)
    try:
        await container.mongo.ping()
        print(f"database={settings.mongodb.database} collection={settings.chunks_collection}")
        actions = await init_database(container)
        health = await container.health.check()
        for name, action in actions.items():
            print(f"search index {name}: {action}, status {health.search_indexes.get(name)}")
        return 0
    except DocQAError as err:
        print(f"init_db failed: [{err.code}] {err.message}", file=sys.stderr)
        return 1
    finally:
        await container.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
