"""Warm up before a demo: ping MongoDB, wake the deployed app, load the models.

Usage:
    uv run python -m scripts.warmup --url https://<app>     # demo morning
    uv run python -m scripts.warmup --mongo-only           # weekly keep-alive (backup.yml)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

import httpx

from docqa.adapters.mongo.client import MongoConnection
from docqa.domain.errors import DocQAError
from docqa.settings import load_settings

COLD_START_TIMEOUT_S = 90  # PRD: cold start after sleep <= 60 s, plus margin


async def ping_mongo() -> None:
    """One round trip, which also counts as activity for the free cluster's idle timer."""
    settings = load_settings()
    mongo = MongoConnection(settings.mongodb_uri(), settings.mongodb)
    try:
        await mongo.ping()
    finally:
        await mongo.close()
    print("mongodb: ok")


def wake_app(url: str) -> None:
    """Poll /health until the app answers 200 (a cold start can take a minute)."""
    start = time.monotonic()
    while time.monotonic() - start < COLD_START_TIMEOUT_S:
        try:
            if httpx.get(f"{url.rstrip('/')}/health", timeout=30).status_code == 200:
                print(f"app: healthy after {time.monotonic() - start:.0f}s")
                return
        except httpx.HTTPError:
            pass
        time.sleep(5)
    raise SystemExit(f"app: not healthy after {COLD_START_TIMEOUT_S}s")


def warm_models() -> None:
    """Download and load the embedding and re-ranking models once."""
    from docqa.bootstrap import build_container

    container = build_container(load_settings())
    container.embedder.embed_query("warm up")
    print("embedding model: loaded")


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Warm up DocQA before a demo.")
    parser.add_argument("--url", help="deployed app URL")
    parser.add_argument("--mongo-only", action="store_true")
    args = parser.parse_args()
    try:
        asyncio.run(ping_mongo())
    except DocQAError as err:
        print(f"mongodb: {err.message}")
        return 1
    if not args.mongo_only:
        warm_models()
        if args.url:
            wake_app(args.url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
