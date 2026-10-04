"""Check the PRD's alert thresholds; exit 1 on any breach (for a scheduled job).

MongoDB: storage size (dbStats), ops/s (serverStatus opcounters sampled twice),
search index status ($listSearchIndexes). Langfuse (when LANGFUSE_* keys are set):
p95 latency, error rate and abstention rate over the last ``alerts.window_minutes``.

Usage: uv run python scripts/check_alerts.py
"""

from __future__ import annotations

import asyncio
import statistics
import sys
from datetime import UTC, datetime, timedelta
from typing import Any

from docqa.adapters.mongo.client import MongoConnection
from docqa.adapters.mongo.health import MongoHealthProbe
from docqa.adapters.mongo.storage_meter import MongoStorageMeter
from docqa.settings import Settings, load_settings

OPCOUNTERS = ("insert", "query", "update", "delete", "getmore", "command")


async def mongo_breaches(settings: Settings, mongo: MongoConnection) -> list[str]:
    """Storage, ops/s and search-index breaches."""
    breaches = []
    usage = await MongoStorageMeter(mongo.db).usage()
    if usage.size_mb > settings.mongodb.storage_cap_mb:
        breaches.append(f"storage {usage.size_mb:.0f} MB > {settings.mongodb.storage_cap_mb} MB")
    ops = await ops_per_second(mongo, settings.alerts.ops_sample_s)
    if ops > settings.alerts.max_ops_per_s:
        breaches.append(f"{ops:.0f} ops/s > {settings.alerts.max_ops_per_s}")
    names = [settings.mongodb.vector_index, settings.mongodb.text_index]
    health = await MongoHealthProbe(mongo, settings.chunks_collection, names).check()
    if not health.healthy:
        breaches.append(f"search indexes not READY: {health.search_indexes or 'unreachable'}")
    return breaches


async def ops_per_second(mongo: MongoConnection, sample_s: float) -> float:
    """Average operations per second over ``sample_s`` seconds."""

    async def total() -> int:
        counters = (await mongo.client.admin.command("serverStatus"))["opcounters"]
        return sum(int(counters.get(name, 0)) for name in OPCOUNTERS)

    before = await total()
    await asyncio.sleep(sample_s)
    return (await total() - before) / sample_s


def langfuse_breaches(settings: Settings) -> list[str]:
    """Latency, error and abstention breaches from recent traces (skipped without keys)."""
    public, secret = (
        settings.optional_secret("LANGFUSE_PUBLIC_KEY"),
        settings.optional_secret("LANGFUSE_SECRET_KEY"),
    )
    if not (public and secret):
        print("langfuse: skipped (LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set)")
        return []
    from langfuse import Langfuse

    client = Langfuse(
        public_key=public, secret_key=secret, host=settings.optional_secret("LANGFUSE_HOST")
    )
    since = datetime.now(UTC) - timedelta(minutes=settings.alerts.window_minutes)
    traces = [t for t in client.api.trace.list(name="query", from_timestamp=since, limit=100).data]
    return trace_breaches(traces, settings)


def trace_breaches(traces: list[Any], settings: Settings) -> list[str]:
    """Pure part of the Langfuse check, on objects with .latency (s) and .metadata."""
    if not traces:
        return []
    a, breaches = settings.alerts, []
    latencies = sorted(float(t.latency or 0) for t in traces)
    p95 = statistics.quantiles(latencies, n=20)[-1] if len(latencies) > 1 else latencies[0]
    outcomes = [str((t.metadata or {}).get("outcome", "answer")) for t in traces]
    errors = outcomes.count("error") / len(outcomes)
    abstained = outcomes.count("abstain") / len(outcomes)
    if p95 > a.p95_latency_s:
        breaches.append(f"p95 latency {p95:.1f}s > {a.p95_latency_s}s")
    if errors > a.error_rate:
        breaches.append(f"error rate {errors:.1%} > {a.error_rate:.0%}")
    if not a.abstention_min <= abstained <= a.abstention_max:
        breaches.append(
            f"abstention rate {abstained:.0%} outside {a.abstention_min:.0%}-{a.abstention_max:.0%}"
        )
    return breaches


async def main() -> int:
    """Print every breach; exit code 1 if there is any."""
    settings = load_settings()
    mongo = MongoConnection(settings.mongodb_uri(), settings.mongodb)
    try:
        breaches = await mongo_breaches(settings, mongo) + langfuse_breaches(settings)
    finally:
        await mongo.close()
    for breach in breaches:
        print(f"ALERT: {breach}")
    print("all checks passed" if not breaches else f"{len(breaches)} alert(s)")
    return 1 if breaches else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
