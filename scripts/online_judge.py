"""Online judge: score a sample of the last day's answered queries; write scores to Langfuse.

Samples tracing.sample_judge_rate of yesterday's "query" traces, reads the scrubbed
question and answer from the turns collection (by trace id) and the scrubbed quotes from
the trace, asks the Mistral judge (prompts/judge/groundedness_v1.yaml) for groundedness
and citation validity, and attaches both as Langfuse scores.

Usage: uv run python -m scripts.online_judge
Needs LANGFUSE_*, MISTRAL_API_KEY and MONGODB_URI.
"""

from __future__ import annotations

import asyncio
import random
import sys
from datetime import UTC, datetime, timedelta

from docqa.adapters.mongo.client import MongoConnection
from docqa.adapters.mongo.names import TURNS
from docqa.settings import load_settings
from eval.metrics.judge import build_judge

LOOKBACK = timedelta(days=1)
PAGE_SIZE = 100


async def main() -> int:
    """Judge a sample of traces; print how many were scored."""
    settings = load_settings()
    judge = build_judge(settings, settings.eval.online_judge_prompt)
    public, secret = (
        settings.optional_secret("LANGFUSE_PUBLIC_KEY"),
        settings.optional_secret("LANGFUSE_SECRET_KEY"),
    )
    if judge is None or not (public and secret):
        print("online judge needs MISTRAL_API_KEY and LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY")
        return 1
    from langfuse import Langfuse

    langfuse = Langfuse(
        public_key=public, secret_key=secret, host=settings.optional_secret("LANGFUSE_HOST")
    )
    since = datetime.now(UTC) - LOOKBACK
    traces = langfuse.api.trace.list(name="query", from_timestamp=since, limit=PAGE_SIZE).data
    answered = [t for t in traces if (t.metadata or {}).get("outcome") == "answer"]
    sample = random.sample(answered, k=round(len(answered) * settings.tracing.sample_judge_rate))
    mongo = MongoConnection(settings.mongodb_uri(), settings.mongodb)
    scored = 0
    try:
        for trace in sample:
            turns = {
                t["role"]: t["content_scrubbed"]
                async for t in mongo.db[TURNS].find({"trace_id": trace.id})
            }
            reply = await judge.ask(
                question=turns.get("user", ""),
                answer=turns.get("assistant", ""),
                quotes=str((trace.metadata or {}).get("quotes", "")),
            )
            if reply is None:
                continue
            for name in ("groundedness", "citation_validity"):
                if isinstance(reply.get(name), int | float):
                    langfuse.create_score(
                        trace_id=trace.id, name=f"judge_{name}", value=float(reply[name])
                    )
            scored += 1
    finally:
        await mongo.close()
        langfuse.flush()
    print(f"scored {scored} of {len(sample)} sampled traces ({len(answered)} answered yesterday)")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
