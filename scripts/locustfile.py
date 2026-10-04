"""Locust load test: 8 users asking questions over SSE (PRD: Load test).

Stub run (pipeline capacity; start the API with DOCQA_PROFILE=loadtest), i.e. `make load`:
    uv run locust -f scripts/locustfile.py --headless -u 8 -r 8 -t 10m \
        --host http://localhost:8000 --mode stub
Real run (provider caps apply; start the API with the free profile):
    uv run locust -f scripts/locustfile.py --headless -u 2 -r 1 -t 5m \
        --host http://localhost:8000 --mode real

Reports first_token and done times separately, samples MongoDB opcounters every second
and writes eval/reports/load-<mode>.json with throughput, latencies, error rate and peak ops/s.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import gevent
from locust import HttpUser, constant, events, task
from pymongo import MongoClient

from docqa.settings import load_settings

FIXTURE = Path(__file__).parents[1] / "tests" / "fixtures" / "text.pdf"
REPORT_DIR = Path(__file__).parents[1] / "eval" / "reports"
SESSION = "load-test"
QUESTIONS = [
    "What is the refund window?",
    "How much paid leave do employees accrue?",
    "What was the annual revenue in FY2025?",
    "What does clause 14.3(b) say?",
]
OPCOUNTERS = ("insert", "query", "update", "delete", "getmore", "command")
STATE: dict[str, Any] = {"doc_ids": [], "peak_ops": 0.0, "stop": False}


@events.init_command_line_parser.add_listener
def add_arguments(parser: Any) -> None:
    """Add --mode (stub | real) to Locust's command line."""
    parser.add_argument(
        "--mode", choices=["stub", "real"], default="stub", help="label for the report"
    )


@events.test_start.add_listener
def upload_and_sample(environment: Any, **_: Any) -> None:
    """Upload the fixture once (shared session), wait for ready, start the ops/s sampler."""
    import httpx

    headers = {"X-Session-Id": SESSION}
    with httpx.Client(base_url=environment.host, timeout=120) as http:
        files = {"file": (FIXTURE.name, FIXTURE.read_bytes(), "application/pdf")}
        response = http.post("/documents", files=files, headers=headers)
        if response.status_code not in (202, 409):
            raise SystemExit(f"upload failed: {response.text}")
        for _ in range(240):
            docs = [
                d for d in http.get("/documents", headers=headers).json() if d["status"] == "ready"
            ]
            if docs:
                STATE["doc_ids"] = [d["doc_id"] for d in docs]
                break
            time.sleep(0.5)
    gevent.spawn(sample_opcounters)


def sample_opcounters() -> None:
    """Every second: total operations delta from serverStatus; keep the peak."""
    settings = load_settings()
    with MongoClient(settings.mongodb_uri()) as mongo:

        def total() -> int:
            counters = mongo.admin.command("serverStatus")["opcounters"]
            return sum(int(counters.get(n, 0)) for n in OPCOUNTERS)

        last = total()
        while not STATE["stop"]:
            gevent.sleep(1)
            now = total()
            STATE["peak_ops"] = max(STATE["peak_ops"], float(now - last))
            last = now


class AskUser(HttpUser):
    """Asks questions back to back and times the stream."""

    wait_time = constant(0)

    @task
    def ask(self) -> None:
        """One /query request, timed to the first token and to done."""
        question = QUESTIONS[int(time.time() * 1000) % len(QUESTIONS)]
        body = {"session_id": SESSION, "question": question, "doc_ids": STATE["doc_ids"]}
        start = time.perf_counter()
        with self.client.post(
            "/query", json=body, stream=True, catch_response=True, name="query"
        ) as response:
            first, done = None, False
            for line in response.iter_lines(decode_unicode=True):
                if line and line.startswith("event: token") and first is None:
                    first = (time.perf_counter() - start) * 1000
                elif line and line.startswith("event: done"):
                    done = True
            if not done:
                response.failure("stream ended without a done event")
                return
            response.success()
        elapsed = (time.perf_counter() - start) * 1000
        for name, value in (("first_token", first), ("done", elapsed)):
            if value is not None:
                events.request.fire(
                    request_type="SSE",
                    name=name,
                    response_time=value,
                    response_length=0,
                    exception=None,
                    context={},
                )


@events.test_stop.add_listener
def write_report(environment: Any, **_: Any) -> None:
    """Throughput, latencies, error rate and peak ops/s, as JSON for the README renderer."""
    STATE["stop"] = True
    stats = environment.stats
    query, done, first = (
        stats.get("query", "POST"),
        stats.get("done", "SSE"),
        stats.get("first_token", "SSE"),
    )
    report = {
        "mode": environment.parsed_options.mode,
        "users": environment.runner.user_count,
        "requests_per_s": query.total_rps,
        "error_rate": query.fail_ratio,
        "done_ms": {
            "p50": done.get_response_time_percentile(0.5),
            "p99": done.get_response_time_percentile(0.99),
        },
        "first_token_ms": {
            "p50": first.get_response_time_percentile(0.5),
            "p95": first.get_response_time_percentile(0.95),
        },
        "peak_ops_per_s": STATE["peak_ops"],
        "ops_cap": 100,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"load-{report['mode']}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
