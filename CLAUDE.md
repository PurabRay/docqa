# DocQA: rules for coding agents

DocQA is a retrieval-augmented question-answering service over user-uploaded PDFs. A user
uploads text PDFs; DocQA extracts, chunks and embeds them into MongoDB (Atlas free cluster
for the demo, the `mongodb/mongodb-atlas-local` container for dev and CI), answers questions
with hybrid `$vectorSearch` + `$search` retrieval fused by `$rankFusion`, re-ranks with a
small cross-encoder, and streams a short answer whose every claim cites a verified page, or
abstains when the documents do not contain the answer. It is one Python service in
ports-and-adapters style, running on free tiers only.

`AGENTS.md` is an identical copy of this file for other agents; a unit test keeps them equal.

## Rules

1. Source of truth: `docs/PRD.md` (what), `docs/DESIGN.md` (how), `docs/FREE_STACK.md` (free services). If code and docs disagree, stop and ask. Never edit `docs/` unless asked.
2. Follow ports and adapters exactly as in DESIGN.md: api -> services -> domain + ports. Nothing in `domain/`, `ports/` or `services/` imports FastAPI, PyMongo, httpx, openai, fastembed or any vendor SDK.
3. Keep one composition root: `bootstrap.py` builds every object, including the single `AsyncMongoClient`. Only `settings.py` reads environment variables.
4. Put config over code: models, k values, thresholds, `$rankFusion` weights, index definitions and prompt versions live in `config/` and `prompts/`. No magic numbers.
5. MongoDB: use the PyMongo Async API only (never Motor, never the sync client in the event loop). Store vectors as BSON binary float32 via `Binary.from_vector`. Only `adapters/mongo/codecs.py` converts Pydantic <-> BSON. Build every search pipeline from an `AccessFilter` and pre-filter `owner_id` and `doc_id` in BOTH `$vectorSearch` and `$search`; a pipeline without it raises. Never put `$project` inside `$rankFusion`. Respect the free-cluster limits (512 MB, 100 ops/s, 3 search indexes).
6. Use Pydantic at every boundary; Python 3.11+, full type hints, mypy strict on `domain/`, `ports/`, `services/`.
7. Adapters translate vendor exceptions into `domain/errors.py` types; services never catch bare `Exception`; only API exception handlers produce HTTP responses.
8. Never log raw questions, document text, API keys or `MONGODB_URI`; scrub PII before logging or writing turns. Never invent citations.
9. Ship every change with tests. Unit tests need no network or Docker. Integration tests use the `mongodb/mongodb-atlas-local` container, never the Atlas free cluster. A bug fix starts with a failing test.
10. Make ruff + mypy + pytest pass before calling a task done; keep functions under ~40 lines and files under ~200; write Google-style docstrings; use Conventional Commits.
11. Never write, generate or edit questions in `eval/golden.jsonl`.
12. Plan first and list the files you will touch; finish by reporting what you built, the exact commands you ran and their output, and what is left. Never claim a check you did not run.

Keep code small and plain: a human reviews every line. Prefer fewer, readable lines over
abstractions, and do not add code for a later milestone until that milestone starts.

## Commands

| Command | What it does |
|---|---|
| `make dev` | Run the API locally with auto-reload |
| `make up` | Start the `mongodb` (atlas-local) container and wait until healthy |
| `make down` | Stop the containers |
| `make initdb` | Create collections, B-tree and search indexes; wait for READY |
| `make test` | Unit tests (no network, no Docker) |
| `make test-int` | Integration tests against atlas-local (`make up` first) |
| `make test-e2e` | End-to-end tests (from M2) |
| `make lint` | `ruff check` + `ruff format --check` |
| `make typecheck` | mypy |
| `make eval` | Golden-set evaluation (from M4) |
| `make load` | Locust load test (from M5) |

Without `make` (e.g. on Windows), run the `uv run ...` line from the `Makefile` directly.
