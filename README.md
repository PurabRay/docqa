# DocQA

Question answering over your PDFs with page-level citations, built on MongoDB Atlas
(hybrid `$vectorSearch` + `$search` fused by `$rankFusion`) and free-tier LLMs.

- What and why: [docs/PRD.md](docs/PRD.md)
- How: [docs/DESIGN.md](docs/DESIGN.md)
- Free services used: [docs/FREE_STACK.md](docs/FREE_STACK.md)
- Rules for contributors and coding agents: [CLAUDE.md](CLAUDE.md) (same as AGENTS.md)

## Quick start

```bash
cp .env.example .env
uv sync
make up        # atlas-local MongoDB in Docker
make initdb    # collections + indexes, waits until search indexes are READY
make test      # unit tests
make test-int  # integration tests on atlas-local
make dev       # API on http://localhost:8000/health
```

Status: milestone 1 (scaffold, settings, domain, MongoDB adapters, /health).
