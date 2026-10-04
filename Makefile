.PHONY: dev up down initdb test test-int test-e2e lint typecheck eval load

dev:
	uv run uvicorn docqa.api.app:create_app --factory --reload --port 8000

up:
	docker compose up -d --wait mongodb

down:
	docker compose down

initdb:
	uv run python scripts/init_db.py

test:
	uv run pytest -m unit

test-int:
	uv run pytest -m integration

test-e2e:
	uv run pytest -m e2e

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy

eval:
	uv run python -m eval.run_eval --config free --split dev --out eval/reports

load:
	uv run locust -f scripts/locustfile.py --headless -u 8 -r 8 -t 10m --host http://localhost:8000 --mode stub
