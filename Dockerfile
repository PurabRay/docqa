# One image runs either the API (default) or the UI (override the command).
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PATH="/app/.venv/bin:$PATH"

# Dependencies first so code changes do not reinstall them.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY config ./config
COPY scripts ./scripts
COPY ui ./ui
RUN uv sync --frozen --no-dev

EXPOSE 8000 8501
CMD ["uvicorn", "docqa.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
