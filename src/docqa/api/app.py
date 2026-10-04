"""FastAPI application factory.

Run with: ``uvicorn docqa.api.app:create_app --factory``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from docqa.api.routers import health
from docqa.api.schemas import ErrorBody
from docqa.bootstrap import Container, build_container
from docqa.domain.errors import DocQAError
from docqa.settings import Settings, load_settings


def create_app(
    settings: Settings | None = None,
    build: Callable[[Settings], Container] = build_container,
) -> FastAPI:
    """Create the API.

    Args:
        settings: Settings to use; loaded from config and env when omitted.
        build: Builds the container; tests pass a fake.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        container = build(settings or load_settings())
        app.state.container = container
        try:
            yield
        finally:
            await container.close()

    app = FastAPI(title="DocQA", lifespan=lifespan)
    app.include_router(health.router)
    app.add_exception_handler(DocQAError, _docqa_error_handler)  # type: ignore[arg-type]
    return app


async def _docqa_error_handler(_: Request, err: DocQAError) -> JSONResponse:
    """Turn any DocQAError into its HTTP status with a {code, message} body."""
    body = ErrorBody(code=err.code, message=err.message)
    return JSONResponse(status_code=err.http_status, content=body.model_dump())
