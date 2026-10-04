"""FastAPI application factory.

Run with: ``uvicorn docqa.api.app:create_app --factory``.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from docqa.api.routers import documents, feedback, health, query
from docqa.api.schemas import ErrorBody
from docqa.bootstrap import Container, build_container
from docqa.domain.errors import DocQAError
from docqa.observability.logging import configure_logging
from docqa.settings import Settings, load_settings

log = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    build: Callable[[Settings], Container] = build_container,
) -> FastAPI:
    """Create the API.

    Args:
        settings: Settings to use; loaded from config and env when omitted.
        build: Builds the container; tests pass a fake or a container with a stub LLM.
    """
    configure_logging()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        container = build(settings or load_settings())
        app.state.container = container
        container.start()
        try:
            yield
        finally:
            await container.close()

    app = FastAPI(title="DocQA", lifespan=lifespan)
    for router in (health.router, documents.router, query.router, feedback.router):
        app.include_router(router)
    app.add_exception_handler(DocQAError, _docqa_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, _unexpected_error_handler)
    app.middleware("http")(_request_id)
    return app


async def _request_id(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Bind a request id to every log line of this request."""
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=uuid.uuid4().hex[:12])
    return await call_next(request)


async def _docqa_error_handler(_: Request, err: DocQAError) -> JSONResponse:
    """Turn any DocQAError into its HTTP status with a {code, message} body."""
    body = ErrorBody(code=err.code, message=err.message)
    return JSONResponse(status_code=err.http_status, content=body.model_dump())


async def _unexpected_error_handler(_: Request, err: Exception) -> JSONResponse:
    """A bug: log its type only, and never send a stack trace to the client."""
    log.error("request.crashed error=%s", type(err).__name__)
    body = ErrorBody(code=DocQAError.code, message=DocQAError().message)
    return JSONResponse(status_code=DocQAError.http_status, content=body.model_dump())
