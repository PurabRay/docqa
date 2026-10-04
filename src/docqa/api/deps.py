"""FastAPI dependencies: hand routers the objects built by bootstrap."""

from __future__ import annotations

from fastapi import Request

from docqa.bootstrap import Container


def get_container(request: Request) -> Container:
    """Return the container created in the app lifespan."""
    container: Container = request.app.state.container
    return container
