"""FastAPI dependencies: hand routers the objects built by bootstrap."""

from __future__ import annotations

import hashlib
from typing import Annotated

from fastapi import Header, Request

from docqa.bootstrap import Container


def get_container(request: Request) -> Container:
    """Return the container created in the app lifespan."""
    container: Container = request.app.state.container
    return container


def owner_id(session_id: Annotated[str, Header(alias="X-Session-Id", min_length=1)]) -> str:
    """v1 has no accounts: the owner is a hash of the session id."""
    return hashlib.sha256(session_id.encode()).hexdigest()[:32]
