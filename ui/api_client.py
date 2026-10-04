"""The only place the UI talks HTTP. Every call is one API request."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx

# Readable messages for the upload errors in DESIGN.md's REST table.
UPLOAD_ERRORS = {
    409: "You already uploaded this file.",
    413: "The file is too large (limit 25 MB, 300 pages).",
    415: "Only PDF files are supported.",
    422: "This PDF can't be used: it is scanned, encrypted or damaged.",
    507: "Storage is full; delete a document first.",
}


class ApiError(Exception):
    """An error response from the API, with its status, code and message."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


class ApiClient:
    """Calls the DocQA API for one browser session.

    Args:
        http: An httpx.Client with base_url set (tests pass FastAPI's TestClient).
        session_id: Sent as X-Session-Id; the API derives the owner from it.
    """

    def __init__(self, http: httpx.Client, session_id: str) -> None:
        self._http = http
        self.session_id = session_id
        self._headers = {"X-Session-Id": session_id}

    def upload(self, filename: str, data: bytes) -> dict[str, Any]:
        """POST /documents -> {doc_id, status}."""
        files = {"file": (filename, data, "application/pdf")}
        return self._json(self._http.post("/documents", files=files, headers=self._headers))

    def list_documents(self) -> list[dict[str, Any]]:
        """GET /documents."""
        documents: list[dict[str, Any]] = self._json(
            self._http.get("/documents", headers=self._headers)
        )
        return documents

    def delete(self, doc_id: str) -> None:
        """DELETE /documents/{id}."""
        self._check(self._http.delete(f"/documents/{doc_id}", headers=self._headers))

    def file(self, doc_id: str) -> bytes:
        """GET /documents/{id}/file -> PDF bytes."""
        response = self._http.get(f"/documents/{doc_id}/file", headers=self._headers)
        self._check(response)
        return response.content

    def highlights(self, doc_id: str, page: int, quote: str) -> dict[str, Any]:
        """GET /documents/{id}/highlights -> {page, page_width, page_height, rects}."""
        params = {"page": page, "quote": quote}
        return self._json(
            self._http.get(f"/documents/{doc_id}/highlights", params=params, headers=self._headers)
        )

    def query(self, question: str, doc_ids: list[str]) -> Iterator[tuple[str, dict[str, Any]]]:
        """POST /query and yield (event, data) pairs as they arrive."""
        body = {"session_id": self.session_id, "question": question, "doc_ids": doc_ids}
        with self._http.stream("POST", "/query", json=body) as response:
            if response.status_code != 200:
                response.read()
                self._check(response)
            yield from parse_sse_lines(response.iter_lines())

    def feedback(self, trace_id: str, rating: int, comment: str | None = None) -> None:
        """POST /feedback (rating -1 or 1)."""
        body = {
            "trace_id": trace_id,
            "session_id": self.session_id,
            "rating": rating,
            "comment": comment,
        }
        self._check(self._http.post("/feedback", json=body))

    def _json(self, response: httpx.Response) -> Any:
        self._check(response)
        return response.json()

    def _check(self, response: httpx.Response) -> None:
        if response.status_code < 400:
            return
        try:
            body = response.json()
        except ValueError:
            body = {}
        is_upload = response.request.method == "POST" and response.request.url.path == "/documents"
        message = UPLOAD_ERRORS.get(response.status_code) if is_upload else None
        message = message or body.get("message") or response.text
        raise ApiError(response.status_code, body.get("code", "error"), message)


def parse_sse_lines(lines: Iterator[str]) -> Iterator[tuple[str, dict[str, Any]]]:
    """Turn SSE lines into (event, data) pairs; ping comments are skipped."""
    event, data = None, []
    for line in lines:
        if not line:
            if event:
                yield event, json.loads("".join(data) or "{}")
            event, data = None, []
        elif line.startswith("event:"):
            event = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data.append(line[len("data:") :].strip())
    if event:
        yield event, json.loads("".join(data) or "{}")
