from pathlib import Path

import httpx
import pytest
from ui.api_client import ApiClient, ApiError, parse_sse_lines

from docqa.domain.errors import InvalidUploadError
from docqa.ingestion.highlights import find_quote
from tests.fixtures.make_fixtures import KNOWN_SENTENCES

TEXT_PDF = Path(__file__).parents[1] / "fixtures" / "text.pdf"


def test_known_quote_gets_a_rectangle_on_its_page():
    found = find_quote(TEXT_PDF, 3, KNOWN_SENTENCES[3])
    assert found.rects and found.page_width > 0
    assert all(0 <= r.y <= found.page_height for r in found.rects)


def test_quote_on_another_page_gets_no_rectangles():
    assert find_quote(TEXT_PDF, 4, KNOWN_SENTENCES[3]).rects == []


def test_missing_page_is_an_error():
    with pytest.raises(InvalidUploadError):
        find_quote(TEXT_PDF, 99, "anything")


def test_parse_sse_lines_skips_pings_and_reads_events():
    lines = [
        ": ping",
        "",
        "event: meta",
        'data: {"trace_id": "t"}',
        "",
        "event: token",
        'data: {"text": "Hi"}',
        "",
    ]
    assert list(parse_sse_lines(iter(lines))) == [
        ("meta", {"trace_id": "t"}),
        ("token", {"text": "Hi"}),
    ]


def client(status, body):
    transport = httpx.MockTransport(lambda request: httpx.Response(status, json=body))
    return ApiClient(httpx.Client(transport=transport, base_url="http://api"), session_id="s")


@pytest.mark.parametrize(
    "status, text",
    [
        (409, "already uploaded"),
        (413, "too large"),
        (415, "Only PDF"),
        (422, "scanned"),
        (507, "Storage is full"),
    ],
)
def test_upload_errors_are_readable(status, text):
    with pytest.raises(ApiError) as caught:
        client(status, {"code": "x", "message": "raw"}).upload("a.pdf", b"%PDF-")
    assert text in caught.value.message and caught.value.status == status


def test_other_errors_keep_the_api_message():
    with pytest.raises(ApiError, match="Document not found"):
        client(404, {"code": "document_not_found", "message": "Document not found."}).delete("d1")
