"""Find where a quote sits on a PDF page, so the UI can highlight it (FR-9)."""

from __future__ import annotations

from pathlib import Path

import pymupdf
from pydantic import BaseModel

from docqa.domain.errors import InvalidUploadError


class Rect(BaseModel):
    """A rectangle in PDF points, origin at the page's top-left corner."""

    x: float
    y: float
    width: float
    height: float


class Highlights(BaseModel):
    """Rectangles covering every occurrence of the quote on one page."""

    page: int
    page_width: float
    page_height: float
    rects: list[Rect]


def find_quote(path: Path, page: int, quote: str) -> Highlights:
    """Search ``quote`` on 1-based ``page`` with PyMuPDF's page.search_for.

    Raises:
        InvalidUploadError: The file cannot be opened or the page does not exist.
    """
    try:
        with pymupdf.open(path) as doc:
            if not 1 <= page <= doc.page_count:
                raise InvalidUploadError(f"Page {page} does not exist in this document.")
            pdf_page = doc[page - 1]
            found = pdf_page.search_for(" ".join(quote.split()))
            rects = [Rect(x=r.x0, y=r.y0, width=r.width, height=r.height) for r in found]
            return Highlights(
                page=page,
                page_width=pdf_page.rect.width,
                page_height=pdf_page.rect.height,
                rects=rects,
            )
    except (RuntimeError, ValueError) as err:  # PyMuPDF's errors on damaged files
        raise InvalidUploadError("The PDF could not be read.") from err
