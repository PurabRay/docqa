"""Reject bad uploads before any real work starts (FR-1, FR-2)."""

from __future__ import annotations

from pathlib import Path

import pymupdf
from pydantic import BaseModel

from docqa.config_schema import LimitsConfig
from docqa.domain.errors import (
    EncryptedPdfError,
    FileTooLargeError,
    NoTextLayerError,
    NotPdfError,
)

PDF_MAGIC = b"%PDF-"
BYTES_PER_MB = 1024 * 1024


class PdfInfo(BaseModel):
    """What the validator learned about an accepted PDF."""

    page_count: int
    chars_per_page: float


def validate_pdf(path: Path, limits: LimitsConfig) -> PdfInfo:
    """Check size, type, encryption, page count and text layer, cheapest check first.

    The text-layer check reads the PDF's own text with PyMuPDF (no OCR), so
    scanned PDFs are rejected before pymupdf4llm ever sees them.

    Raises:
        FileTooLargeError, NotPdfError, EncryptedPdfError, NoTextLayerError.
    """
    if path.stat().st_size > limits.max_pdf_mb * BYTES_PER_MB:
        raise FileTooLargeError(f"The file is larger than {limits.max_pdf_mb} MB.")
    with path.open("rb") as file:
        if file.read(len(PDF_MAGIC)) != PDF_MAGIC:
            raise NotPdfError()
    try:
        doc = pymupdf.open(path)
    except (pymupdf.FileDataError, RuntimeError) as err:
        raise NotPdfError("The file is not a readable PDF.") from err
    with doc:
        if doc.needs_pass:
            raise EncryptedPdfError()
        if doc.page_count > limits.max_pages:
            raise FileTooLargeError(f"The PDF has more than {limits.max_pages} pages.")
        total_chars = sum(len(page.get_text().strip()) for page in doc)
        chars_per_page = total_chars / max(doc.page_count, 1)
        if chars_per_page < limits.min_chars_per_page:
            raise NoTextLayerError()
        return PdfInfo(page_count=doc.page_count, chars_per_page=chars_per_page)
