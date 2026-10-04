from pathlib import Path

import pytest

from docqa.domain.errors import (
    EncryptedPdfError,
    FileTooLargeError,
    NoTextLayerError,
    NotPdfError,
)
from docqa.ingestion.validator import validate_pdf
from docqa.settings import load_settings

FIXTURES = Path(__file__).parents[1] / "fixtures"
LIMITS = load_settings(env_file=None).limits


@pytest.mark.parametrize("name, pages", [("text.pdf", 12), ("tables.pdf", 2)])
def test_accepts_text_pdfs(name, pages):
    info = validate_pdf(FIXTURES / name, LIMITS)
    assert info.page_count == pages
    assert info.chars_per_page >= LIMITS.min_chars_per_page


@pytest.mark.parametrize(
    "name, error",
    [
        ("scanned.pdf", NoTextLayerError),
        ("encrypted.pdf", EncryptedPdfError),
        ("pages301.pdf", FileTooLargeError),
    ],
)
def test_rejects_bad_pdfs(name, error):
    with pytest.raises(error):
        validate_pdf(FIXTURES / name, LIMITS)


def test_rejects_a_file_over_the_size_limit(tmp_path):
    big = tmp_path / "big.pdf"
    big.write_bytes(b"%PDF-1.7\n" + b"0" * (2 * 1024 * 1024))
    with pytest.raises(FileTooLargeError, match="1 MB"):
        validate_pdf(big, LIMITS.model_copy(update={"max_pdf_mb": 1}))


def test_rejects_a_non_pdf(tmp_path):
    fake = tmp_path / "notes.pdf"
    fake.write_text("just text, not a PDF")
    with pytest.raises(NotPdfError):
        validate_pdf(fake, LIMITS)


def test_rejects_a_corrupt_pdf(tmp_path):
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"%PDF-1.7\nthis is not really a pdf")
    with pytest.raises(NotPdfError):
        validate_pdf(broken, LIMITS)


def test_upload_errors_map_to_the_right_http_status():
    assert FileTooLargeError.http_status == 413
    assert NotPdfError.http_status == 415
    assert EncryptedPdfError.http_status == NoTextLayerError.http_status == 422
