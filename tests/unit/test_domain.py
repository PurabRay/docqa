import pytest
from pydantic import ValidationError

from docqa.domain import errors
from docqa.domain.models import (
    AccessFilter,
    DocumentRecord,
    Feedback,
    IngestionStatus,
    chunk_id,
)


def test_chunk_id_is_deterministic():
    assert chunk_id("doc-1", 0) == chunk_id("doc-1", 0)


def test_chunk_id_differs_by_doc_and_index():
    ids = {chunk_id("doc-1", 0), chunk_id("doc-1", 1), chunk_id("doc-2", 0)}
    assert len(ids) == 3


@pytest.mark.parametrize(
    "owner_id, doc_ids",
    [("", ["d1"]), ("   ", ["d1"]), ("u1", []), ("u1", [""]), ("u1", ["d1", " "])],
)
def test_access_filter_rejects_empty_values(owner_id, doc_ids):
    with pytest.raises(ValidationError):
        AccessFilter(owner_id=owner_id, doc_ids=doc_ids)


def test_access_filter_is_immutable():
    access = AccessFilter(owner_id="u1", doc_ids=["d1"])
    with pytest.raises(ValidationError):
        access.owner_id = "u2"


def test_document_record_defaults():
    record = DocumentRecord(
        owner_id="u1", filename="a.pdf", sha256="0" * 64, page_count=3, embed_model="m"
    )
    assert record.status is IngestionStatus.QUEUED
    assert record.created_at.tzinfo is not None
    assert record.id


def test_status_enum_matches_design():
    assert [s.value for s in IngestionStatus] == [
        "queued",
        "extracting",
        "indexing",
        "syncing",
        "ready",
        "failed",
        "deleted",
    ]


def test_feedback_rating_must_be_plus_or_minus_one():
    with pytest.raises(ValidationError):
        Feedback(trace_id="t", session_id="s", rating=0)


@pytest.mark.parametrize(
    "error, status",
    [
        (errors.DocQAError, 500),
        (errors.ConfigurationError, 500),
        (errors.InvalidUploadError, 422),
        (errors.FileTooLargeError, 413),
        (errors.NotPdfError, 415),
        (errors.EncryptedPdfError, 422),
        (errors.NoTextLayerError, 422),
        (errors.DuplicateDocumentError, 409),
        (errors.StorageCapReachedError, 507),
        (errors.DocumentNotFoundError, 404),
        (errors.DocumentNotReadyError, 400),
        (errors.InputRejectedError, 400),
        (errors.RateLimitedError, 429),
        (errors.ProviderUnavailableError, 503),
        (errors.AllProvidersUnavailableError, 503),
        (errors.DatabaseUnavailableError, 503),
        (errors.SearchIndexNotReadyError, 503),
        (errors.IndexSyncTimeoutError, 503),
        (errors.EmbeddingVersionMismatchError, 503),
        (errors.AccessFilterMissingError, 500),
    ],
)
def test_error_http_status(error, status):
    assert error.http_status == status
    assert error().message  # every error has a default message


def test_every_error_is_listed_and_has_a_unique_code():
    def all_subclasses(cls):
        return {cls} | {s for c in cls.__subclasses__() for s in all_subclasses(c)}

    classes = all_subclasses(errors.DocQAError)
    assert len(classes) == 20  # update the table above when adding an error
    assert len({cls.code for cls in classes}) == len(classes)


def test_error_message_can_be_overridden():
    assert errors.DocumentNotFoundError("no such doc").message == "no such doc"
