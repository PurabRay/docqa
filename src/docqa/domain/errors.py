"""DocQAError hierarchy (docs/DESIGN.md, "Error hierarchy").

Each error has a stable ``code`` and the ``http_status`` the API returns for it.
The class docstring is the default user-facing message.
"""

from __future__ import annotations

from typing import ClassVar


class DocQAError(Exception):
    """Something went wrong."""

    code: ClassVar[str] = "internal_error"
    http_status: ClassVar[int] = 500

    def __init__(self, message: str | None = None) -> None:
        self.message = message or (type(self).__doc__ or "").strip()
        super().__init__(self.message)


class ConfigurationError(DocQAError):
    """DocQA is not configured correctly."""

    code = "configuration_error"


class InvalidUploadError(DocQAError):
    """This file cannot be processed."""

    code, http_status = "invalid_upload", 422


class FileTooLargeError(InvalidUploadError):
    """The file is too large."""

    code, http_status = "file_too_large", 413


class NotPdfError(InvalidUploadError):
    """Only PDF files are supported."""

    code, http_status = "not_pdf", 415


class EncryptedPdfError(InvalidUploadError):
    """Encrypted PDFs are not supported."""

    code = "encrypted_pdf"


class NoTextLayerError(InvalidUploadError):
    """This PDF looks scanned; scanned PDFs are not supported in v1."""

    code = "no_text_layer"


class DuplicateDocumentError(DocQAError):
    """You already uploaded this document."""

    code, http_status = "duplicate_document", 409


class StorageCapReachedError(DocQAError):
    """Storage is full; delete a document before uploading another."""

    code, http_status = "storage_cap_reached", 507


class DocumentNotFoundError(DocQAError):
    """Document not found."""

    code, http_status = "document_not_found", 404


class DocumentNotReadyError(DocQAError):
    """One of the selected documents is not ready yet."""

    code, http_status = "document_not_ready", 400


class InputRejectedError(DocQAError):
    """The question was rejected."""

    code, http_status = "input_rejected", 400


class RateLimitedError(DocQAError):
    """Too many requests; please wait a moment."""

    code, http_status = "rate_limited", 429


class ProviderUnavailableError(DocQAError):
    """The language model provider is unavailable."""

    code, http_status = "provider_unavailable", 503


class AllProvidersUnavailableError(DocQAError):
    """No language model provider is available."""

    code, http_status = "all_providers_unavailable", 503


class DatabaseUnavailableError(DocQAError):
    """The database is unavailable."""

    code, http_status = "database_unavailable", 503


class SearchIndexNotReadyError(DocQAError):
    """Search indexes are not ready."""

    code, http_status = "search_index_not_ready", 503


class IndexSyncTimeoutError(DocQAError):
    """Indexing took too long; please retry the upload."""

    code, http_status = "index_sync_timeout", 503


class EmbeddingVersionMismatchError(DocQAError):
    """The search index was built with a different embedding model."""

    code, http_status = "embedding_version_mismatch", 503


class AccessFilterMissingError(DocQAError):
    """A search was attempted without an access filter."""

    code = "access_filter_missing"
