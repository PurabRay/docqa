"""/documents: upload, list, get and delete PDFs."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query, Response, UploadFile, status
from fastapi.responses import FileResponse

from docqa.api.deps import get_container, owner_id
from docqa.api.schemas import DocumentBody, UploadAccepted
from docqa.bootstrap import Container
from docqa.domain.models import DocumentRecord
from docqa.ingestion.highlights import Highlights
from docqa.ingestion.validator import BYTES_PER_MB

router = APIRouter(tags=["documents"])
ContainerDep = Annotated[Container, Depends(get_container)]
OwnerDep = Annotated[str, Depends(owner_id)]


def to_body(doc: DocumentRecord) -> DocumentBody:
    """Domain record -> API body."""
    return DocumentBody(
        doc_id=doc.id,
        filename=doc.filename,
        page_count=doc.page_count,
        chunk_count=doc.chunk_count,
        status=doc.status.value,
        error=doc.error,
        created_at=doc.created_at,
    )


@router.post("/documents", status_code=status.HTTP_202_ACCEPTED, response_model=UploadAccepted)
async def upload(file: UploadFile, owner: OwnerDep, container: ContainerDep) -> UploadAccepted:
    """Accept a PDF and queue it for ingestion."""
    # Read one byte past the limit so an oversized file is detected without reading it all.
    max_bytes = container.settings.limits.max_pdf_mb * BYTES_PER_MB
    data = await file.read(max_bytes + 1)
    doc = await container.documents.upload(owner, file.filename or "upload.pdf", data)
    return UploadAccepted(doc_id=doc.id, status=doc.status.value)


@router.get("/documents", response_model=list[DocumentBody])
async def list_documents(owner: OwnerDep, container: ContainerDep) -> list[DocumentBody]:
    """The caller's documents, newest first."""
    return [to_body(doc) for doc in await container.documents.list(owner)]


@router.get("/documents/{doc_id}", response_model=DocumentBody)
async def get_document(doc_id: str, owner: OwnerDep, container: ContainerDep) -> DocumentBody:
    """One document and its ingestion status."""
    return to_body(await container.documents.get(owner, doc_id))


@router.get("/documents/{doc_id}/file")
async def get_file(doc_id: str, owner: OwnerDep, container: ContainerDep) -> FileResponse:
    """The owner's PDF, for the UI's viewer."""
    path = await container.documents.file_path(owner, doc_id)
    return FileResponse(path, media_type="application/pdf")


@router.get("/documents/{doc_id}/highlights", response_model=Highlights)
async def get_highlights(
    doc_id: str,
    owner: OwnerDep,
    container: ContainerDep,
    page: Annotated[int, Query(ge=1)],
    quote: Annotated[str, Query(min_length=1, max_length=300)],
) -> Highlights:
    """Rectangles of the quote on that page, so the UI never parses PDFs."""
    return await container.documents.highlights(owner, doc_id, page, quote)


@router.delete("/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(doc_id: str, owner: OwnerDep, container: ContainerDep) -> Response:
    """Delete the document and all its chunks."""
    await container.documents.delete(owner, doc_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
