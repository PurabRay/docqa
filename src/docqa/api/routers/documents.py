"""/documents: upload, list, get and delete PDFs."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Response, UploadFile, status

from docqa.api.deps import get_container, owner_id
from docqa.api.schemas import DocumentBody, UploadAccepted
from docqa.bootstrap import Container
from docqa.domain.models import DocumentRecord
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


@router.delete("/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(doc_id: str, owner: OwnerDep, container: ContainerDep) -> Response:
    """Delete the document and all its chunks."""
    await container.documents.delete(owner, doc_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
