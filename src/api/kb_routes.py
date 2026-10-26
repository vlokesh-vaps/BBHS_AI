"""FastAPI routes for knowledge base management in the AI Control Center."""

import gc
import hashlib
import time
import uuid
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, Response, UploadFile, status

from src.admin.kb_indexer import deindex_document_by_id, reindex_all_active_documents, sync_document_index
from src.admin.kb_repository import (
    create_document,
    delete_document,
    get_document,
    list_documents,
    update_document,
    update_document_file_and_version,
    update_workflow_status,
)
from src.admin.kb_service import (
    check_embedding_service_reachable,
    get_kb_status,
    is_rebuilding,
    sanitize_display_name,
    start_rebuild_job,
    validate_pdf_content,
)
from src.admin.schemas import (
    DocumentCreateRequest,
    DocumentRecord,
    DocumentUpdateRequest,
    KBStatusResponse,
    WorkflowActionRequest,
)
from src.api.admin import require_admin_access
from src.config import KB_DIR, KB_MAX_UPLOAD_MB, KB_STAGING_DIR
from src.utils.logging import log_error, log_step

router = APIRouter(
    prefix="/admin/kb",
    tags=["admin-kb"],
    dependencies=[Depends(require_admin_access)],
)


def _safe_unlink(path: Path) -> None:
    """Safely unlink a file with retries on Windows."""
    gc.collect()
    if not path.exists():
        return
    for _ in range(5):
        try:
            path.unlink()
            return
        except OSError:
            gc.collect()
            time.sleep(0.05)


# ============================================================================
# Legacy Single-PDF Compatibility Routes
# ============================================================================


@router.get("/status", response_model=KBStatusResponse)
def get_status() -> KBStatusResponse:
    """Return active knowledge base metadata and current rebuild state."""
    return get_kb_status()


@router.post("/upload", status_code=status.HTTP_202_ACCEPTED, response_model=KBStatusResponse)
async def upload_pdf(
    request: Request,
    file: UploadFile = File(...),
) -> KBStatusResponse:
    """
    Legacy endpoint: Upload a single knowledge-base PDF and begin asynchronous rebuild.
    """
    if is_rebuilding():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A knowledge-base rebuild is already in progress.",
        )

    filename = file.filename or "uploaded.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format: must be a .pdf file",
        )

    KB_STAGING_DIR.mkdir(parents=True, exist_ok=True)
    temp_job_id = uuid.uuid4().hex[:8]
    temp_file_path = KB_STAGING_DIR / f"upload_{temp_job_id}.pdf"

    max_bytes = KB_MAX_UPLOAD_MB * 1024 * 1024
    bytes_written = 0
    hasher = hashlib.sha256()

    try:
        with open(temp_file_path, "wb") as out_file:
            while chunk := await file.read(64 * 1024):
                bytes_written += len(chunk)
                if bytes_written > max_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"File exceeds maximum allowed size ({KB_MAX_UPLOAD_MB}MB)",
                    )
                hasher.update(chunk)
                out_file.write(chunk)
    except HTTPException:
        _safe_unlink(temp_file_path)
        raise
    except Exception as exc:
        _safe_unlink(temp_file_path)
        log_error("KB_UPLOAD_IO_ERROR", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to read uploaded file",
        ) from exc
    finally:
        await file.close()

    file_sha256 = hasher.hexdigest()

    try:
        page_count, _ = validate_pdf_content(temp_file_path)
    except ValueError as val_err:
        _safe_unlink(temp_file_path)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        ) from val_err

    ok, err_msg = check_embedding_service_reachable()
    if not ok:
        _safe_unlink(temp_file_path)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Embedding service is unavailable: {err_msg}",
        )

    started, msg = start_rebuild_job(
        temp_file_path=temp_file_path,
        original_filename=filename,
        file_sha256=file_sha256,
        page_count=page_count,
        app_instance=request.app,
    )

    if not started:
        _safe_unlink(temp_file_path)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=msg,
        )

    log_step("KB_UPLOAD_ACCEPTED", filename=filename, size_bytes=bytes_written, pages=page_count)
    return get_kb_status()


# ============================================================================
# Multi-Document Knowledge Base Repository Routes
# ============================================================================


@router.get("/documents", response_model=list[DocumentRecord])
def list_kb_documents(
    category: str | None = Query(default=None),
    status: str | None = Query(default=None),
    school_campus: str | None = Query(default=None),
    search: str | None = Query(default=None),
) -> list[DocumentRecord]:
    """List documents with optional filtering by category, status, campus, or keyword."""
    return list_documents(
        category=category,
        status=status,
        school_campus=school_campus,
        search=search,
    )


@router.post("/documents/text", response_model=DocumentRecord)
def create_text_document(
    payload: DocumentCreateRequest,
    request: Request,
) -> DocumentRecord:
    """Create a text-based document (Notice, FAQ, Web Content) with full metadata."""
    if payload.content_type in ("pdf", "docx"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File-based content must be uploaded via /admin/kb/documents/file",
        )
    if not (payload.content_text or payload.source_url):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either content_text or source_url is required for text-based documents",
        )

    record = create_document(payload, file_name=None, size_bytes=len(payload.content_text or ""))
    log_step("KB_DOC_CREATED", doc_id=record.id, type=record.content_type, status=record.status)
    return record


@router.post("/documents/file", response_model=DocumentRecord)
async def create_file_document(
    request: Request,
    file: UploadFile = File(...),
    title: str = Form(...),
    category: str = Form("General"),
    school_campus: str = Form("All Campuses"),
    academic_year: str = Form("2026-2027"),
    language: str = Form("English"),
    owner: str = Form("Staff Member"),
    version: str = Form("1.0"),
    effective_date: str | None = Form(None),
    expiry_date: str | None = Form(None),
    submit_for_review: bool = Form(False),
) -> DocumentRecord:
    """Upload a PDF or Word document (.docx) with attached institutional metadata."""
    filename = file.filename or "document.pdf"
    clean_name = sanitize_display_name(filename)
    suffix = Path(clean_name).suffix.lower()

    if suffix not in (".pdf", ".docx"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported format. Only .pdf and .docx files are accepted.",
        )

    content_type = "pdf" if suffix == ".pdf" else "docx"
    KB_DIR.mkdir(parents=True, exist_ok=True)

    hasher = hashlib.sha256()
    temp_bytes = bytearray()
    max_bytes = KB_MAX_UPLOAD_MB * 1024 * 1024

    while chunk := await file.read(64 * 1024):
        if len(temp_bytes) + len(chunk) > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File exceeds maximum allowed size ({KB_MAX_UPLOAD_MB}MB)",
            )
        temp_bytes.extend(chunk)
        hasher.update(chunk)

    file_hash = hasher.hexdigest()[:12]
    stored_filename = f"{file_hash}_{Path(clean_name).stem[:30]}{suffix}"
    target_path = KB_DIR / stored_filename
    target_path.write_bytes(temp_bytes)

    create_req = DocumentCreateRequest(
        title=title,
        content_type=content_type,
        category=category,
        school_campus=school_campus,
        academic_year=academic_year,
        language=language,
        owner=owner,
        version=version,
        effective_date=effective_date,
        expiry_date=expiry_date,
        submit_for_review=submit_for_review,
    )
    record = create_document(create_req, file_name=stored_filename, size_bytes=len(temp_bytes))
    log_step("KB_FILE_DOC_CREATED", doc_id=record.id, file=stored_filename, status=record.status)
    return record


@router.get("/documents/{doc_id}", response_model=DocumentRecord)
def get_kb_document(doc_id: str) -> DocumentRecord:
    """Retrieve details of a single document."""
    record = get_document(doc_id)
    if not record:
        raise HTTPException(status_code=404, detail="Document not found")
    return record


@router.put("/documents/{doc_id}", response_model=DocumentRecord)
def update_kb_document(
    doc_id: str,
    payload: DocumentUpdateRequest,
    request: Request,
) -> DocumentRecord:
    """Update metadata or content. Re-indexes automatically if document is Published."""
    record = update_document(doc_id, payload)
    if not record:
        raise HTTPException(status_code=404, detail="Document not found")

    # If the document is currently Published, automatically re-index updated content
    if record.status == "Published":
        sync_document_index(record, app_instance=request.app)

    log_step("KB_DOC_UPDATED", doc_id=doc_id, status=record.status)
    return record


@router.post("/documents/{doc_id}/action", response_model=DocumentRecord)
def workflow_action(
    doc_id: str,
    payload: WorkflowActionRequest,
    request: Request,
) -> DocumentRecord:
    """
    Execute approval workflow state transition.
    Actions: submit_for_review, approve, reject, publish, disable, archive.
    """
    record = update_workflow_status(
        doc_id=doc_id,
        action=payload.action,
        comment=payload.comment,
        actor=payload.actor,
    )
    if not record:
        raise HTTPException(status_code=404, detail="Document not found")

    # Automatically re-index or de-index when entering/exiting Published state
    sync_document_index(record, app_instance=request.app)

    log_step("KB_WORKFLOW_ACTION", doc_id=doc_id, action=payload.action, new_status=record.status)
    return record


@router.post("/documents/{doc_id}/replace", response_model=DocumentRecord)
async def replace_document_file(
    doc_id: str,
    request: Request,
    file: UploadFile = File(...),
    new_version: str = Form("2.0"),
) -> DocumentRecord:
    """Upload a replacement file to create an updated version of an existing document."""
    record = get_document(doc_id)
    if not record:
        raise HTTPException(status_code=404, detail="Document not found")

    filename = file.filename or "replacement.pdf"
    clean_name = sanitize_display_name(filename)
    suffix = Path(clean_name).suffix.lower()

    if suffix not in (".pdf", ".docx"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported format. Only .pdf and .docx files are accepted.",
        )

    KB_DIR.mkdir(parents=True, exist_ok=True)
    hasher = hashlib.sha256()
    temp_bytes = bytearray()
    max_bytes = KB_MAX_UPLOAD_MB * 1024 * 1024

    while chunk := await file.read(64 * 1024):
        if len(temp_bytes) + len(chunk) > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File exceeds maximum allowed size ({KB_MAX_UPLOAD_MB}MB)",
            )
        temp_bytes.extend(chunk)
        hasher.update(chunk)

    file_hash = hasher.hexdigest()[:12]
    stored_filename = f"{file_hash}_{Path(clean_name).stem[:30]}{suffix}"
    target_path = KB_DIR / stored_filename
    target_path.write_bytes(temp_bytes)

    updated_record = update_document_file_and_version(
        doc_id=doc_id,
        file_name=stored_filename,
        size_bytes=len(temp_bytes),
        version=new_version,
    )
    if not updated_record:
        raise HTTPException(status_code=500, detail="Failed to update document version")

    if updated_record.status == "Published":
        sync_document_index(updated_record, app_instance=request.app)

    log_step("KB_DOC_REPLACED", doc_id=doc_id, version=new_version)
    return updated_record


@router.delete("/documents/{doc_id}")
def delete_kb_document(doc_id: str, request: Request) -> dict[str, str]:
    """Delete a document and purge all its vectors from the index."""
    deindex_document_by_id(doc_id, app_instance=request.app)
    deleted = delete_document(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    log_step("KB_DOC_DELETED", doc_id=doc_id)
    return {"status": "deleted", "doc_id": doc_id}


@router.post("/sync-all")
def sync_all_documents(request: Request) -> dict[str, Any]:
    """Force re-index all active published documents."""
    total_chunks = reindex_all_active_documents(app_instance=request.app)
    log_step("KB_SYNC_ALL_COMPLETED", chunks=total_chunks)
    return {"status": "ok", "total_indexed_chunks": total_chunks}
