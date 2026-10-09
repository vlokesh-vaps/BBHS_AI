"""FastAPI routes for knowledge base management in the AI Control Center."""

import gc
import hashlib
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, Response, UploadFile, status

from src.admin.kb_service import (
    check_embedding_service_reachable,
    get_kb_status,
    is_rebuilding,
    start_rebuild_job,
    validate_pdf_content,
)
from src.admin.schemas import KBStatusResponse
from src.api.admin import require_admin_access
from src.config import KB_MAX_UPLOAD_MB, KB_STAGING_DIR
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
    Upload a new knowledge-base PDF, validate, stage, and begin asynchronous rebuild.
    Returns 202 Accepted on success, 409 Conflict if rebuild in progress.
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

    # Content validation
    try:
        page_count, _ = validate_pdf_content(temp_file_path)
    except ValueError as val_err:
        _safe_unlink(temp_file_path)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        ) from val_err

    # Embedding service reachability check
    ok, err_msg = check_embedding_service_reachable()
    if not ok:
        _safe_unlink(temp_file_path)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Embedding service is unavailable: {err_msg}",
        )

    # Launch background rebuild
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
