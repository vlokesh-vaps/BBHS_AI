"""Knowledge Base (KB) upload, validation, staging, atomic swap, and recovery service."""

import gc
import hashlib
import os
import re
import shutil
import threading
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pymupdf

from src.admin.config_service import load_config, write_json_atomic
from src.admin.schemas import KBActiveInfo, KBRebuildStatus, KBStatusResponse
from src.config import (
    BM25_CACHE_FILE,
    KB_DIR,
    KB_MAX_PAGES,
    KB_MAX_UPLOAD_MB,
    KB_MANIFEST_FILE,
    KB_STAGING_DIR,
    OLLAMA_BASE_URL,
    PDF_PATH,
    PERSIST_DIRECTORY,
)
from src.core.retrievers import create_ensemble_retriever, load_bm25_index
from src.storage.vector_db import load_vector_store
from src.utils.logging import log_error, log_step

BACKUP_CHROMA_DIR = PERSIST_DIRECTORY.parent / "chroma_db_backup"

_rebuild_lock = threading.Lock()
_current_rebuild = KBRebuildStatus(state="idle", message="No rebuild running")


def is_rebuilding() -> bool:
    """Return True if a rebuild job is currently executing."""
    return _current_rebuild.state in ("validating", "building", "swapping")


def get_current_rebuild_status() -> KBRebuildStatus:
    """Return the current rebuild status model."""
    return _current_rebuild.model_copy()


def sanitize_display_name(filename: str) -> str:
    """Sanitize original client filename to prevent path traversal and control characters."""
    cleaned = Path(filename).name
    cleaned = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", cleaned).strip()
    if not cleaned:
        cleaned = "uploaded_document.pdf"
    if len(cleaned) > 120:
        stem = Path(cleaned).stem[:110]
        cleaned = f"{stem}.pdf"
    return cleaned


def load_active_kb_info() -> KBActiveInfo:
    """
    Load active KB information from manifest file.
    If no manifest exists, falls back to the default PDF_PATH if present.
    Never exposes internal server filesystem paths.
    """
    if KB_MANIFEST_FILE.exists():
        try:
            import json
            data = json.loads(KB_MANIFEST_FILE.read_text(encoding="utf-8"))
            active_data = data.get("active", {})
            return KBActiveInfo(
                display_name=str(active_data.get("display_name", "")),
                size_bytes=int(active_data.get("size_bytes", 0)),
                pages=int(active_data.get("pages", 0)),
                chunks=int(active_data.get("chunks", 0)),
                uploaded_at=active_data.get("uploaded_at"),
                sha256=str(active_data.get("sha256", "")),
            )
        except Exception as exc:
            log_error("MANIFEST_READ_FAILED", exc)

    fallback_pdf = None
    if PDF_PATH.is_dir():
        manifest_filename = ""
        if KB_MANIFEST_FILE.exists():
            try:
                import json
                manifest_filename = str(
                    json.loads(KB_MANIFEST_FILE.read_text(encoding="utf-8"))
                    .get("active", {}).get("filename", "")
                )
            except Exception:
                manifest_filename = ""
        candidate = PDF_PATH / Path(manifest_filename).name if manifest_filename else None
        if candidate and candidate.is_file():
            fallback_pdf = candidate
        else:
            pdfs = sorted(PDF_PATH.glob("*.pdf"))
            fallback_pdf = pdfs[0] if pdfs else None
    elif PDF_PATH.is_file():
        fallback_pdf = PDF_PATH

    if fallback_pdf is not None:
        doc = None
        try:
            size_bytes = fallback_pdf.stat().st_size
            doc = pymupdf.open(str(fallback_pdf))
            pages = doc.page_count
            doc.close()
            doc = None
            with open(fallback_pdf, "rb") as f:
                sha256_hash = hashlib.sha256(f.read()).hexdigest()
            return KBActiveInfo(
                display_name=fallback_pdf.name,
                size_bytes=size_bytes,
                pages=pages,
                chunks=0,
                uploaded_at=None,
                sha256=sha256_hash,
            )
        except Exception as exc:
            log_error("FALLBACK_PDF_INSPECT_FAILED", exc)
        finally:
            if doc is not None:
                try:
                    doc.close()
                except Exception:
                    pass
            gc.collect()

    return KBActiveInfo(
        display_name="None",
        size_bytes=0,
        pages=0,
        chunks=0,
        uploaded_at=None,
        sha256="",
    )


def get_kb_status() -> KBStatusResponse:
    """Return full KB status including active PDF info and rebuild status."""
    return KBStatusResponse(
        active=load_active_kb_info(),
        rebuild=get_current_rebuild_status(),
    )


def check_embedding_service_reachable() -> tuple[bool, str]:
    """Check if the embedding service (Ollama) is reachable before rebuilding."""
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            if resp.status == 200:
                return True, "Embedding service is reachable"
            return False, f"Embedding service returned HTTP {resp.status}"
    except (OSError, urllib.error.URLError) as exc:
        return False, f"Embedding service unreachable: {exc}"


def validate_pdf_content(temp_path: Path) -> tuple[int, int]:
    """
    Validate that the uploaded file is a valid, readable PDF.

    Returns:
        (page_count, total_text_length)

    Raises:
        ValueError with human-readable error on validation failure.
    """
    # 1. Type check: first bytes must be %PDF-
    try:
        with open(temp_path, "rb") as f:
            header = f.read(5)
            if header != b"%PDF-":
                raise ValueError("Invalid file format: must be a valid PDF starting with %PDF-")
    except OSError as exc:
        raise ValueError(f"Could not read uploaded file: {exc}") from exc

    # 2. Parse check using PyMuPDF
    doc = None
    try:
        try:
            doc = pymupdf.open(str(temp_path))
        except Exception as exc:
            raise ValueError("Corrupt or unreadable PDF file") from exc

        if doc.is_encrypted:
            raise ValueError("Encrypted or password-protected PDFs are not supported")

        page_count = doc.page_count
        if page_count < 1:
            raise ValueError("PDF contains no pages")
        if page_count > KB_MAX_PAGES:
            raise ValueError(f"PDF exceeds maximum page limit ({KB_MAX_PAGES} pages)")

        total_text_length = 0
        for page in doc:
            text = page.get_text()
            total_text_length += len(text.strip())

        # 3. Text layer check: reject scanned / OCR-less PDFs
        if total_text_length < 200:
            raise ValueError("No readable text found. Scanned PDFs (OCR) are not supported.")

        return page_count, total_text_length
    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass
        gc.collect()


def release_vector_store(vector_db: Any) -> None:
    """Safely release Chroma SQLite file handles on Windows before renaming."""
    if vector_db is not None:
        try:
            if hasattr(vector_db, "_client") and vector_db._client is not None:
                client = vector_db._client
                if hasattr(client, "close"):
                    client.close()
                if hasattr(client, "clear_system_cache"):
                    client.clear_system_cache()
        except Exception as exc:
            log_error("VECTOR_STORE_RELEASE_WARN", exc)
    gc.collect()


def _safe_unlink(path: Path) -> None:
    """Remove a file safely with retries for Windows file locking."""
    gc.collect()
    if not path.exists():
        return
    for attempt in range(5):
        try:
            path.unlink()
            return
        except OSError:
            gc.collect()
            time.sleep(0.05)


def _safe_rename(src: Path, dst: Path, retries: int = 5, delay: float = 0.2) -> None:
    """Rename path with retry loop for Windows file system safety."""
    for attempt in range(retries):
        try:
            if dst.exists():
                if dst.is_dir():
                    shutil.rmtree(dst)
                else:
                    dst.unlink()
            os.replace(src, dst)
            return
        except OSError:
            gc.collect()
            if attempt == retries - 1:
                raise
            time.sleep(delay)


def startup_recovery(app_instance: Any = None) -> None:
    """
    Perform cleanup on startup:
    1. Delete leftover staging directories.
    2. If live Chroma directory is missing but backup exists, restore it.
    3. Delete leftover backup directory if live exists.
    """
    KB_STAGING_DIR.mkdir(parents=True, exist_ok=True)
    KB_DIR.mkdir(parents=True, exist_ok=True)
    PDF_PATH.mkdir(parents=True, exist_ok=True)

    # Clean leftover staging
    for item in KB_STAGING_DIR.iterdir():
        try:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                _safe_unlink(item)
        except OSError as exc:
            log_error("STARTUP_STAGING_CLEAN_FAILED", exc)

    # Restore backup if live Chroma DB is missing
    if not PERSIST_DIRECTORY.exists() and BACKUP_CHROMA_DIR.exists():
        try:
            _safe_rename(BACKUP_CHROMA_DIR, PERSIST_DIRECTORY)
            log_step("STARTUP_RECOVERED_BACKUP_CHROMA")
        except OSError as exc:
            log_error("STARTUP_BACKUP_RESTORE_FAILED", exc)
    elif BACKUP_CHROMA_DIR.exists():
        try:
            shutil.rmtree(BACKUP_CHROMA_DIR)
        except OSError:
            pass


def execute_build_and_swap(
    job_id: str,
    temp_file_path: Path,
    original_filename: str,
    file_sha256: str,
    page_count: int,
    app_instance: Any,
    embedding_model: Any = None,
) -> None:
    """
    Background worker that runs ingestion in staging, then atomically swaps into production.
    """
    global _current_rebuild

    staging_job_dir = KB_STAGING_DIR / job_id
    staging_chroma_dir = staging_job_dir / "chroma_db"
    staging_bm25_file = staging_job_dir / "bm25_index.pkl"

    display_name = sanitize_display_name(original_filename)
    file_size = temp_file_path.stat().st_size
    # KB_PDF is the canonical persisted source directory for rebuilds.
    PDF_PATH.mkdir(parents=True, exist_ok=True)
    permanent_pdf_path = PDF_PATH / f"{file_sha256[:12]}.pdf"
    persisted_for_job = False

    try:
        KB_DIR.mkdir(parents=True, exist_ok=True)
        _current_rebuild.state = "building"
        _current_rebuild.message = "Extracting text and building vector embeddings in staging..."
        log_step("KB_REBUILD_BUILDING", job_id=job_id, pages=page_count)

        # Persist first, then build from the exact file that will represent
        # the active knowledge base.
        if not permanent_pdf_path.exists():
            shutil.copy2(temp_file_path, permanent_pdf_path)
            persisted_for_job = True

        from scripts.build_kb import build_vector_store
        from src.core.retrievers import BM25Index
        import pickle

        rag_config = load_config().rag
        chunks = build_vector_store(
            pdf_path=permanent_pdf_path,
            persist_dir=staging_chroma_dir,
            chunk_size=rag_config.chunk_size,
            chunk_overlap=rag_config.chunk_overlap,
            embedding_model=embedding_model,
        )

        bm25_index = BM25Index(chunks)
        staging_job_dir.mkdir(parents=True, exist_ok=True)
        with open(staging_bm25_file, "wb") as f:
            pickle.dump(bm25_index, f)

        _current_rebuild.state = "swapping"
        _current_rebuild.message = "Swapping new knowledge base into production..."
        log_step("KB_REBUILD_SWAPPING", job_id=job_id, chunks=len(chunks))

        if hasattr(app_instance.state, "vector_db"):
            release_vector_store(app_instance.state.vector_db)
            app_instance.state.vector_db = None
            app_instance.state.ensemble_retriever = None

        if PERSIST_DIRECTORY.exists():
            _safe_rename(PERSIST_DIRECTORY, BACKUP_CHROMA_DIR)

        try:
            _safe_rename(staging_chroma_dir, PERSIST_DIRECTORY)

            BM25_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(staging_bm25_file, BM25_CACHE_FILE)

            app_instance.state.vector_db = load_vector_store()
            app_instance.state.bm25_index = load_bm25_index(app_instance.state.vector_db)
            runtime_cfg = load_config()
            app_instance.state.ensemble_retriever = create_ensemble_retriever(
                app_instance.state.vector_db,
                app_instance.state.bm25_index,
                vector_top_k=runtime_cfg.rag.vector_top_k,
                bm25_top_k=runtime_cfg.rag.bm25_top_k,
                final_top_k=runtime_cfg.rag.final_top_k,
                similarity_threshold=max(
                    runtime_cfg.rag.similarity_threshold,
                    runtime_cfg.guardrails.minimum_retrieval_score,
                ),
            )

            if hasattr(app_instance.state, "cache") and app_instance.state.cache is not None:
                app_instance.state.cache.clear()

            manifest_payload = {
                "active": {
                    "display_name": display_name,
                    "size_bytes": file_size,
                    "pages": page_count,
                    "chunks": len(chunks),
                    "uploaded_at": datetime.now(timezone.utc).isoformat(),
                    "sha256": file_sha256,
                    "filename": f"{file_sha256[:12]}.pdf",
                }
            }
            write_json_atomic(manifest_payload, KB_MANIFEST_FILE)

            # Keep KB_PDF a one-source directory. Only remove old hash-named
            # uploads; never remove an administrator's unrelated file.
            for old_pdf in PDF_PATH.glob("*.pdf"):
                if old_pdf != permanent_pdf_path and re.fullmatch(r"[0-9a-f]{12}\.pdf", old_pdf.name):
                    _safe_unlink(old_pdf)

            if BACKUP_CHROMA_DIR.exists():
                shutil.rmtree(BACKUP_CHROMA_DIR, ignore_errors=True)

        except Exception as swap_exc:
            log_error("KB_SWAP_FAILED", swap_exc)
            if BACKUP_CHROMA_DIR.exists():
                try:
                    if PERSIST_DIRECTORY.exists():
                        shutil.rmtree(PERSIST_DIRECTORY, ignore_errors=True)
                    _safe_rename(BACKUP_CHROMA_DIR, PERSIST_DIRECTORY)
                    app_instance.state.vector_db = load_vector_store()
                    app_instance.state.bm25_index = load_bm25_index(app_instance.state.vector_db)
                    cfg = load_config()
                    app_instance.state.ensemble_retriever = create_ensemble_retriever(
                        app_instance.state.vector_db,
                        app_instance.state.bm25_index,
                        vector_top_k=cfg.rag.vector_top_k,
                        bm25_top_k=cfg.rag.bm25_top_k,
                        final_top_k=cfg.rag.final_top_k,
                        similarity_threshold=max(
                            cfg.rag.similarity_threshold,
                            cfg.guardrails.minimum_retrieval_score,
                        ),
                    )
                except Exception as rollback_exc:
                    log_error("KB_ROLLBACK_FAILED", rollback_exc)
            raise swap_exc

        _current_rebuild.state = "done"
        _current_rebuild.message = f"Successfully rebuilt and activated knowledge base ({len(chunks)} chunks)."
        _current_rebuild.finished_at = datetime.now(timezone.utc).isoformat()
        log_step("KB_REBUILD_SUCCESS", job_id=job_id, chunks=len(chunks))

    except Exception as exc:
        _current_rebuild.state = "failed"
        _current_rebuild.message = (
            "Build failed while indexing the PDF. Check the embedding service "
            "and server logs, then try again."
        )
        _current_rebuild.finished_at = datetime.now(timezone.utc).isoformat()
        log_error("KB_BUILD_ERROR", exc, job_id=job_id)
        if persisted_for_job:
            try:
                import json
                active_sha = json.loads(
                    KB_MANIFEST_FILE.read_text(encoding="utf-8")
                ).get("active", {}).get("sha256", "") if KB_MANIFEST_FILE.exists() else ""
                if active_sha != file_sha256:
                    _safe_unlink(permanent_pdf_path)
            except (OSError, ValueError, TypeError):
                _safe_unlink(permanent_pdf_path)
    finally:
        _safe_unlink(temp_file_path)
        if staging_job_dir.exists():
            shutil.rmtree(staging_job_dir, ignore_errors=True)

        if _rebuild_lock.locked():
            _rebuild_lock.release()


def start_rebuild_job(
    temp_file_path: Path,
    original_filename: str,
    file_sha256: str,
    page_count: int,
    app_instance: Any,
    embedding_model: Any = None,
) -> tuple[bool, str]:
    """
    Acquire lock and launch rebuild worker in a background thread.
    Returns (True, job_id) if started, (False, error_message) if locked.
    """
    global _current_rebuild

    if not _rebuild_lock.acquire(blocking=False):
        return False, "A knowledge-base rebuild is already running."

    job_id = uuid.uuid4().hex[:8]
    _current_rebuild = KBRebuildStatus(
        state="validating",
        message="Upload validated. Queued for background build.",
        started_at=datetime.now(timezone.utc).isoformat(),
        finished_at=None,
    )

    thread = threading.Thread(
        target=execute_build_and_swap,
        args=(
            job_id,
            temp_file_path,
            original_filename,
            file_sha256,
            page_count,
            app_instance,
            embedding_model,
        ),
        daemon=True,
    )
    thread.start()
    return True, job_id
