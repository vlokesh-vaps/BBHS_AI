"""Atomic JSON-backed Knowledge Base document repository.

Stores institutional document records in data/kb_catalog.json using
atomic write-and-replace for thread safety, zero database overhead,
and maximum portability.
"""

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import uuid

from src.admin.schemas import DocumentCreateRequest, DocumentRecord, DocumentUpdateRequest
from src.config import KB_CATALOG_FILE, KB_DIR, KB_MANIFEST_FILE
from src.utils.logging import log_error, log_step

_repo_lock = threading.Lock()


def _get_utc_now_iso() -> str:
    """Return current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _load_catalog_unlocked() -> list[dict[str, Any]]:
    """Read the catalog JSON file. Caller must hold _repo_lock."""
    if not KB_CATALOG_FILE.exists():
        return []
    try:
        content = KB_CATALOG_FILE.read_text(encoding="utf-8").strip()
        if not content:
            return []
        data = json.loads(content)
        return data if isinstance(data, list) else []
    except Exception as exc:
        log_error("KB_CATALOG_READ_ERROR", exc)
        return []


def _save_catalog_unlocked(records: list[dict[str, Any]]) -> None:
    """Atomically write the catalog to data/kb_catalog.json. Caller must hold _repo_lock."""
    KB_CATALOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = KB_CATALOG_FILE.with_suffix(".tmp")
    tmp_file.write_text(json.dumps(records, indent=2), encoding="utf-8")
    tmp_file.replace(KB_CATALOG_FILE)


def init_kb_repository() -> None:
    """Initialize catalog JSON and seed from legacy manifest if empty."""
    with _repo_lock:
        records = _load_catalog_unlocked()
        if not records and KB_MANIFEST_FILE.exists():
            try:
                manifest_data = json.loads(KB_MANIFEST_FILE.read_text(encoding="utf-8"))
                active = manifest_data.get("active", {})
                if active and active.get("filename"):
                    doc_id = f"doc_{active.get('sha256', uuid.uuid4().hex)[:12]}"
                    now_str = active.get("uploaded_at") or _get_utc_now_iso()
                    seed_record = {
                        "id": doc_id,
                        "title": active.get("display_name", "Legacy Active PDF"),
                        "content_type": "pdf",
                        "category": "General",
                        "school_campus": "All Campuses",
                        "academic_year": "2026-2027",
                        "language": "English",
                        "owner": "System Admin",
                        "version": "1.0",
                        "status": "Published",
                        "effective_date": now_str[:10],
                        "expiry_date": None,
                        "file_name": active.get("filename"),
                        "source_url": None,
                        "content_text": None,
                        "chunks_count": int(active.get("chunks", 0)),
                        "size_bytes": int(active.get("size_bytes", 0)),
                        "rejection_reason": None,
                        "created_at": now_str,
                        "updated_at": now_str,
                        "reviewed_by": "System Admin",
                        "reviewed_at": now_str,
                    }
                    records.append(seed_record)
                    _save_catalog_unlocked(records)
                    log_step("KB_REPO_SEEDED_LEGACY", doc_id=doc_id)
            except Exception as exc:
                log_error("KB_REPO_SEED_FAILED", exc)
        elif not KB_CATALOG_FILE.exists():
            _save_catalog_unlocked(records)


def list_documents(
    category: str | None = None,
    status: str | None = None,
    school_campus: str | None = None,
    search: str | None = None,
) -> list[DocumentRecord]:
    """Retrieve filtered list of documents from catalog."""
    init_kb_repository()
    with _repo_lock:
        raw_list = _load_catalog_unlocked()

    results: list[DocumentRecord] = []
    term = search.strip().lower() if search else None

    for item in raw_list:
        doc = DocumentRecord(**item)
        if category and doc.category.lower() != category.lower():
            continue
        if status and doc.status.lower() != status.lower():
            continue
        if school_campus and school_campus != "All Campuses":
            if doc.school_campus not in (school_campus, "All Campuses"):
                continue
        if term:
            searchable = f"{doc.title} {doc.owner} {doc.category}".lower()
            if term not in searchable:
                continue
        results.append(doc)

    # Sort descending by updated_at
    results.sort(key=lambda d: d.updated_at, reverse=True)
    return results


def get_document(doc_id: str) -> DocumentRecord | None:
    """Retrieve single document by ID."""
    init_kb_repository()
    with _repo_lock:
        raw_list = _load_catalog_unlocked()
        for item in raw_list:
            if item.get("id") == doc_id:
                return DocumentRecord(**item)
    return None


def create_document(
    doc_create: DocumentCreateRequest,
    file_name: str | None = None,
    size_bytes: int = 0,
) -> DocumentRecord:
    """Create a new document in Draft (or Review if requested) state."""
    init_kb_repository()
    doc_id = f"doc_{uuid.uuid4().hex[:12]}"
    now = _get_utc_now_iso()
    initial_status = "Review" if doc_create.submit_for_review else "Draft"

    new_dict: dict[str, Any] = {
        "id": doc_id,
        "title": doc_create.title.strip(),
        "content_type": doc_create.content_type,
        "category": doc_create.category.strip(),
        "school_campus": doc_create.school_campus.strip(),
        "academic_year": doc_create.academic_year.strip(),
        "language": doc_create.language.strip(),
        "owner": doc_create.owner.strip(),
        "version": doc_create.version.strip(),
        "status": initial_status,
        "effective_date": doc_create.effective_date,
        "expiry_date": doc_create.expiry_date,
        "file_name": file_name,
        "source_url": doc_create.source_url,
        "content_text": doc_create.content_text,
        "chunks_count": 0,
        "size_bytes": size_bytes,
        "rejection_reason": None,
        "created_at": now,
        "updated_at": now,
        "reviewed_by": None,
        "reviewed_at": None,
    }

    with _repo_lock:
        records = _load_catalog_unlocked()
        records.append(new_dict)
        _save_catalog_unlocked(records)

    return DocumentRecord(**new_dict)


def update_document(doc_id: str, doc_update: DocumentUpdateRequest) -> DocumentRecord | None:
    """Update metadata or content for an existing document."""
    init_kb_repository()
    now = _get_utc_now_iso()

    with _repo_lock:
        records = _load_catalog_unlocked()
        target = None
        for item in records:
            if item.get("id") == doc_id:
                target = item
                break

        if not target:
            return None

        if doc_update.title is not None:
            target["title"] = doc_update.title.strip()
        if doc_update.category is not None:
            target["category"] = doc_update.category.strip()
        if doc_update.school_campus is not None:
            target["school_campus"] = doc_update.school_campus.strip()
        if doc_update.academic_year is not None:
            target["academic_year"] = doc_update.academic_year.strip()
        if doc_update.language is not None:
            target["language"] = doc_update.language.strip()
        if doc_update.version is not None:
            target["version"] = doc_update.version.strip()
        if doc_update.effective_date is not None:
            target["effective_date"] = doc_update.effective_date
        if doc_update.expiry_date is not None:
            target["expiry_date"] = doc_update.expiry_date
        if doc_update.content_text is not None:
            target["content_text"] = doc_update.content_text
        if doc_update.source_url is not None:
            target["source_url"] = doc_update.source_url

        target["updated_at"] = now
        _save_catalog_unlocked(records)
        return DocumentRecord(**target)


def update_workflow_status(
    doc_id: str,
    action: str,
    comment: str | None = None,
    actor: str | None = None,
) -> DocumentRecord | None:
    """
    Execute state machine transition for document approval workflow.

    Actions:
    - submit_for_review: Draft -> Review
    - approve: Review -> Approved
    - reject: Review -> Draft (with rejection_reason)
    - publish: Approved/Review/Draft -> Published
    - disable: Published -> Disabled
    - archive: Any -> Archived
    """
    init_kb_repository()
    now = _get_utc_now_iso()

    with _repo_lock:
        records = _load_catalog_unlocked()
        target = None
        for item in records:
            if item.get("id") == doc_id:
                target = item
                break

        if not target:
            return None

        if action == "submit_for_review":
            target["status"] = "Review"
            target["rejection_reason"] = None
        elif action == "approve":
            target["status"] = "Approved"
            target["reviewed_by"] = actor or "Administrator"
            target["reviewed_at"] = now
            target["rejection_reason"] = None
        elif action == "reject":
            target["status"] = "Draft"
            target["rejection_reason"] = comment or "Changes requested by reviewer"
            target["reviewed_by"] = actor or "Administrator"
            target["reviewed_at"] = now
        elif action == "publish":
            target["status"] = "Published"
            target["reviewed_by"] = actor or target.get("reviewed_by") or "Administrator"
            target["reviewed_at"] = now
            target["rejection_reason"] = None
        elif action == "disable":
            target["status"] = "Disabled"
        elif action == "archive":
            target["status"] = "Archived"

        target["updated_at"] = now
        _save_catalog_unlocked(records)
        return DocumentRecord(**target)


def update_document_file_and_version(
    doc_id: str,
    file_name: str,
    size_bytes: int,
    version: str,
) -> DocumentRecord | None:
    """Update file reference and version when replacing a document."""
    init_kb_repository()
    now = _get_utc_now_iso()

    with _repo_lock:
        records = _load_catalog_unlocked()
        target = None
        for item in records:
            if item.get("id") == doc_id:
                target = item
                break

        if not target:
            return None

        target["file_name"] = file_name
        target["size_bytes"] = size_bytes
        target["version"] = version
        target["updated_at"] = now
        _save_catalog_unlocked(records)
        return DocumentRecord(**target)


def update_chunks_count(doc_id: str, count: int) -> None:
    """Update chunk count in document record."""
    init_kb_repository()
    now = _get_utc_now_iso()

    with _repo_lock:
        records = _load_catalog_unlocked()
        for item in records:
            if item.get("id") == doc_id:
                item["chunks_count"] = count
                item["updated_at"] = now
                break
        _save_catalog_unlocked(records)


def delete_document(doc_id: str) -> bool:
    """Delete document from catalog and its associated file if present."""
    init_kb_repository()
    file_to_delete = None

    with _repo_lock:
        records = _load_catalog_unlocked()
        remaining: list[dict[str, Any]] = []
        found = False

        for item in records:
            if item.get("id") == doc_id:
                found = True
                file_to_delete = item.get("file_name")
            else:
                remaining.append(item)

        if not found:
            return False

        _save_catalog_unlocked(remaining)

    if file_to_delete:
        file_path = KB_DIR / file_to_delete
        if file_path.is_file():
            try:
                file_path.unlink()
            except OSError:
                pass

    return True


def get_active_published_documents() -> list[DocumentRecord]:
    """Retrieve all documents that are currently Published and eligible for search."""
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    docs = list_documents(status="Published")
    eligible: list[DocumentRecord] = []
    for doc in docs:
        if doc.effective_date and doc.effective_date > today_str:
            continue
        if doc.expiry_date and doc.expiry_date < today_str:
            continue
        eligible.append(doc)
    return eligible
