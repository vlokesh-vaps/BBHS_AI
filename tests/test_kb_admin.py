"""Unit and integration tests for the Knowledge Base Admin Panel, Approval Workflow, and QA Telemetry."""

import io
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from docx import Document as DocxDocument
from fastapi.testclient import TestClient

from src.admin.kb_indexer import deindex_document_by_id, sync_document_index
from src.admin.kb_repository import (
    create_document,
    delete_document,
    get_active_published_documents,
    get_document,
    init_kb_repository,
    list_documents,
    update_document,
    update_workflow_status,
)
from src.admin.parsers import chunk_and_enrich_document, parse_docx, parse_faq, parse_notice
from src.admin.schemas import DocumentCreateRequest, DocumentUpdateRequest
from src.api.app import app
from src.config import ADMIN_PASSWORD, ADMIN_USERNAME


@pytest.fixture(autouse=True)
def setup_test_dbs():
    """Ensure repositories are initialized."""
    init_kb_repository()


def test_document_metadata_crud():
    """Test full metadata persistence and update for a document."""
    req = DocumentCreateRequest(
        title="Admission Policy 2027",
        content_type="notice",
        category="Admissions",
        school_campus="Main Campus",
        academic_year="2027-2028",
        language="English",
        owner="Admissions Office",
        version="1.0",
        effective_date="2027-01-01",
        expiry_date="2027-12-31",
        content_text="# Admission Guidelines\nAll applicants must submit their forms by March 31.",
        submit_for_review=False,
    )

    doc = create_document(req)
    assert doc.id.startswith("doc_")
    assert doc.title == "Admission Policy 2027"
    assert doc.category == "Admissions"
    assert doc.school_campus == "Main Campus"
    assert doc.status == "Draft"

    # Update metadata
    update_req = DocumentUpdateRequest(
        title="Admission Policy 2027 - Revised",
        version="1.1",
        school_campus="All Campuses",
    )
    updated = update_document(doc.id, update_req)
    assert updated is not None
    assert updated.title == "Admission Policy 2027 - Revised"
    assert updated.version == "1.1"
    assert updated.school_campus == "All Campuses"

    # Verify retrieval by filter
    filtered = list_documents(category="Admissions", school_campus="Main Campus")
    assert any(d.id == doc.id for d in filtered)

    # Clean up
    delete_document(doc.id)
    assert get_document(doc.id) is None


def test_approval_workflow_state_machine():
    """Test Draft -> Review -> Approved -> Published and rejection cycles."""
    req = DocumentCreateRequest(
        title="Fee Structure 2027",
        content_type="notice",
        category="Fees & Payments",
        school_campus="High School",
        academic_year="2027-2028",
        owner="Accountant",
        content_text="Annual tuition fee is due on the 5th of each quarter.",
        submit_for_review=False,
    )
    doc = create_document(req)
    assert doc.status == "Draft"

    # 1. Staff submits for review
    doc = update_workflow_status(doc.id, "submit_for_review")
    assert doc.status == "Review"

    # 2. Administrator requests changes (rejection to Draft)
    doc = update_workflow_status(
        doc.id, "reject", comment="Please clarify late payment penalties", actor="Principal"
    )
    assert doc.status == "Draft"
    assert doc.rejection_reason == "Please clarify late payment penalties"
    assert doc.reviewed_by == "Principal"

    # 3. Staff resubmits
    doc = update_workflow_status(doc.id, "submit_for_review")
    assert doc.status == "Review"
    assert doc.rejection_reason is None

    # 4. Administrator approves
    doc = update_workflow_status(doc.id, "approve", actor="Principal")
    assert doc.status == "Approved"

    # 5. Published
    doc = update_workflow_status(doc.id, "publish", actor="Admin")
    assert doc.status == "Published"

    # 6. Deactivate / Disable
    doc = update_workflow_status(doc.id, "disable")
    assert doc.status == "Disabled"

    delete_document(doc.id)


def test_word_document_parser():
    """Test parsing .docx Word files preserving paragraphs and tables."""
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        doc = DocxDocument()
        doc.add_heading("School Transport Policy", level=1)
        doc.add_paragraph("Buses will depart punctually at 3:30 PM.")
        table = doc.add_table(rows=2, cols=2)
        table.rows[0].cells[0].text = "Route"
        table.rows[0].cells[1].text = "Timing"
        table.rows[1].cells[0].text = "Route 4 (Indiranagar)"
        table.rows[1].cells[1].text = "7:15 AM"
        doc.save(str(tmp_path))

        docs = parse_docx(tmp_path)
        assert len(docs) == 1
        content = docs[0].page_content
        assert "# School Transport Policy" in content
        assert "Buses will depart punctually at 3:30 PM." in content
        assert "Route 4 (Indiranagar)" in content
        assert "7:15 AM" in content
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_faq_and_notice_parsers():
    """Test parsing Notice and FAQ records into structured chunk documents."""
    faq_req = DocumentCreateRequest(
        title="Admission Registration Deadlines",
        content_type="faq",
        category="Admissions",
        school_campus="All Campuses",
        academic_year="2026-2027",
        owner="Staff",
        content_text="Q: When does admission registration close?\nA: Registration closes on November 30.",
    )
    faq_doc = create_document(faq_req)
    chunks = parse_faq(faq_doc)
    assert len(chunks) >= 1
    assert "Registration closes on November 30" in chunks[0].page_content
    delete_document(faq_doc.id)

    notice_req = DocumentCreateRequest(
        title="Annual Sports Day Circular",
        content_type="notice",
        category="General",
        school_campus="Primary Wing",
        academic_year="2026-2027",
        owner="Sports Dept",
        content_text="Sports Day will be held on December 15 at the school grounds.",
    )
    notice_doc = create_document(notice_req)
    notice_chunks = parse_notice(notice_doc)
    assert len(notice_chunks) == 1
    assert "Annual Sports Day Circular" in notice_chunks[0].page_content
    delete_document(notice_doc.id)


def test_json_catalog_persistence():
    """Test that KB catalog persists correctly to data/kb_catalog.json without SQLite."""
    from src.config import KB_CATALOG_FILE

    req = DocumentCreateRequest(
        title="Sports Calendar 2027",
        content_type="notice",
        category="Sports",
        school_campus="Main Campus",
        academic_year="2026-2027",
        owner="Coach",
        content_text="Inter-school athletic meet starts next month.",
    )
    doc = create_document(req)
    assert KB_CATALOG_FILE.exists()

    # Re-read through get_document
    fetched = get_document(doc.id)
    assert fetched is not None
    assert fetched.title == "Sports Calendar 2027"

    # Clean up
    delete_document(doc.id)
    assert get_document(doc.id) is None


def test_fastapi_admin_endpoints():
    """Integration test for admin API endpoints via TestClient."""
    client = TestClient(app)
    auth_headers = {"Authorization": "Basic dm1zY2hhdDp2bXMxMjM="}

    # Mock sync_document_index to avoid requiring a running Ollama instance during unit test
    with patch("src.api.kb_routes.sync_document_index") as mock_sync:
        mock_sync.return_value = 5

        # 1. Create text document
        resp = client.post(
            "/admin/kb/documents/text",
            headers=auth_headers,
            json={
                "title": "API Test Examination Schedule",
                "content_type": "notice",
                "category": "Examinations",
                "school_campus": "High School",
                "academic_year": "2026-2027",
                "language": "English",
                "owner": "Exam Cell",
                "version": "1.0",
                "content_text": "Mid-term exams commence on October 10.",
                "submit_for_review": True,
            },
        )
        assert resp.status_code == 200, resp.text
        doc_data = resp.json()
        doc_id = doc_data["id"]
        assert doc_data["status"] == "Review"

        preview_resp = client.get(f"/admin/kb/documents/{doc_id}/preview", headers=auth_headers)
        assert preview_resp.status_code == 200
        assert preview_resp.json()["content"] == "Mid-term exams commence on October 10."
        assert "data/" not in preview_resp.text

        # 2. Approve and publish document
        action_resp = client.post(
            f"/admin/kb/documents/{doc_id}/action",
            headers=auth_headers,
            json={"action": "publish", "actor": "Administrator"},
        )
        assert action_resp.status_code == 200
        assert action_resp.json()["status"] == "Published"

        # Clean up test document
        client.delete(f"/admin/kb/documents/{doc_id}", headers=auth_headers)
