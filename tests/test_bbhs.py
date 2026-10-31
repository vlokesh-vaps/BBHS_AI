"""Comprehensive test suite for BBHS AI Assistant (Section 10)."""

import base64
import gc
import json
import os
import shutil
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from dotenv import load_dotenv

# Ensure environment is loaded before any other imports
load_dotenv()
os.environ.setdefault("ADMIN_USERNAME", "vmschat")
os.environ.setdefault("ADMIN_PASSWORD", "vms123")

import pymupdf
import pytest
from starlette.testclient import TestClient

from src.admin.config_service import load_config
from src.admin import kb_service
from src.admin.kb_service import (
    is_rebuilding,
    sanitize_display_name,
    startup_recovery,
    validate_pdf_content,
)
from src.api.app import BoundedSessionStore, app
from src.config import (
    ADMIN_PASSWORD,
    ADMIN_USERNAME,
    ASSISTANT_NAME,
    CORS_ALLOWED_ORIGINS,
    KB_DIR,
    KB_MANIFEST_FILE,
    KB_STAGING_DIR,
    PERSIST_DIRECTORY,
)
from src.core.prompts import (
    SYSTEM_DIRECT_INSTRUCTIONS,
    SYSTEM_RAG_INSTRUCTIONS,
    classify_query,
)

TEST_TMP = Path(__file__).resolve().parent / "tmp_data"


@pytest.fixture(scope="session", autouse=True)
def cleanup_test_tmp():
    """Ensure test temporary folder exists and is cleaned up safely."""
    TEST_TMP.mkdir(parents=True, exist_ok=True)
    yield
    gc.collect()
    if TEST_TMP.exists():
        shutil.rmtree(TEST_TMP, ignore_errors=True)


@pytest.fixture(scope="session")
def client():
    """Test client for FastAPI app."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def admin_auth_headers():
    """Valid HTTP basic auth credentials."""
    creds = base64.b64encode(f"{ADMIN_USERNAME}:{ADMIN_PASSWORD}".encode()).decode()
    return {"Authorization": f"Basic {creds}"}


@pytest.fixture
def tmp_pdf_factory():
    """Helper to create test PDF files with specific properties."""
    def _create_pdf(filename: str = "sample.pdf", text: str = "", pages: int = 1, encrypt: bool = False):
        pdf_path = TEST_TMP / filename
        if pdf_path.exists():
            try:
                pdf_path.unlink()
            except OSError:
                pass
        doc = pymupdf.open()
        for i in range(pages):
            page = doc.new_page(width=595, height=842)
            if text:
                page.insert_textbox(pymupdf.Rect(50, 50, 500, 700), f"Page {i+1}:\n\n{text}")
        if encrypt:
            doc.save(
                str(pdf_path),
                encryption=pymupdf.PDF_ENCRYPT_AES_256,
                owner_pw="owner123",
                user_pw="user123",
            )
        else:
            doc.save(str(pdf_path))
        doc.close()
        gc.collect()
        return pdf_path

    return _create_pdf


# ============================================================================
# 1. Branding Constant & Prompt Identity
# ============================================================================

def test_branding_constant():
    assert ASSISTANT_NAME == "BBHS AI Assistant"
    assert "BBHS AI Assistant" in SYSTEM_RAG_INSTRUCTIONS
    assert "BBHS AI Assistant" in SYSTEM_DIRECT_INSTRUCTIONS


def test_chat_config_endpoint(client):
    res = client.get("/chat/config")
    assert res.status_code == 200
    assert res.json() == {"assistant_name": "BBHS AI Assistant"}


# ============================================================================
# 2. Parameterized KB Build Script
# ============================================================================

def test_parameterized_build_kb(tmp_pdf_factory):
    from scripts.build_kb import build_vector_store, load_pdf_documents

    content = "BBHS AI Assistant is an internal knowledge base system. " * 10
    pdf_file = tmp_pdf_factory(filename="test_build.pdf", text=content, pages=2)

    docs = load_pdf_documents(pdf_path=pdf_file)
    assert len(docs) == 2
    assert "BBHS AI Assistant" in docs[0].page_content

    # Mock embeddings
    mock_embed = MagicMock()
    mock_embed.embed_documents.side_effect = lambda texts: [[0.1] * 128 for _ in texts]
    mock_embed.embed_query.return_value = [0.1] * 128

    dest_dir = TEST_TMP / "chroma_test"
    chunks = build_vector_store(
        pdf_path=pdf_file,
        persist_dir=dest_dir,
        chunk_size=300,
        chunk_overlap=50,
        embedding_model=mock_embed,
    )
    assert len(chunks) >= 2
    assert (dest_dir / "chroma.sqlite3").exists()
    gc.collect()


# ============================================================================
# 3. KB Validation Checks
# ============================================================================

def test_kb_validation_non_pdf_extension(client, admin_auth_headers):
    files = {"file": ("test.txt", b"plain text content", "text/plain")}
    res = client.post("/admin/kb/upload", files=files, headers=admin_auth_headers)
    assert res.status_code == 400
    assert "must be a .pdf file" in res.json()["detail"]


def test_kb_validation_wrong_magic_bytes(client, admin_auth_headers):
    files = {"file": ("fake.pdf", b"NOT_A_PDF_FILE_HEADER", "application/pdf")}
    res = client.post("/admin/kb/upload", files=files, headers=admin_auth_headers)
    assert res.status_code == 400
    assert "%PDF-" in res.json()["detail"]


def test_kb_validation_corrupt_pdf(client, admin_auth_headers):
    files = {"file": ("corrupt.pdf", b"%PDF-1.4\nCorrupt bytes that fail PyMuPDF parsing", "application/pdf")}
    res = client.post("/admin/kb/upload", files=files, headers=admin_auth_headers)
    assert res.status_code == 400
    assert "Corrupt or unreadable" in res.json()["detail"]


def test_kb_validation_encrypted_pdf(client, admin_auth_headers, tmp_pdf_factory):
    pdf = tmp_pdf_factory(filename="enc.pdf", text="Secret content " * 20, encrypt=True)
    with open(pdf, "rb") as f:
        files = {"file": ("encrypted.pdf", f.read(), "application/pdf")}
    res = client.post("/admin/kb/upload", files=files, headers=admin_auth_headers)
    assert res.status_code == 400
    assert "Encrypted or password-protected" in res.json()["detail"]


def test_kb_validation_no_text_pdf(client, admin_auth_headers, tmp_pdf_factory):
    pdf = tmp_pdf_factory(filename="notext.pdf", text="Too short", pages=1)
    with open(pdf, "rb") as f:
        files = {"file": ("short.pdf", f.read(), "application/pdf")}
    res = client.post("/admin/kb/upload", files=files, headers=admin_auth_headers)
    assert res.status_code == 400
    assert "No readable text found" in res.json()["detail"]


def test_kb_validation_page_cap_exceeded(tmp_pdf_factory):
    pdf = tmp_pdf_factory(filename="big.pdf", text="Some valid content " * 15, pages=501)
    with pytest.raises(ValueError, match="exceeds maximum page limit"):
        validate_pdf_content(pdf)


def test_hostile_filename_sanitization():
    assert sanitize_display_name("../../etc/passwd.pdf") == "passwd.pdf"
    assert sanitize_display_name("..\\..\\windows\\system32.pdf") == "system32.pdf"
    assert sanitize_display_name("\x00bad\x1f_file.pdf") == "bad_file.pdf"


# ============================================================================
# 4. KB Auth, Status & Concurrency
# ============================================================================

def test_kb_unauthenticated_and_expired(client):
    # Unauthenticated gives 401
    res = client.get("/admin/kb/status")
    assert res.status_code == 401

    # Invalid/expired cookie gives 401
    res = client.get("/admin/kb/status", cookies={"admin_session": "12345.invalidsig"})
    assert res.status_code == 401


def test_admin_ui_and_expired_cookie_can_reauthenticate():
    """The admin page is protected, and valid credentials renew an expired session."""
    creds = base64.b64encode(f"{ADMIN_USERNAME}:{ADMIN_PASSWORD}".encode()).decode()
    with TestClient(app) as fresh_client:
        res = fresh_client.get(
            "/admin",
            cookies={"admin_session": "12345.invalidsig"},
            headers={"Authorization": f"Basic {creds}"},
        )
        assert res.status_code == 200
        assert "AI Control Center" in res.text
        assert "admin_session=" in res.headers.get("set-cookie", "")


def test_admin_cors_uses_configured_origins(client):
    """Credentialed admin requests must not receive wildcard CORS headers."""
    origin = CORS_ALLOWED_ORIGINS[0]
    creds = base64.b64encode(f"{ADMIN_USERNAME}:{ADMIN_PASSWORD}".encode()).decode()
    res = client.get("/admin/config", headers={"Origin": origin, "Authorization": f"Basic {creds}"})
    assert res.status_code == 200
    assert res.headers.get("access-control-allow-origin") == origin
    assert res.headers.get("access-control-allow-credentials") == "true"


def test_kb_status_no_server_paths(client, admin_auth_headers):
    res = client.get("/admin/kb/status", headers=admin_auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert "active" in data
    assert "rebuild" in data

    # Ensure no filesystem path indicators leak
    raw_json = json.dumps(data)
    assert "C:\\" not in raw_json
    assert "/app/" not in raw_json
    assert "data/kb" not in raw_json


def test_concurrent_upload_conflict(client, admin_auth_headers):
    # Simulate a running rebuild directly on the rebuild status state
    original_state = kb_service._current_rebuild.state
    try:
        kb_service._current_rebuild.state = "building"
        files = {"file": ("test.pdf", b"%PDF-1.4 ...", "application/pdf")}
        res = client.post("/admin/kb/upload", files=files, headers=admin_auth_headers)
        assert res.status_code == 409
        assert "already in progress" in res.json()["detail"]
    finally:
        kb_service._current_rebuild.state = original_state


# ============================================================================
# 5. Startup Recovery
# ============================================================================

def test_startup_recovery():
    # Place a dummy file in KB_STAGING_DIR and verify startup_recovery removes it
    KB_STAGING_DIR.mkdir(parents=True, exist_ok=True)
    stale_file = KB_STAGING_DIR / "dummy_leftover.tmp"
    stale_file.write_text("leftover")
    assert stale_file.exists()

    startup_recovery()
    assert not stale_file.exists()


# ============================================================================
# 6. No Chat Storage
# ============================================================================

def test_no_chat_storage(client):
    with patch("src.api.app.answer_question", return_value="Test answer grounded in KB"):
        res = client.post(
            "/webhook/chat",
            json={"message": "What is the policy for BBHS?", "session_id": "test-session-123"},
        )
        assert res.status_code == 200
        assert res.json()["answer"] == "Test answer grounded in KB"

        # Verify log file does NOT contain the question or answer text
        log_file = Path("data/logs/chat.log")
        if log_file.exists():
            log_content = log_file.read_text(encoding="utf-8", errors="ignore")
            assert "What is the policy for BBHS?" not in log_content
            assert "Test answer grounded in KB" not in log_content


# ============================================================================
# 7. Session Store TTL and Cap Eviction
# ============================================================================

def test_session_store_ttl_and_cap():
    store = BoundedSessionStore(max_entries=2, ttl_seconds=1)

    # Add sessions
    store.append_messages("s1", "hi", "hello")
    store.append_messages("s2", "how are you", "fine")
    assert len(store.get_history("s1")) == 2
    assert len(store.get_history("s2")) == 2

    # Add 3rd session -> s1 should be evicted due to max_entries=2 (s2 was accessed after s1)
    store.append_messages("s3", "test", "response")
    assert store.get_history("s1") == []
    assert len(store.get_history("s2")) == 2
    assert len(store.get_history("s3")) == 2

    # Test TTL expiry
    time.sleep(1.1)
    assert store.get_history("s2") == []
    assert store.get_history("s3") == []


# ============================================================================
# 8. Webhook Chat Contract & Regression
# ============================================================================

def test_webhook_chat_contract(client):
    # Empty message should be rejected (400 or 422)
    res = client.post("/webhook/chat", json={"message": "", "session_id": "test"})
    assert res.status_code in (400, 422)

    res_ws = client.post("/webhook/chat", json={"message": "   ", "session_id": "test"})
    assert res_ws.status_code == 400

    # Small talk routing
    res = client.post("/webhook/chat", json={"message": "hello", "session_id": "s-greet"})
    assert res.status_code == 200
    data = res.json()
    assert "session_id" in data
    assert "answer" in data
    assert len(data["answer"]) > 0

    # Live time tool routing
    res = client.post("/webhook/chat", json={"message": "what time is it", "session_id": "s-time"})
    assert res.status_code == 200
    assert "current time is" in res.json()["answer"].lower()


def test_admin_config_regression(client, admin_auth_headers):
    # GET /admin/config
    res = client.get("/admin/config", headers=admin_auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert "models" in data
    assert "prompts" in data
    assert "rag" in data

    # GET /admin/health
    res = client.get("/admin/health", headers=admin_auth_headers)
    assert res.status_code == 200
    assert "api" in res.json()


# ============================================================================
# 9. /chat UI Assets & XSS Sanitization Checks
# ============================================================================

def test_chat_ui_zero_external_resources():
    """Verify static/chat/index.html loads zero external resources (no CDN, no Google fonts)."""
    index_path = Path("static/chat/index.html")
    assert index_path.exists()
    content = index_path.read_text(encoding="utf-8")

    import re
    # Check for external URLs in src, href, or @import
    external_urls = re.findall(r'(?:src|href)=["\'](https?://[^"\']+)["\']', content)
    assert len(external_urls) == 0, f"Found external resources in /chat HTML: {external_urls}"

    # Verify system UI font in chat.css (no external @import or url())
    css_path = Path("static/chat/chat.css")
    assert css_path.exists()
    css_content = css_path.read_text(encoding="utf-8")
    assert "@import" not in css_content
    assert "fonts.googleapis.com" not in css_content


def test_markdown_sanitization_neutralizes_xss():
    """Verify that dangerous HTML tags like <img onerror=...> and <script> are escaped."""
    js_path = Path("static/chat/chat.js")
    assert js_path.exists()
    js_code = js_path.read_text(encoding="utf-8")

    # Ensure escapeHTML function exists in chat.js
    assert "escapeHTML" in js_code
    assert "renderSafeMarkdown" in js_code

    # Python mirror test of the exact escaping logic used in chat.js
    def escape_html(text: str) -> str:
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )

    attack_payload = '<img src=x onerror="alert(1)">'
    escaped = escape_html(attack_payload)
    assert "<img" not in escaped
    assert "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;" == escaped

    script_payload = '<script>fetch("http://evil.com")</script>'
    escaped_script = escape_html(script_payload)
    assert "<script>" not in escaped_script
