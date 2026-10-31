"""Configuration and constants for BBHS AI Assistant."""

import os
import re
from pathlib import Path

# ============================================================================
# Directory and File Paths
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
PERSIST_DIRECTORY = BASE_DIR / "data/chroma_db"
CACHE_FILE = BASE_DIR / "data/cache/chat_cache.json"
BM25_CACHE_FILE = BASE_DIR / "data/cache/bm25_index.pkl"
LOG_DIR = BASE_DIR / "data/logs"
LOG_FILE = LOG_DIR / "chat.log"
ADMIN_CONFIG_FILE = BASE_DIR / "data/admin_config.json"
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")
ADMIN_SESSION_TTL_SECONDS = 300
ADMIN_COOKIE_SECURE = os.getenv("ADMIN_COOKIE_SECURE", "false").lower() == "true"
CORS_ALLOWED_ORIGINS = tuple(
    origin.strip()
    for origin in os.getenv(
        "CORS_ALLOWED_ORIGINS",
        "http://localhost:8122,http://127.0.0.1:8122",
    ).split(",")
    if origin.strip()
)

# ============================================================================
# Knowledge Base (KB) Configuration
# ============================================================================

KB_DIR = BASE_DIR / "data/kb"
KB_STAGING_DIR = BASE_DIR / "data/kb_staging"
KB_MANIFEST_FILE = BASE_DIR / "data/kb_manifest.json"
KB_CATALOG_FILE = BASE_DIR / "data/kb_catalog.json"
KB_MAX_UPLOAD_MB = int(os.getenv("KB_MAX_UPLOAD_MB", "25"))
KB_MAX_PAGES = int(os.getenv("KB_MAX_PAGES", "500"))
KB_SUPPORTED_FORMATS = ("pdf", "docx", "notice", "faq", "web_content")
KB_STATUSES = ("Draft", "Review", "Approved", "Published", "Disabled", "Archived")
KB_CATEGORIES = (
    "Admissions",
    "Academics",
    "Fees & Payments",
    "Transport",
    "Examinations",
    "Code of Conduct",
    "General",
)
KB_CAMPUSES = ("All Campuses", "Main Campus", "Primary Wing", "High School")

# ============================================================================
# Model and Embedding Configuration
# ============================================================================

EMBEDDING_MODEL = "embeddinggemma:latest"
OLLAMA_BASE_URL = os.getenv("OLLAMA_HOST", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
COLLECTION_NAME = "BBHS_KB"
CHAT_MODEL = "openai/gpt-oss-20b"

# ============================================================================
# Retrieval Parameters
# ============================================================================

VECTOR_TOP_K = 4
BM25_TOP_K = 4
FINAL_TOP_K = 6
MAX_CONTEXT_CHARS = 3500

# ============================================================================
# Chat and Cache Configuration
# ============================================================================

ASSISTANT_NAME = os.getenv("ASSISTANT_NAME", "BBHS AI Assistant")
MAX_HISTORY_MESSAGES = 6
MAX_HISTORY_CHARS = 4000
MAX_HISTORY_MESSAGE_CHARS = 1600
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "1800"))
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "1000"))
MAX_SESSION_ENTRIES = MAX_SESSIONS
CACHE_VERSION = 2
MAX_CACHE_ENTRIES = 200
SEMANTIC_CACHE_THRESHOLD = 0.82
SUGGESTED_QUESTIONS: list[str] = []

# ============================================================================
# Regex Patterns
# ============================================================================

TOKEN_PATTERN = re.compile(r"\b\w+\b")
GREETING_PATTERN = re.compile(r"^(hi|hello|hey|good morning|good afternoon|good evening)\b")
TIME_QUERY_PATTERN = re.compile(
    r"\b(what(?:'s| is)? the time|what time is it|current time|time now|tell me the time)\b"
)
CHAT_TIMEZONE = os.getenv("CHAT_TIMEZONE", "Asia/Kolkata")
TIME_DISPLAY_FORMAT = "%A, %d %B %Y at %I:%M:%S %p"

# Verified school contacts used for relevant next steps.
ADMISSION_PHONE = "8147883880"
ADMISSION_EMAIL = "admission@baldwinboyshighschool.edu.in"
FEE_ENQUIRY_PHONE = "8147883880"
GENERAL_ENQUIRY_PHONE = "8147883880"
GENERAL_ENQUIRY_EMAIL = "enquiry@baldwinboyshighschool.edu.in"
ONLINE_APPLICATION_URL = "https://portal.baldwinboyshighschool.edu.in/#/auth/login"

# Conversational Agent and Lead Configuration
ENABLE_CONVERSATIONAL_AGENT = os.getenv("ENABLE_CONVERSATIONAL_AGENT", "true").lower() == "true"
LEADS_FILE = BASE_DIR / "data/leads.json"
INDIAN_PHONE_PATTERN = re.compile(r"^(?:\+91|91|0)?[6-9]\d{9}$")

# ============================================================================
# Prompt Hints
# ============================================================================

DIRECT_CONTENT_HINTS = (
    "base the questions on the following content",
    "return only the json",
    "use this exact json structure",
    "following content:",
)

# ============================================================================
# Helper Functions
# ============================================================================

def normalize_question(question: str) -> str:
    """Normalize a question by lowercasing and collapsing whitespace."""
    return " ".join(question.lower().split())

# ============================================================================
# API Configuration
# ============================================================================

API_HOST = "0.0.0.0"
API_PORT = int(os.getenv("API_PORT", "8122"))
API_TITLE = "BBHS AI Assistant API"

# ============================================================================
# LLM Configuration
# ============================================================================

LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 1024
LLM_TEMPERATURE_CHAT = 0.7

# ============================================================================
# PDF Configuration
# ============================================================================

# Directory containing the source PDF/DOCX files used for a knowledge-base build.
# Admin uploads are stored separately in data/kb using safe content-hash names.
PDF_PATH = BASE_DIR / "KB_PDF"
PDF_CHUNK_SIZE = 500
PDF_CHUNK_OVERLAP = 80
