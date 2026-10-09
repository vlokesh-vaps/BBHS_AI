# AGENTS.md - Developer & AI Agent Guidelines

Welcome to the **BBHS AI Assistant** repository. This document serves as the primary technical specification, operational guide, and rulebook for AI coding agents and developers working on this codebase.

---

## 1. Project Overview & Core Mission

**BBHS AI Assistant** is a specialized, production-ready **Pure Retrieval-Augmented Generation (RAG) Assistant** designed to answer questions strictly from the active knowledge base document.

### Binding Invariants

1. **Pure RAG Focus**: Dedicated exclusively to accurate, grounded question-answering. Do **NOT** add lead collection forms, contact collection funnels, sales qualification gates, or marketing prompts.
2. **Strict Grounding**: The LLM must answer questions using retrieved context from the active knowledge base. If context is missing or irrelevant, it must gracefully inform the user rather than hallucinating.
3. **One Active KB PDF (D2)**: There is exactly one active knowledge base PDF at any time, tracked in `data/kb_manifest.json`. Administrators can upload a new PDF in the AI Control Center (`/admin`), which validates, stages, rebuilds the vector store and BM25 index, and hot-swaps without downtime.
4. **Strictly No Chat Storage (D3)**: Conversations are never persisted to disk, databases, external datastores, or client storage (`localStorage`/`sessionStorage`/cookies). Cache is in-memory only. Logs strictly record length and metadata, never question or answer text.
5. **Centralized Configuration**: Default thresholds, model identifiers, paths, and limits reside in `src/config.py`; administrator-editable runtime values are validated and atomically persisted by `src/admin/config_service.py`. Saved prompts in `data/admin_config.json` take precedence over code defaults.
6. **Single Uvicorn Worker (D5)**: The application is designed to run with `--workers 1` to ensure in-memory sessions, cache, and rebuild synchronization remain coherent.

---

## 2. System Architecture & Directory Structure

```
AIChat_BBHS/
├── src/                                # Main application package
│   ├── __init__.py
│   ├── config.py                       # Single source of truth for constants
│   ├── core/                           # Core RAG engine
│   │   ├── __init__.py
│   │   ├── rag.py                      # Multi-route pipeline (answer_question, hybrid_search)
│   │   ├── retrievers.py               # BM25Index, VectorRetriever, EnsembleRetriever
│   │   ├── prompts.py                  # Prompt templates, formatting, query classification
│   │   └── tools.py                    # Deterministic tools (current time)
│   ├── admin/                          # Control Center services
│   │   ├── config_service.py           # Validated atomic JSON configuration persistence
│   │   ├── health_service.py           # Real dependency health checks
│   │   ├── kb_service.py               # PDF validation, staging build, atomic swap & recovery
│   │   ├── provider_service.py         # Backend provider/model catalog
│   │   └── schemas.py                  # Pydantic admin and KB contracts
│   ├── api/                            # REST API service layer
│   │   ├── __init__.py
│   │   ├── app.py                      # FastAPI app initialization, middleware, routes
│   │   ├── admin.py                    # Control Center API routes
│   │   ├── kb_routes.py                # Knowledge Base upload and status endpoints
│   │   └── schemas.py                  # Pydantic schemas (ChatRequest, ChatResponse)
│   ├── storage/                        # Persistence & memory layer
│   │   ├── __init__.py
│   │   ├── cache.py                    # In-memory exact and semantic cache
│   │   ├── vector_db.py                # Chroma vector store loading and persistence
│   │   └── datastore.py                # Deprecated/unused datastore (chat storage disabled per D3)
│   └── utils/                          # Cross-cutting utilities
│       ├── __init__.py
│       └── logging.py                  # Structured console and file logger (sanitized per D3)
├── scripts/                            # Operational & CLI scripts
│   ├── __init__.py
│   ├── build_kb.py                     # Parameterized PDF ingestion and Chroma builder
│   └── cli_chat.py                     # Interactive terminal chat interface
├── static/                             # Static web assets
│   ├── admin.html / .css / .js         # AI Control Center SPA
│   └── chat/                           # Gemini-style Chat UI
│       ├── index.html                  # End-user chat page
│       ├── chat.css                    # Theme and responsive styles
│       └── chat.js                     # In-memory transcript & safe markdown
├── data/                               # Data persistence directory
│   ├── kb/                             # Permanently stored active PDF (<sha256[:12]>.pdf)
│   ├── kb_manifest.json                # Active knowledge base manifest
│   ├── chroma_db/                      # Chroma persistent SQLite & parquet files
│   ├── cache/                          # BM25 index cache (bm25_index.pkl)
│   └── logs/                           # System logs (chat.log, sanitized)
├── tests/                              # Comprehensive test suite
│   └── test_bbhs.py                    # Verification tests
├── app.py                              # Root backward-compatible FastAPI entry point
├── main.py                             # Root backward-compatible CLI chat entry point
├── requirements.txt                    # Project Python dependencies
├── Dockerfile                          # Container specification
├── docker-compose.yml                  # Service orchestration
├── .env                                # Local environment secrets
└── README.md                           # User-facing documentation
```

---

## 3. Technology Stack & External Services

| Component | Technology | Details |
|---|---|---|
| **Language** | Python 3.10+ | Strict type hints, modern syntax |
| **API Framework** | FastAPI + Uvicorn | Single worker (`--workers 1`), CORS enabled |
| **Data Validation** | Pydantic v2 | Request/Response payload validation |
| **LLM Inference** | Groq API (`langchain-groq`) | Model: `openai/gpt-oss-20b` (configurable) |
| **Embeddings** | Ollama (`langchain-ollama`) | Model: `embeddinggemma:latest` |
| **Vector Store** | ChromaDB (`langchain-chroma`) | Persisted in `data/chroma_db/` |
| **Keyword Search** | Custom BM25 | Reciprocal Rank Fusion with Vector search |
| **Document Processing** | PyMuPDF (`fitz`), `pymupdf4llm` | Structure-preserving Markdown chunking |
| **Control Center** | Bootstrap 5 + Vanilla JS | Signed session cookies (`ADMIN_SESSION_TTL_SECONDS = 300`) |
| **Chat Frontend** | Vanilla HTML/CSS/JS | Gemini layout, in-memory transcript, zero external CDNs |

---

## 4. API Request Contracts

### Chat Webhook (`POST /webhook/chat`)
```json
{
  "message": "What is the policy for BBHS?",
  "session_id": "optional-session-id"
}
```

Response:
```json
{
  "session_id": "optional-session-id",
  "answer": "..."
}
```

### Knowledge Base Status (`GET /admin/kb/status`)
```json
{
  "active": {
    "display_name": "...",
    "size_bytes": 0,
    "pages": 0,
    "chunks": 0,
    "uploaded_at": "ISO-8601",
    "sha256": "..."
  },
  "rebuild": {
    "state": "idle|validating|building|swapping|done|failed",
    "message": "...",
    "started_at": null,
    "finished_at": null
  }
}
```
*Note: Never returns server filesystem paths.*

---

## 5. Development Guidelines & Invariants for Agents

1. **Rule 1: Think Before Coding**: Always assess existing architecture before modifying files. Code wins over stale documentation.
2. **Rule 2: Simplicity First**: Minimum code footprint that meets the requirements. No unneeded abstraction layers.
3. **Rule 3: Surgical Changes**: Touch only necessary files, follow established conventions, preserve existing docstrings.
4. **Rule 4: Goal-Driven Execution**: Run verification commands and include concrete outputs.
