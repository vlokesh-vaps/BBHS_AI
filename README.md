# BBHS AI Assistant - Pure RAG Chatbot

This is the **BBHS AI Assistant**, a **pure Retrieval Augmented Generation (RAG) assistant** designed to answer questions strictly from the active knowledge base. It combines vector similarity search, keyword-based retrieval (BM25 with Reciprocal Rank Fusion), and LLM-generated responses to provide accurate answers grounded in real documentation.

---

## Key Capabilities & Core Mission

- ✅ **Gemini-style Chat UI (`/chat`)**: Clean, responsive, full-height chat interface with zero external CDN dependencies, system fonts, dark/light theme support, copy buttons, stop generation, and client-side XSS sanitization.
- ✅ **Knowledge Base Ingestion**: Build the vector database from all PDF and DOCX files in `KB_PDF/`, or upload managed PDF/DOCX documents through the AI Control Center.
- ✅ **Strictly No Chat Storage (D3)**: Conversations are never stored on disk, databases, or client-side storage (`localStorage`/`sessionStorage`/cookies). Logs strictly record request metadata and latency, never user message or answer text. Cache is in-memory only.
- ✅ **Pure RAG**: No lead capture, no sales funnels, no accounts.
- ✅ **Hybrid Retrieval**: Combines vector similarity (Chroma) + BM25 keyword retrieval using Reciprocal Rank Fusion (RRF).
- ✅ **AI Control Center (`/admin`)**: Protected Bootstrap 5 SPA for model settings, RAG parameters, knowledge base management, and component health monitoring.
- ✅ **Single Worker Architecture (D5)**: Assumes a single Uvicorn process to ensure consistent in-memory session state, cache, and rebuild locking.

---

## Project Structure

```
AIChat_BBHS/
├── src/                                # Main application source code
│   ├── __init__.py
│   ├── config.py                       # Central configuration constants
│   ├── core/                           # Core RAG pipeline
│   │   ├── __init__.py
│   │   ├── rag.py                      # Multi-route pipeline (answer_question, hybrid search)
│   │   ├── retrievers.py               # BM25Index, VectorRetriever, EnsembleRetriever
│   │   ├── prompts.py                  # Prompt templates and query classification
│   │   └── tools.py                    # Deterministic tools (current time)
│   ├── admin/                          # Control Center services
│   │   ├── config_service.py           # Validated atomic JSON configuration persistence
│   │   ├── health_service.py           # Component health monitoring
│   │   ├── kb_service.py               # PDF validation, staging build, atomic swap & recovery
│   │   ├── provider_service.py         # Provider and model catalog
│   │   └── schemas.py                  # Pydantic admin and KB contracts
│   ├── api/                            # REST API layer
│   │   ├── __init__.py
│   │   ├── app.py                      # FastAPI app initialization, routes, lifespan
│   │   ├── admin.py                    # Control Center API routes
│   │   ├── kb_routes.py                # Knowledge Base upload and status endpoints
│   │   └── schemas.py                  # Pydantic request/response models
│   ├── storage/                        # Storage layer
│   │   ├── __init__.py
│   │   ├── cache.py                    # In-memory exact and semantic cache
│   │   ├── vector_db.py                # Chroma vector database utilities
│   │   └── datastore.py                # Unused legacy datastore (chat storage disabled)
│   └── utils/                          # Cross-cutting utilities
│       ├── __init__.py
│       └── logging.py                  # Sanitized structured logging helpers
├── scripts/                            # Operational & CLI scripts
│   ├── __init__.py
│   ├── build_kb.py                     # Parameterized vector DB builder
│   └── cli_chat.py                     # Terminal-based chat interface
├── static/                             # Static frontend files
│   ├── admin.html                      # AI Control Center HTML
│   ├── admin.css                       # Control Center styling
│   ├── admin.js                        # Control Center logic & KB upload handler
│   └── chat/                           # Gemini-style Chat UI
│       ├── index.html                  # End-user chat page
│       ├── chat.css                    # Responsive themes & styling
│       └── chat.js                     # In-memory transcript & safe markdown
├── data/                               # Persistent runtime data
│   ├── kb/                             # Permanently stored active PDF (<sha256[:12]>.pdf)
│   ├── kb_manifest.json                # Active knowledge base manifest
│   ├── chroma_db/                      # Chroma persistent vector database
│   ├── cache/                          # BM25 index cache
│   └── logs/                           # System logs (no message text)
├── tests/                              # Comprehensive test suite
│   └── test_bbhs.py                    # KB upload, no chat storage, auth & regression tests
├── app.py                              # Root API server entry point
├── main.py                             # Root CLI chat entry point
├── .env                                # Local secrets and credentials
├── requirements.txt                    # Project dependencies (including python-multipart)
├── Dockerfile                          # Container runtime specification
├── docker-compose.yml                  # Compose setup with Ollama integration
└── README.md                           # Documentation
```

---

## Invariants & Design Decisions

| ID | Decision | Description |
|---|---|---|
| **D1** | **Display Name** | The assistant is named **"BBHS AI Assistant"**, configured centrally via `ASSISTANT_NAME` in `src/config.py`. |
| **D2** | **Controlled Knowledge Base** | A rebuild indexes all PDF/DOCX files supplied in `KB_PDF/`; managed admin documents are validated, staged, indexed, and published through the approval workflow. |
| **D3** | **No Chat Storage** | Messages are never stored to external datastores, disk files, or browser storage (`localStorage`, `sessionStorage`, `IndexedDB`, cookies). Cache is in-memory only. Logs never record user query or answer text. |
| **D4** | **Public `/chat` UI** | Public end-user interface inspired by Gemini's layout, loading zero external resources (no external CDNs or Google Fonts). |
| **D5** | **Single Worker** | Single Uvicorn worker process assumed (`--workers 1`) to preserve in-memory session history, cache, and rebuild locking. |
| **D6** | **Auth Isolation** | `/chat` and `/webhook/chat` are public; all `/admin/*` and `/admin/kb/*` endpoints require signed admin session authentication. |
| **D7** | **Zero External Resources** | The `/chat` page loads no external CSS/JS/fonts; inline SVGs and system font stacks are used. |

---

## API Endpoints

### Public Endpoints
- **`GET /`**: Health check returning `{"status": "ok"}`
- **`GET /chat`**: Gemini-style BBHS AI Assistant UI
- **`GET /chat/config`**: Public branding configuration returning `{"assistant_name": "BBHS AI Assistant"}`
- **`POST /webhook/chat`**: Main chat API
  - Request: `{"message": "What are the school timings?", "session_id": "optional-id"}`
  - Response: `{"session_id": "optional-id", "answer": "..."}`

### Admin Endpoints (Requires Admin Basic Auth or Session Cookie)
- **`GET /admin`**: AI Control Center UI
- **`GET /admin/config`**: Get current configuration without secrets
- **`PUT /admin/config`**: Update configuration partially and atomically
- **`POST /admin/config/test`**: Test configuration in-memory without applying to production
- **`GET /admin/health`**: Real dependency health checks
- **`GET /admin/providers`**: Catalog of supported providers and models
- **`GET /admin/kb/status`**: Active knowledge base metadata and rebuild state
- **`POST /admin/kb/upload`**: Multipart file upload (`file: <pdf>`). Returns `202 Accepted` or `409 Conflict`.

---

## Operational Notes & Follow-ups

1. **Saved Admin Prompts Override Code Defaults**: If you previously configured or saved custom prompt templates in `data/admin_config.json`, those saved templates take precedence over code defaults. You can view and edit them in `/admin` under **Prompts**.
2. **Old Disk Cache Cleanup**: If a previous version created `data/cache/chat_cache.json`, it may contain historical chat records. Delete `data/cache/chat_cache.json` manually if you want all prior traces cleared.
3. **Session Expiry**: Admin sessions expire after 5 minutes of inactivity (`ADMIN_SESSION_TTL_SECONDS = 300`). End-user in-memory chat sessions expire after 30 minutes of idle time (`SESSION_TTL_SECONDS = 1800`).

---

## Running the Application

### 1. Local Development
```powershell
# Run the API server
python app.py
# Or with Uvicorn directly
uvicorn src.api.app:app --host 0.0.0.0 --port 8122 --workers 1
```
Visit:
- Chat Interface: `http://localhost:8122/chat`
- Control Center: `http://localhost:8122/admin`

For a full knowledge-base rebuild, place `.pdf` and `.docx` files in `KB_PDF/` and run `python -m scripts.build_kb`. Configure production browser origins with the comma-separated `CORS_ALLOWED_ORIGINS` environment variable.

### 2. Running Tests
```powershell
python -m pytest -v tests/test_bbhs.py
```

### 3. Docker Deployment
```powershell
docker compose up -d --build
```
