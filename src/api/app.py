"""FastAPI application and endpoints for BBHS AI Assistant."""

import os
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from langchain_groq import ChatGroq

load_dotenv()

from src.admin.config_service import default_config, load_config
from src.admin.kb_service import startup_recovery
from src.api.admin import require_admin_access, router as admin_router
from src.api.kb_routes import router as kb_router
from src.api.schemas import ChatRequest, ChatResponse
from src.config import (
    API_HOST,
    API_PORT,
    API_TITLE,
    ASSISTANT_NAME,
    CHAT_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    MAX_HISTORY_MESSAGES,
    MAX_SESSION_ENTRIES,
    MAX_SESSIONS,
    SESSION_TTL_SECONDS,
)
from src.core.rag import answer_question
from src.core.retrievers import create_ensemble_retriever, load_bm25_index
from src.storage.cache import load_cache
from src.storage.vector_db import load_vector_store
from src.utils.logging import log_error, log_step


class BoundedSessionStore:
    """Thread-safe bounded in-memory session history store with LRU eviction and TTL."""

    def __init__(
        self,
        max_entries: int = MAX_SESSIONS,
        ttl_seconds: int = SESSION_TTL_SECONDS,
    ) -> None:
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        # session_id -> (history_list, last_active_timestamp)
        self._store: OrderedDict[str, tuple[list[dict[str, str]], float]] = OrderedDict()
        self._lock = threading.Lock()

    def _purge_expired(self, now: float) -> None:
        """Evict sessions that have exceeded the idle TTL under lock."""
        expired = [sid for sid, (_, last_active) in self._store.items() if now - last_active > self.ttl_seconds]
        for sid in expired:
            del self._store[sid]

    def get_history(self, session_id: str) -> list[dict[str, str]]:
        """Retrieve conversation history if not expired, updating access time."""
        now = time.time()
        with self._lock:
            self._purge_expired(now)
            if session_id in self._store:
                history, _ = self._store[session_id]
                self._store[session_id] = (history, now)
                self._store.move_to_end(session_id)
                return list(history)
            return []

    def append_messages(self, session_id: str, user_message: str, ai_response: str) -> None:
        """Append messages to session history with TTL and capacity bounds."""
        now = time.time()
        with self._lock:
            self._purge_expired(now)
            history = list(self._store[session_id][0]) if session_id in self._store else []
            history.append({"role": "user", "content": user_message})
            history.append({"role": "assistant", "content": ai_response})
            if len(history) > MAX_HISTORY_MESSAGES:
                history = history[-MAX_HISTORY_MESSAGES:]
            self._store[session_id] = (history, now)
            self._store.move_to_end(session_id)
            while len(self._store) > self.max_entries:
                self._store.popitem(last=False)


session_store = BoundedSessionStore()


def _runtime_config():
    """Load admin settings while retaining startup safety if the file is invalid."""
    try:
        return load_config()
    except ValueError as exc:
        log_error("ADMIN_CONFIG_LOAD_FAILED", exc)
        return default_config()


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """Manage application startup and shutdown lifecycle."""
    log_step("SERVER_STARTING")

    # Clean leftover staging files and recover backup if needed
    startup_recovery(app_instance)

    runtime_config = _runtime_config()

    try:
        app_instance.state.vector_db = load_vector_store()
        app_instance.state.bm25_index = load_bm25_index(app_instance.state.vector_db)
        app_instance.state.ensemble_retriever = create_ensemble_retriever(
            app_instance.state.vector_db,
            app_instance.state.bm25_index,
            vector_top_k=runtime_config.rag.vector_top_k,
            bm25_top_k=runtime_config.rag.bm25_top_k,
            final_top_k=runtime_config.rag.final_top_k,
            similarity_threshold=max(
                runtime_config.rag.similarity_threshold,
                runtime_config.guardrails.minimum_retrieval_score,
            ),
        )
    except Exception as exc:
        log_error("VECTOR_STORE_INIT_NOTICE", exc)
        app_instance.state.vector_db = None
        app_instance.state.bm25_index = None
        app_instance.state.ensemble_retriever = None

    app_instance.state.llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model=runtime_config.models.primary_model,
        temperature=runtime_config.models.temperature,
        max_tokens=runtime_config.models.max_tokens,
        timeout=runtime_config.models.request_timeout,
    )
    # In-memory response cache only (D3)
    app_instance.state.cache = load_cache()
    log_step("SERVER_READY", model=runtime_config.models.primary_model)
    yield
    log_step("SERVER_STOPPED")


# Initialize FastAPI app with lifespan
app = FastAPI(title=API_TITLE, lifespan=lifespan)
app.include_router(admin_router)
app.include_router(kb_router)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
_STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "static"
if _STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


def _get_resources():
    """Ensure runtime resources are initialized even if lifespan wasn't triggered."""
    runtime_config = _runtime_config()

    if not hasattr(app.state, "vector_db") or app.state.vector_db is None:
        try:
            app.state.vector_db = load_vector_store()
        except Exception:
            app.state.vector_db = None

    if not hasattr(app.state, "bm25_index") or app.state.bm25_index is None:
        if app.state.vector_db is not None:
            try:
                app.state.bm25_index = load_bm25_index(app.state.vector_db)
            except Exception:
                app.state.bm25_index = None
        else:
            app.state.bm25_index = None

    if not hasattr(app.state, "ensemble_retriever") or app.state.ensemble_retriever is None:
        if app.state.vector_db is not None and app.state.bm25_index is not None:
            app.state.ensemble_retriever = create_ensemble_retriever(
                app.state.vector_db,
                app.state.bm25_index,
                vector_top_k=runtime_config.rag.vector_top_k,
                bm25_top_k=runtime_config.rag.bm25_top_k,
                final_top_k=runtime_config.rag.final_top_k,
                similarity_threshold=max(
                    runtime_config.rag.similarity_threshold,
                    runtime_config.guardrails.minimum_retrieval_score,
                ),
            )
        else:
            app.state.ensemble_retriever = None

    if not hasattr(app.state, "llm") or app.state.llm is None:
        app.state.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model=runtime_config.models.primary_model,
            temperature=runtime_config.models.temperature,
            max_tokens=runtime_config.models.max_tokens,
            timeout=runtime_config.models.request_timeout,
        )

    if not hasattr(app.state, "cache") or app.state.cache is None:
        app.state.cache = load_cache()

    return (
        app.state.vector_db,
        app.state.bm25_index,
        app.state.llm,
        app.state.cache,
        app.state.ensemble_retriever,
    )


@app.get("/")
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/chat", include_in_schema=False)
def chat_ui() -> FileResponse:
    """Serve the Gemini-style BBHS AI Assistant chat UI."""
    chat_file = _STATIC_DIR / "chat" / "index.html"
    if chat_file.is_file():
        return FileResponse(str(chat_file))
    return FileResponse(str(_STATIC_DIR / "index.html"))


@app.get("/chat/config")
def chat_config() -> dict[str, str]:
    """Return lightweight public branding configuration for the chat frontend."""
    return {"assistant_name": ASSISTANT_NAME}


@app.get("/admin", include_in_schema=False, dependencies=[Depends(require_admin_access)])
def admin_ui() -> FileResponse:
    """Serve the internal AI Control Center UI."""
    return FileResponse(str(_STATIC_DIR / "admin.html"))


@app.post("/webhook/chat", response_model=ChatResponse)
def webhook_chat(payload: ChatRequest, background_tasks: BackgroundTasks) -> ChatResponse:
    """
    Main chat webhook endpoint.

    Args:
        payload: Chat request with message and optional session_id
        background_tasks: Background task queue

    Returns:
        Chat response with session_id and answer

    Raises:
        HTTPException: If message is empty or processing fails
    """
    message = payload.message.strip()
    session_id = payload.session_id.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message is required")

    vector_db, bm25_index, llm, cache, ensemble_retriever = _get_resources()
    chat_history = session_store.get_history(session_id)

    # Graceful handling when knowledge base has not been indexed yet
    if vector_db is None or ensemble_retriever is None:
        from src.core.prompts import classify_query
        from src.core.tools import call_time_tool
        route = classify_query(message)
        if route == "tool_call":
            return ChatResponse(session_id=session_id, answer=call_time_tool())
        if route == "small_talk":
            return ChatResponse(
                session_id=session_id,
                answer=f"Hello! I am {ASSISTANT_NAME}. The knowledge base is currently being set up. How can I assist you?",
            )
        return ChatResponse(
            session_id=session_id,
            answer=f"The knowledge base is currently empty or initializing. Please upload a knowledge base document in the Control Center.",
        )

    try:
        total_start = time.perf_counter()
        # Message text omitted from logs per D3 (no chat storage)
        log_step("QUERY_START", query_len=len(message), session_id=session_id)

        answer = answer_question(
            vector_db=vector_db,
            bm25_index=bm25_index,
            llm=llm,
            cache=cache,
            chat_history=chat_history,
            question=message,
            ensemble_retriever=ensemble_retriever,
        )
        session_store.append_messages(session_id, message, answer)

        log_step("BOT_MESSAGE", answer_len=len(answer), session_id=session_id)

        total_latency_ms = int((time.perf_counter() - total_start) * 1000)
        log_step("TOTAL", latency=f"{total_latency_ms}ms", session_id=session_id)
        return ChatResponse(session_id=session_id, answer=answer)
    except HTTPException:
        raise
    except Exception as exc:
        log_error("API_ERROR", exc, session_id=session_id, query_len=len(message))
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing your request.",
        ) from exc


def run(host: str = API_HOST, port: int = API_PORT, reload: bool = True) -> None:
    """Run the FastAPI server with a single worker per D5."""
    import uvicorn
    uvicorn.run("src.api.app:app", host=host, port=port, reload=reload)
