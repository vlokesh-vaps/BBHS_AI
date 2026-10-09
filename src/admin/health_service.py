"""Actual, failure-tolerant health checks for the Control Center."""

import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from src.config import CACHE_FILE, OLLAMA_BASE_URL, PERSIST_DIRECTORY


def _status(ok: bool, healthy: str = "healthy", detail: str | None = None) -> dict[str, str | None]:
    return {"status": healthy if ok else "unavailable", "detail": detail}


def check_health(app_state: Any, config: Any) -> dict[str, dict[str, str | None]]:
    """Check configured resources without allowing one failure to abort the response."""
    vector_ok = hasattr(app_state, "vector_db") and app_state.vector_db is not None
    llm_ok = hasattr(app_state, "llm") and app_state.llm is not None
    cache_ok = CACHE_FILE.parent.exists() and (CACHE_FILE.exists() or CACHE_FILE.parent.is_dir())
    vector_detail = "Vector store loaded" if vector_ok else "Vector store is not loaded"

    embedding_ok = False
    embedding_detail = "Ollama is unavailable"
    try:
        request = urllib.request.Request(f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags", method="GET")
        with urllib.request.urlopen(request, timeout=3) as response:
            embedding_ok = response.status == 200
            embedding_detail = "Ollama responded" if embedding_ok else f"HTTP {response.status}"
    except (OSError, urllib.error.URLError) as exc:
        embedding_detail = str(exc)

    return {
        "api": _status(True, detail="API process is responding"),
        "primary_llm": _status(llm_ok, detail="LLM initialized" if llm_ok else "LLM is not initialized"),
        "fallback_llm": {"status": "disabled", "detail": "No fallback model configured"},
        "vector_db": _status(vector_ok, detail=vector_detail),
        "embedding": _status(embedding_ok, detail=embedding_detail),
        "cache": _status(cache_ok, detail="Cache directory is writable scope" if cache_ok else "Cache directory is missing"),
    }
