"""Backend-owned provider/model catalog for the admin UI."""

from src.config import CHAT_MODEL, EMBEDDING_MODEL


def supported_providers() -> list[dict[str, object]]:
    """Return supported provider choices without exposing credentials."""
    return [
        {"id": "groq", "name": "Groq", "models": [CHAT_MODEL]},
        {"id": "ollama", "name": "Ollama", "models": [EMBEDDING_MODEL]},
    ]
