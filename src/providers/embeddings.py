"""Pluggable embedding provider protocol and implementations."""

import urllib.error
import urllib.request
from typing import Any, Protocol, runtime_checkable

from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings

from src.config import EMBEDDING_MODEL, OLLAMA_BASE_URL
from src.utils.logging import log_error, log_step


@runtime_checkable
class BaseEmbeddingProvider(Protocol):
    """Protocol for interchangeable vector embedding models."""

    @property
    def model_name(self) -> str:
        """Name of the embedding model."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Generate embedding vector for a single search query."""
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for a batch of texts."""
        ...

    def is_available(self, timeout_seconds: float = 2.0) -> bool:
        """Probe the embedding provider without blocking the process."""
        ...


def check_embedding_reachability(
    base_url: str = OLLAMA_BASE_URL, timeout_seconds: float = 2.0
) -> tuple[bool, str]:
    """
    Check if the embedding service (e.g. Ollama) is reachable within timeout.
    Returns (is_reachable, detail_message).
    """
    endpoint = f"{base_url.rstrip('/')}/api/tags"
    try:
        req = urllib.request.Request(endpoint, method="GET")
        with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
            if response.status == 200:
                return True, "Embedding service is responding"
            return False, f"Embedding service returned HTTP {response.status}"
    except (OSError, urllib.error.URLError) as exc:
        return False, f"Embedding service unreachable: {exc}"


class OllamaEmbeddingProvider:
    """Ollama embedding adapter with reachability detection and safe fallback."""

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        dimension: int = 768,
    ) -> None:
        self._model_name = model_name
        self._base_url = base_url
        self._dimension = dimension
        self._client = OllamaEmbeddings(model=self._model_name, base_url=self._base_url)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def is_available(self, timeout_seconds: float = 2.0) -> bool:
        reachable, _ = check_embedding_reachability(self._base_url, timeout_seconds)
        return reachable

    def embed_query(self, text: str) -> list[float]:
        return self._client.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._client.embed_documents(texts)

    @property
    def langchain_embeddings(self) -> Embeddings:
        """Expose native LangChain Embeddings interface for Chroma vector stores."""
        return self._client


class MockEmbeddingProvider:
    """Deterministic mock embedding provider for fast offline unit tests."""

    def __init__(self, model_name: str = "mock-embeddings", dimension: int = 128) -> None:
        self._model_name = model_name
        self._dimension = dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def is_available(self, timeout_seconds: float = 2.0) -> bool:
        return True

    def embed_query(self, text: str) -> list[float]:
        # Simple deterministic vector based on text length
        val = (len(text) % 100) / 100.0
        return [val] * self._dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]


def get_embedding_provider(
    provider_name: str = "ollama",
    model_name: str = EMBEDDING_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    dimension: int = 768,
) -> BaseEmbeddingProvider:
    """Factory function for embedding providers."""
    normalized = provider_name.lower().strip()
    if normalized == "mock":
        return MockEmbeddingProvider(model_name=model_name, dimension=dimension)
    # Default to Ollama
    return OllamaEmbeddingProvider(model_name=model_name, base_url=base_url, dimension=dimension)

