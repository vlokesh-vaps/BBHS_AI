"""Exact and semantic response cache (in-memory only per D3)."""

import math
import time
from hashlib import sha256

from langchain_ollama import OllamaEmbeddings

from src.config import (
    CACHE_VERSION,
    EMBEDDING_MODEL,
    MAX_CACHE_ENTRIES,
    OLLAMA_BASE_URL,
    SEMANTIC_CACHE_THRESHOLD,
    normalize_question,
)
from src.utils.logging import log_error, log_step


_semantic_embedder: OllamaEmbeddings | None = None


def get_semantic_embedder() -> OllamaEmbeddings:
    """Get or create the semantic embedder singleton."""
    global _semantic_embedder
    if _semantic_embedder is None:
        _semantic_embedder = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    return _semantic_embedder


def make_cache_key(question: str) -> str:
    """
    Generate a cache key for a question.

    If the question is short enough, use it directly.
    Otherwise, use SHA256 hash of the normalized question.
    """
    normalized_question = normalize_question(question)
    if len(normalized_question) <= 160:
        return normalized_question
    digest = sha256(normalized_question.encode("utf-8")).hexdigest()[:16]
    return f"sha256:{digest}"


def load_cache() -> dict[str, dict[str, object]]:
    """
    Load the cache in-memory.

    Per D3 (no chat storage), disk persistence is disabled.
    Always returns a new in-memory dictionary.
    """
    return {}


def save_cache(cache: dict[str, dict[str, object]]) -> None:
    """
    Enforce in-memory cache capacity without writing to disk.

    Removes oldest entries if cache exceeds MAX_CACHE_ENTRIES.
    """
    if len(cache) > MAX_CACHE_ENTRIES:
        ordered_items = sorted(
            cache.items(),
            key=lambda item: int(item[1].get("updated_at", 0)),
            reverse=True,
        )
        cache.clear()
        cache.update(dict(ordered_items[:MAX_CACHE_ENTRIES]))


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def get_cached_answer(
    cache: dict[str, dict[str, object]], question: str
) -> str | None:
    """
    Retrieve a cached answer using exact or semantic matching.

    First tries exact key lookup, then semantic similarity lookup.

    Args:
        cache: Cache dictionary
        question: User question

    Returns:
        Cached answer if found, None otherwise
    """
    # Try exact match first
    cache_key = make_cache_key(question)
    cached_entry = cache.get(cache_key)
    if isinstance(cached_entry, dict):
        answer = cached_entry.get("answer")
        if isinstance(answer, str):
            return answer

    # Try semantic match
    normalized_question = normalize_question(question)
    try:
        query_embedding = get_semantic_embedder().embed_query(normalized_question)
    except Exception as exc:
        log_error("SEMANTIC_CACHE_EMBED_FAILED", exc, query_len=len(normalized_question))
        return None

    best_score = -1.0
    best_answer: str | None = None

    for entry in cache.values():
        if not isinstance(entry, dict):
            continue
        answer = entry.get("answer")
        embedding = entry.get("question_embedding")
        if not isinstance(answer, str) or not isinstance(embedding, list):
            continue
        if not all(isinstance(value, (int, float)) for value in embedding):
            continue

        score = _cosine_similarity(query_embedding, [float(value) for value in embedding])
        if score > best_score:
            best_score = score
            best_answer = answer

    threshold = SEMANTIC_CACHE_THRESHOLD
    try:
        from src.admin.config_service import load_config

        threshold = load_config().performance.semantic_cache_threshold
    except (OSError, ValueError):
        pass

    if best_answer and best_score >= threshold:
        log_step("SEMANTIC_CACHE_HIT", similarity=round(best_score, 4))
        return best_answer

    log_step("SEMANTIC_CACHE_MISS", similarity=round(max(best_score, 0.0), 4))
    return None


def update_cache(
    cache: dict[str, dict[str, object]], question: str, answer: str, route: str
) -> None:
    """
    Update the cache with a new question-answer pair.

    Args:
        cache: Cache dictionary
        question: User question
        answer: AI response
        route: Query route classification
    """
    normalized_question = normalize_question(question)
    question_embedding: list[float] | None = None
    try:
        question_embedding = get_semantic_embedder().embed_query(normalized_question)
    except Exception as exc:
        log_error("SEMANTIC_CACHE_WRITE_EMBED_FAILED", exc, query_len=len(normalized_question))

    cache[make_cache_key(question)] = {
        "answer": answer,
        "updated_at": int(time.time()),
        "route": route,
        "version": CACHE_VERSION,
        "normalized_question": normalized_question,
        "question_embedding": question_embedding,
    }
    save_cache(cache)
