"""Validated, atomic persistence for Control Center configuration."""

import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.admin.schemas import AdminConfig, AdminConfigUpdate
from src.config import (    ADMIN_CONFIG_FILE,    ASSISTANT_NAME,    BM25_TOP_K,
    CHAT_MODEL,    FINAL_TOP_K,    LLM_MAX_TOKENS,    LLM_TEMPERATURE,    MAX_HISTORY_CHARS,
    MAX_HISTORY_MESSAGES,    PDF_CHUNK_OVERLAP,    PDF_CHUNK_SIZE,    SEMANTIC_CACHE_THRESHOLD,
    VECTOR_TOP_K,
)


def _default_prompt_values() -> dict[str, str]:
    """Read the existing prompt source as the default admin configuration."""
    from src.core.prompts import (        SYSTEM_DIRECT_INSTRUCTIONS,        SYSTEM_RAG_INSTRUCTIONS,    )

    return {
        "system_prompt": SYSTEM_RAG_INSTRUCTIONS,
        "rag_prompt": "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:",
        "no_answer_prompt": "I'd love to help with that. Could you tell me a little more? For example, are you asking about admissions, fees, school timings, facilities, or school life?",
        "greeting_prompt": f"Hello! How can I assist you today? I am {ASSISTANT_NAME}.",
        "context_instructions": SYSTEM_DIRECT_INSTRUCTIONS,
    }


def default_config() -> AdminConfig:
    """Return the validated configuration derived from existing constants."""
    prompts = _default_prompt_values()
    return AdminConfig(
        models={"primary_provider": "groq",
            "primary_model": CHAT_MODEL,
            "fallback_provider": "",
            "fallback_model": "",
            "temperature": LLM_TEMPERATURE,
            "max_tokens": LLM_MAX_TOKENS,
        },
        prompts=prompts,
        rag={
            "vector_top_k": VECTOR_TOP_K,
            "bm25_top_k": BM25_TOP_K,
            "final_top_k": FINAL_TOP_K,
            "chunk_size": PDF_CHUNK_SIZE,
            "chunk_overlap": PDF_CHUNK_OVERLAP,
        },
        performance={
            "semantic_cache_threshold": SEMANTIC_CACHE_THRESHOLD,
            "max_history_messages": MAX_HISTORY_MESSAGES,
            "max_history_chars": MAX_HISTORY_CHARS,
        },
        guardrails={},
    )


def write_json_atomic(data: Any, path: Path) -> None:
    """Persist arbitrary JSON data through a sibling temporary file and atomic replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    if hasattr(data, "model_dump"):
        serialized = json.dumps(data.model_dump(mode="json"), indent=2)
    elif isinstance(data, (dict, list)):
        serialized = json.dumps(data, indent=2)
    else:
        serialized = str(data)
    temporary_path.write_text(serialized, encoding="utf-8")
    os.replace(temporary_path, path)


def _write_atomic(config: AdminConfig, path: Path = ADMIN_CONFIG_FILE) -> AdminConfig:
    """Persist JSON through a sibling temporary file and atomic replace."""
    write_json_atomic(config, path)
    return config


def load_config(path: Path = ADMIN_CONFIG_FILE) -> AdminConfig:
    """Load and validate persisted configuration, creating defaults if absent."""
    if not path.exists():
        return _write_atomic(default_config(), path)
    try:
        return AdminConfig.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, ValidationError) as exc:
        raise ValueError(f"Admin configuration is missing or invalid: {exc}") from exc


def validate_config(config: AdminConfig) -> AdminConfig:
    """Re-validate a complete configuration payload."""
    return AdminConfig.model_validate(config.model_dump())


def update_config(update: AdminConfigUpdate, path: Path = ADMIN_CONFIG_FILE) -> AdminConfig:
    """Merge, validate, version, and atomically persist a partial update."""
    current = load_config(path)
    values = current.model_dump()
    for section, section_value in update.model_dump(exclude_none=True).items():
        values[section] = section_value
    values["version"] = current.version + 1
    values["updated_at"] = datetime.now(timezone.utc)
    updated = AdminConfig.model_validate(values)
    return _write_atomic(updated, path)
