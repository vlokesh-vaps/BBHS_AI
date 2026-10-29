"""Vector database loading, metadata validation, and management."""

import json
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from src.config import COLLECTION_NAME, EMBEDDING_MODEL, OLLAMA_BASE_URL, PERSIST_DIRECTORY
from src.providers.embeddings import check_embedding_reachability
from src.utils.logging import log_error, log_step

INDEX_META_FILE = PERSIST_DIRECTORY / "index_meta.json"


def validate_index_metadata(persist_dir: Path = PERSIST_DIRECTORY, expected_model: str = EMBEDDING_MODEL) -> None:
    """Validate that the existing vector database matches the configured embedding model."""
    meta_path = persist_dir / "index_meta.json"
    if not meta_path.exists():
        # Save metadata for existing store if absent
        try:
            meta = {
                "embedding_model": expected_model,
                "embedding_provider": "ollama",
                "collection_name": COLLECTION_NAME,
            }
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        except Exception:
            pass
        return

    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        indexed_model = data.get("embedding_model")
        if indexed_model and indexed_model != expected_model:
            raise ValueError(
                f"Embedding model mismatch: index was built with '{indexed_model}', "
                f"but configuration specifies '{expected_model}'. A rebuild is required."
            )
    except (json.JSONDecodeError, OSError) as exc:
        log_error("INDEX_META_READ_ERROR", exc)


def load_vector_store(embedding_model: Any = None) -> Chroma:
    """
    Load the Chroma vector database with resilient startup and metadata validation.

    Raises:
        FileNotFoundError: If the vector database doesn't exist
        ValueError: If embedding model is incompatible with index

    Returns:
        Initialized Chroma vector store
    """
    if not PERSIST_DIRECTORY.exists():
        raise FileNotFoundError(
            f"Vector DB not found at {PERSIST_DIRECTORY}. Run scripts/build_kb.py first."
        )

    validate_index_metadata(PERSIST_DIRECTORY, EMBEDDING_MODEL)

    if embedding_model is None:
        # Check reachability before instantiating or warming up
        reachable, detail = check_embedding_reachability(OLLAMA_BASE_URL, timeout_seconds=1.5)
        embedding_model = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
        if reachable:
            try:
                embedding_model.embed_query("warmup")
            except Exception as exc:
                log_error("EMBEDDING_WARMUP_NOTICE", exc)
        else:
            log_step("EMBEDDING_SERVICE_OFFLINE", detail=detail)

    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(PERSIST_DIRECTORY),
        embedding_function=embedding_model,
    )


def load_all_documents(vector_db: Chroma) -> list[Document]:
    """
    Retrieve all documents from the vector database.

    Args:
        vector_db: Chroma vector store instance

    Returns:
        List of all stored documents

    Raises:
        ValueError: If the vector database is empty
    """
    data = vector_db.get()
    documents = data.get("documents", [])
    metadatas = data.get("metadatas", [])

    if not documents:
        raise ValueError("Vector DB is empty. Run scripts/build_kb.py again to rebuild the knowledge base.")

    loaded_documents = []
    for page_content, metadata in zip(documents, metadatas):
        loaded_documents.append(Document(page_content=page_content, metadata=metadata or {}))
    return loaded_documents
