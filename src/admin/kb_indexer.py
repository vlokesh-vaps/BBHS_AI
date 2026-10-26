"""Incremental Knowledge Base indexer for multi-source school documents."""

import pickle
import threading
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.admin.config_service import load_config
from src.admin.kb_repository import get_active_published_documents, update_chunks_count
from src.admin.parsers import chunk_and_enrich_document
from src.admin.schemas import DocumentRecord
from src.config import BM25_CACHE_FILE, KB_DIR
from src.core.retrievers import BM25Index, create_ensemble_retriever, load_bm25_index
from src.storage.vector_db import load_all_documents, load_vector_store
from src.utils.logging import log_error, log_step

_indexer_lock = threading.Lock()


def _ensure_vector_db(app_instance: Any = None) -> Chroma:
    """Ensure vector store is loaded, updating app_instance.state if provided."""
    vector_db = None
    if app_instance is not None and hasattr(app_instance, "state"):
        vector_db = getattr(app_instance.state, "vector_db", None)

    if vector_db is None:
        vector_db = load_vector_store()
        if app_instance is not None and hasattr(app_instance, "state"):
            app_instance.state.vector_db = vector_db

    return vector_db


def _refresh_retrievers(vector_db: Chroma, app_instance: Any = None) -> None:
    """Rebuild in-memory BM25 index and update EnsembleRetriever on app_instance."""
    try:
        docs = load_all_documents(vector_db)
        bm25_index = BM25Index(docs)
    except Exception:
        # If vector store is empty, create an empty BM25 index with a fallback doc
        bm25_index = BM25Index([Document(page_content="Knowledge base is empty.", metadata={})])

    BM25_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_CACHE_FILE, "wb") as f:
        pickle.dump(bm25_index, f)

    runtime_cfg = load_config()
    ensemble = create_ensemble_retriever(
        vector_db,
        bm25_index,
        vector_top_k=runtime_cfg.rag.vector_top_k,
        bm25_top_k=runtime_cfg.rag.bm25_top_k,
        final_top_k=runtime_cfg.rag.final_top_k,
        similarity_threshold=max(
            runtime_cfg.rag.similarity_threshold,
            runtime_cfg.guardrails.minimum_retrieval_score,
        ),
    )

    if app_instance is not None and hasattr(app_instance, "state"):
        app_instance.state.bm25_index = bm25_index
        app_instance.state.ensemble_retriever = ensemble
        if hasattr(app_instance.state, "cache") and app_instance.state.cache is not None:
            app_instance.state.cache.clear()

    log_step("INDEX_RETRIEVERS_REFRESHED")


def sync_document_index(record: DocumentRecord, app_instance: Any = None) -> int:
    """
    Synchronize a single document's index state:
    - If status == 'Published': chunks, deletes old vectors, and adds new vectors.
    - If status != 'Published': removes vectors so outdated content is not retrieved.

    Returns the number of active chunks indexed for this document.
    """
    with _indexer_lock:
        vector_db = _ensure_vector_db(app_instance)

        # 1. Clean out existing chunks for this doc_id
        try:
            vector_db._collection.delete(where={"doc_id": record.id})
            log_step("DOC_OLD_CHUNKS_DELETED", doc_id=record.id)
        except Exception as exc:
            # If collection has no matching items or delete where is unsupported, log and continue
            log_error("DOC_DEINDEX_NOTICE", exc, doc_id=record.id)

        # 2. If not published, complete de-indexing
        if record.status != "Published":
            update_chunks_count(record.id, 0)
            _refresh_retrievers(vector_db, app_instance)
            log_step("DOC_DEINDEXED", doc_id=record.id, status=record.status)
            return 0

        # 3. If published, parse, chunk, and embed
        rag_config = load_config().rag
        chunks = chunk_and_enrich_document(
            record,
            KB_DIR,
            chunk_size=rag_config.chunk_size,
            chunk_overlap=rag_config.chunk_overlap,
        )

        if not chunks:
            update_chunks_count(record.id, 0)
            _refresh_retrievers(vector_db, app_instance)
            return 0

        # Assign deterministic unique chunk IDs: doc_id#chunk_idx
        chunk_ids = [f"{record.id}#{idx}" for idx in range(len(chunks))]
        vector_db.add_documents(chunks, ids=chunk_ids)

        update_chunks_count(record.id, len(chunks))
        _refresh_retrievers(vector_db, app_instance)

        log_step("DOC_INDEXED_SUCCESS", doc_id=record.id, chunks=len(chunks))
        return len(chunks)


def deindex_document_by_id(doc_id: str, app_instance: Any = None) -> None:
    """Explicitly remove all vectors for a document from the index."""
    with _indexer_lock:
        vector_db = _ensure_vector_db(app_instance)
        try:
            vector_db._collection.delete(where={"doc_id": doc_id})
        except Exception as exc:
            log_error("DOC_DEINDEX_ERROR", exc, doc_id=doc_id)

        update_chunks_count(doc_id, 0)
        _refresh_retrievers(vector_db, app_instance)
        log_step("DOC_DEINDEXED", doc_id=doc_id)


def reindex_all_active_documents(app_instance: Any = None) -> int:
    """Full synchronization of all eligible published documents."""
    active_docs = get_active_published_documents()
    total_chunks = 0
    for doc in active_docs:
        try:
            count = sync_document_index(doc, app_instance)
            total_chunks += count
        except Exception as exc:
            log_error("DOC_REINDEX_ERROR", exc, doc_id=doc.id)
    return total_chunks
