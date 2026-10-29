"""Hybrid search retriever combining Dense Vector search and Sparse BM25 via Reciprocal Rank Fusion."""

import time
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from src.config import BM25_TOP_K, FINAL_TOP_K, VECTOR_TOP_K
from src.retrieval.bm25 import BM25Index, BM25Retriever
from src.utils.logging import log_step

# A zero admin threshold means "use the configured behavior". Real Chroma
# relevance scores still get this small safety floor so a completely unrelated
# nearest-neighbour chunk cannot become evidence for an answer.
MIN_VECTOR_RELEVANCE = 0.12


class VectorRetriever(BaseRetriever):
    """Semantic search using vector similarity as a LangChain BaseRetriever."""

    vector_db: Any = Field(description="Chroma vector database instance")
    top_k: int = Field(default=VECTOR_TOP_K, description="Number of results to retrieve")

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> list[Document]:
        """Retrieve documents matching query semantically."""
        try:
            scored_docs = self.vector_db.similarity_search_with_relevance_scores(
                query, k=self.top_k
            )
            results: list[Document] = []
            for doc, score in scored_docs:
                metadata = dict(doc.metadata or {})
                metadata["_vector_relevance"] = max(0.0, min(1.0, float(score)))
                results.append(Document(page_content=doc.page_content, metadata=metadata))
            return results
        except (AttributeError, TypeError, ValueError):
            return self.vector_db.similarity_search(query, k=self.top_k)

    def get_relevant_documents(self, query: str) -> list[Document]:
        """Backward-compatible retrieval method."""
        return self._get_relevant_documents(query)


def _doc_key(doc: Document) -> tuple[object, str]:
    """Generate a unique key for a document based on page and content snippet."""
    return doc.metadata.get("page"), doc.page_content[:160]


def _is_doc_active(doc: Document) -> bool:
    """Filter out documents that are unpublished or expired based on chunk metadata."""
    from datetime import datetime, timezone
    meta = doc.metadata or {}
    status = meta.get("status")
    if status and status != "Published":
        return False
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    eff = meta.get("effective_date")
    if eff and eff > today_str:
        return False
    exp = meta.get("expiry_date")
    if exp and exp < today_str:
        return False
    return True


class EnsembleRetriever(BaseRetriever):
    """
    Hybrid retriever combining Vector and BM25 retrievers via Reciprocal Rank Fusion (RRF).
    """

    vector_retriever: VectorRetriever = Field(description="Vector search retriever")
    bm25_retriever: BM25Retriever = Field(description="BM25 search retriever")
    final_top_k: int = Field(default=FINAL_TOP_K, description="Max documents to return")
    similarity_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum vector relevance score when supplied by Chroma",
    )

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> list[Document]:
        """Execute hybrid search using Reciprocal Rank Fusion."""
        v_start = time.perf_counter()
        vector_docs = self.vector_retriever.invoke(query)
        vector_docs = [doc for doc in vector_docs if _is_doc_active(doc)]
        if any("_vector_relevance" in doc.metadata for doc in vector_docs):
            threshold = max(self.similarity_threshold, MIN_VECTOR_RELEVANCE)
            vector_docs = [
                doc for doc in vector_docs
                if float(doc.metadata.get("_vector_relevance", 0.0))
                >= threshold
            ]
        v_latency = int((time.perf_counter() - v_start) * 1000)
        log_step("VECTOR_SEARCH", latency=f"{v_latency}ms", hits=len(vector_docs))

        b_start = time.perf_counter()
        bm25_docs = self.bm25_retriever.invoke(query)
        bm25_docs = [doc for doc in bm25_docs if _is_doc_active(doc)]
        b_latency = int((time.perf_counter() - b_start) * 1000)
        log_step("BM25_SEARCH", latency=f"{b_latency}ms", hits=len(bm25_docs))

        # Reciprocal Rank Fusion (RRF)
        fused_scores: dict[tuple[object, str], float] = {}
        docs_by_key: dict[tuple[object, str], Document] = {}

        for rank, doc in enumerate(vector_docs, start=1):
            key = _doc_key(doc)
            fused_scores[key] = fused_scores.get(key, 0.0) + (1.0 / (60.0 + rank))
            docs_by_key[key] = doc

        for rank, doc in enumerate(bm25_docs, start=1):
            key = _doc_key(doc)
            fused_scores[key] = fused_scores.get(key, 0.0) + (1.0 / (60.0 + rank))
            docs_by_key[key] = doc

        ranked_keys = sorted(fused_scores, key=fused_scores.get, reverse=True)
        merged_docs = [docs_by_key[key] for key in ranked_keys[: self.final_top_k]]

        total_latency = v_latency + b_latency
        log_step("SEARCH_DONE", latency=f"{total_latency}ms", hits=len(merged_docs))
        return merged_docs


def create_ensemble_retriever(
    vector_db: Any,
    bm25_index: BM25Index,
    vector_top_k: int = VECTOR_TOP_K,
    bm25_top_k: int = BM25_TOP_K,
    final_top_k: int = FINAL_TOP_K,
    similarity_threshold: float = 0.0,
) -> EnsembleRetriever:
    """Factory function to build a unified EnsembleRetriever."""
    v_retriever = VectorRetriever(vector_db=vector_db, top_k=vector_top_k)
    b_retriever = BM25Retriever(bm25_index=bm25_index, top_k=bm25_top_k)
    return EnsembleRetriever(
        vector_retriever=v_retriever,
        bm25_retriever=b_retriever,
        final_top_k=final_top_k,
        similarity_threshold=similarity_threshold,
    )
