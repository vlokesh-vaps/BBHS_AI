"""Document retrieval using BM25 and vector search with LangChain BaseRetriever."""

import math
import pickle
import time
from collections import Counter
from typing import Any

from langchain_chroma import Chroma
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from src.config import (
    BM25_CACHE_FILE,
    BM25_TOP_K,
    FINAL_TOP_K,
    TOKEN_PATTERN,
    VECTOR_TOP_K,
)

# A zero admin threshold means "use the configured behavior".  Real Chroma
# relevance scores still get this small safety floor so a completely unrelated
# nearest-neighbour chunk cannot become evidence for an answer.
MIN_VECTOR_RELEVANCE = 0.12
from src.storage.vector_db import load_all_documents
from src.utils.logging import log_step


class BM25Index:
    """BM25 (Best Matching 25) keyword search index."""

    def __init__(self, documents: list[Document]) -> None:
        """
        Initialize BM25 index from documents.

        Args:
            documents: List of documents to index
        """
        self.documents = documents
        self.doc_tokens = [self._tokenize(doc.page_content) for doc in documents]
        self.doc_term_freqs = [Counter(tokens) for tokens in self.doc_tokens]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_length = (
            sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        )
        self.doc_freqs = self._build_doc_freqs()

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize text into lowercase words."""
        return TOKEN_PATTERN.findall(text.lower())

    def _build_doc_freqs(self) -> Counter:
        """Build document frequency table for all tokens."""
        doc_freqs: Counter = Counter()
        for tokens in self.doc_tokens:
            for token in set(tokens):
                doc_freqs[token] += 1
        return doc_freqs

    def search(self, query: str, top_k: int) -> list[tuple[float, Document]]:
        """
        Search for documents matching the query.

        Args:
            query: Search query
            top_k: Number of top results to return

        Returns:
            List of (score, document) tuples sorted by score descending
        """
        query_tokens = self._tokenize(query)
        if not query_tokens or not self.documents:
            return []

        total_docs = len(self.documents)
        scores: list[tuple[float, Document]] = []
        k1 = 1.5
        b = 0.75

        for index, doc in enumerate(self.documents):
            score = 0.0
            term_freqs = self.doc_term_freqs[index]
            doc_length = self.doc_lengths[index] or 1

            for token in query_tokens:
                doc_freq = self.doc_freqs.get(token, 0)
                if doc_freq == 0:
                    continue

                idf = math.log(1 + ((total_docs - doc_freq + 0.5) / (doc_freq + 0.5)))
                tf = term_freqs.get(token, 0)
                if tf == 0:
                    continue

                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (
                    1 - b + b * (doc_length / (self.avg_doc_length or 1))
                )
                score += idf * (numerator / denominator)

            if score > 0:
                scores.append((score, doc))

        scores.sort(key=lambda item: item[0], reverse=True)
        return scores[:top_k]


class VectorRetriever(BaseRetriever):
    """Semantic search using vector similarity as a LangChain BaseRetriever."""

    vector_db: Any = Field(description="Chroma vector database instance")
    top_k: int = Field(default=VECTOR_TOP_K, description="Number of results to retrieve")

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> list[Document]:
        """Retrieve documents matching query semantically."""
        # Keep the relevance score on a private metadata key when Chroma can
        # provide it.  The score is used by the ensemble as a grounding gate;
        # ordinary similarity_search is retained as a compatibility fallback
        # for older Chroma versions and test doubles.
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


class BM25Retriever(BaseRetriever):
    """Keyword-based retrieval using BM25 index as a LangChain BaseRetriever."""

    bm25_index: Any = Field(description="BM25Index instance")
    top_k: int = Field(default=BM25_TOP_K, description="Number of results to retrieve")

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> list[Document]:
        """Retrieve documents matching query keywords."""
        results = self.bm25_index.search(query, top_k=self.top_k)
        return [doc for _, doc in results]

    def get_relevant_documents(self, query: str) -> list[Document]:
        """Backward-compatible retrieval method."""
        return self._get_relevant_documents(query)


def _doc_key(doc: Document) -> tuple[object, str]:
    """Generate a unique key for a document based on page and content snippet."""
    return doc.metadata.get("page"), doc.page_content[:160]


class EnsembleRetriever(BaseRetriever):
    """
    Hybrid retriever combining Vector and BM25 retrievers via Reciprocal Rank Fusion.
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
    vector_db: Chroma,
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


def _bm25_cache_is_current() -> bool:
    """Check if cached BM25 index is up-to-date with vector database."""
    from src.config import PERSIST_DIRECTORY

    if not BM25_CACHE_FILE.exists():
        return False

    vector_db_file = PERSIST_DIRECTORY / "chroma.sqlite3"
    if not vector_db_file.exists():
        return False

    return BM25_CACHE_FILE.stat().st_mtime >= vector_db_file.stat().st_mtime


def load_bm25_index(vector_db: Chroma) -> BM25Index:
    """
    Load or build BM25 index.

    Attempts to load from cache if it's current, otherwise rebuilds from vector DB.

    Args:
        vector_db: Chroma vector database instance

    Returns:
        BM25Index instance
    """
    if _bm25_cache_is_current():
        try:
            with BM25_CACHE_FILE.open("rb") as file:
                cached_index = pickle.load(file)
            if isinstance(cached_index, BM25Index):
                return cached_index
        except (OSError, pickle.PickleError, EOFError):
            pass

    documents = load_all_documents(vector_db)
    bm25_index = BM25Index(documents)
    BM25_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with BM25_CACHE_FILE.open("wb") as file:
        pickle.dump(bm25_index, file)
    return bm25_index
