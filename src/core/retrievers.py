"""Document retrieval using BM25 and vector search with LangChain BaseRetriever.

Maintained for backward compatibility. Implementation resides in src/retrieval.
"""

from src.retrieval.bm25 import (
    BM25Index,
    BM25Retriever,
    is_bm25_cache_current as _bm25_cache_is_current,
    load_bm25_index,
)
from src.retrieval.hybrid import (
    MIN_VECTOR_RELEVANCE,
    EnsembleRetriever,
    VectorRetriever,
    _doc_key,
    _is_doc_active,
    create_ensemble_retriever,
)

__all__ = [
    "BM25Index",
    "BM25Retriever",
    "VectorRetriever",
    "EnsembleRetriever",
    "create_ensemble_retriever",
    "load_bm25_index",
    "_bm25_cache_is_current",
    "_doc_key",
    "_is_doc_active",
    "MIN_VECTOR_RELEVANCE",
]
