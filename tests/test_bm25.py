"""Dedicated unit tests for packaged BM25 retrieval engine."""

from langchain_core.documents import Document

from src.retrieval.bm25 import BM25Index, BM25Retriever


def test_bm25_tokenization_and_frequency():
    docs = [
        Document(page_content="Baldwin Boys High School admissions are open for Grade 1 to Grade 10."),
        Document(page_content="School transport routes cover Indiranagar, Koramangala, and Whitefield."),
    ]
    index = BM25Index(docs)
    assert len(index.doc_tokens) == 2
    assert "baldwin" in index.doc_tokens[0]
    assert "indiranagar" in index.doc_tokens[1]
    assert index.doc_freqs["school"] == 2
    assert index.doc_freqs["transport"] == 1


def test_bm25_search_scoring_order():
    docs = [
        Document(page_content="The annual sports day athletic events are scheduled for December."),
        Document(page_content="Tuition fee payment schedule and quarterly deadlines for students."),
        Document(page_content="Fee concession and scholarship guidelines for academic excellence."),
    ]
    index = BM25Index(docs)
    results = index.search("fee payment", top_k=2)

    assert len(results) == 2
    # The document with both 'fee' and 'payment' should rank highest
    top_score, top_doc = results[0]
    assert top_score > 0.0
    assert "Tuition fee payment" in top_doc.page_content


def test_bm25_empty_query_and_empty_docs():
    index = BM25Index([])
    assert index.search("hello", top_k=5) == []

    index_with_docs = BM25Index([Document(page_content="Sample text")])
    assert index_with_docs.search("", top_k=5) == []
    assert index_with_docs.search("   ", top_k=5) == []


def test_bm25_langchain_retriever_adapter():
    docs = [
        Document(page_content="Visiting hours are from 9:00 AM to 12:00 PM on weekdays."),
    ]
    index = BM25Index(docs)
    retriever = BM25Retriever(bm25_index=index, top_k=1)
    retrieved = retriever.invoke("visiting hours")
    assert len(retrieved) == 1
    assert "Visiting hours" in retrieved[0].page_content

