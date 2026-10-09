"""Core RAG (Retrieval Augmented Generation) pipeline built using LangChain methods."""

import os
import time
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_groq import ChatGroq

from src.config import (
    ASSISTANT_NAME,
    CHAT_MODEL,
    FINAL_TOP_K,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE_CHAT,
    MAX_CONTEXT_CHARS,
    MAX_HISTORY_MESSAGES,
)
from src.core.prompts import (
    active_guardrail_text,
    active_response_text,
    build_active_direct_prompt,
    build_active_qa_prompt,
    classify_query,
    to_langchain_messages,
)
from src.core.retrievers import (
    BM25Index,
    BM25Retriever,
    EnsembleRetriever,
    VectorRetriever,
    create_ensemble_retriever,
    load_bm25_index,
)
from src.core.tools import call_time_tool
from src.storage.cache import get_cached_answer, update_cache
from src.storage.vector_db import load_vector_store
from src.utils.logging import log_error, log_step

LOW_CONFIDENCE_CACHE_PATTERNS = (
    "sorry, i missed something",
    "could not find anything relevant",
    "not able to find",
    "unable to find",
    "do not have enough information",
    "don't have enough information",
    "not present in the provided content",
    "not available in the context",
)


def _is_low_confidence_cached_answer(answer: str) -> bool:
    """Return True for cached fallback answers that should not block retrieval."""
    normalized_answer = " ".join(answer.lower().split())
    return any(pattern in normalized_answer for pattern in LOW_CONFIDENCE_CACHE_PATTERNS)


def _retrieval_query(question: str, chat_history: list[dict[str, str]]) -> str:
    """Add the latest user turn so short follow-ups remain searchable.

    The answer history is intentionally excluded from the search query. This
    prevents a previous model response from becoming a false source of truth.
    """
    normalized = question.strip().lower()
    words = set(normalized.split())
    follow_up_markers = {"it", "that", "this", "they", "them", "there", "also"}
    if (
        len(normalized.split()) <= 8
        or bool(words & follow_up_markers)
        or normalized.startswith("what about")
    ):
        previous_user = next(
            (item.get("content", "") for item in reversed(chat_history) if item.get("role") == "user"),
            "",
        )
        if previous_user and previous_user.strip() != question.strip():
            return f"{previous_user}\nFollow-up question: {question}"
    return question


def format_context_docs(docs: list[Document]) -> str:
    """
    Format retrieved LangChain documents into bounded context string.

    Args:
        docs: List of retrieved Document objects

    Returns:
        Formatted context string bounded by MAX_CONTEXT_CHARS
    """
    context_parts: list[str] = []
    current_length = 0
    for doc in docs:
        page = doc.metadata.get("page", "unknown")
        part = f"[Page {page}]\n{doc.page_content}"
        remaining = MAX_CONTEXT_CHARS - current_length
        if remaining <= 0:
            break
        if len(part) > remaining:
            # Preserve the first part of an oversized chunk instead of
            # silently sending no context to the model.
            part = part[:remaining]
        context_parts.append(part)
        current_length += len(part)

    log_step("CONTEXT_READY", chars=current_length, docs=len(context_parts))
    return "\n\n".join(context_parts)


def build_rag_chain(llm: ChatGroq):
    """
    Construct LCEL RAG chain for question answering with context and history.

    Args:
        llm: Language model runnable

    Returns:
        LCEL chain: (input dict -> QA_PROMPT -> llm -> StrOutputParser)
    """
    return (
        RunnableParallel({
            "context": lambda x: format_context_docs(x.get("docs", [])),
            "question": lambda x: x["question"],
            "chat_history": lambda x: x.get("chat_history", []),
        })
        | build_active_qa_prompt()
        | llm
        | StrOutputParser()
    )


def build_direct_chain(llm: ChatGroq):
    """
    Construct LCEL chain for direct-context queries.

    Args:
        llm: Language model runnable

    Returns:
        LCEL chain: (input dict -> DIRECT_CONTEXT_PROMPT -> llm -> StrOutputParser)
    """
    return (
        RunnableParallel({
            "question": lambda x: x["question"],
            "chat_history": lambda x: x.get("chat_history", []),
        })
        | build_active_direct_prompt()
        | llm
        | StrOutputParser()
    )


def hybrid_search(
    vector_retriever: VectorRetriever,
    bm25_retriever: BM25Retriever,
    question: str,
) -> list[Document]:
    """
    Backward-compatible hybrid search using EnsembleRetriever.

    Args:
        vector_retriever: Vector search retriever
        bm25_retriever: BM25 search retriever
        question: User question

    Returns:
        List of top-k merged documents
    """
    ensemble = EnsembleRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        final_top_k=FINAL_TOP_K,
    )
    return ensemble.invoke(question)


def answer_question(
    vector_db: Chroma,
    bm25_index: BM25Index,
    llm: ChatGroq,
    cache: dict[str, dict[str, object]],
    chat_history: list[dict[str, str]],
    question: str,
    ensemble_retriever: EnsembleRetriever | None = None,
) -> str:
    """
    Answer a user question using the LangChain RAG pipeline.

    Multi-route execution:
    1. Check exact and semantic response cache
    2. Handle small talk directly
    3. Perform hybrid retrieval via LangChain EnsembleRetriever
    4. Generate a structured answer with the grounded RAG chain

    Args:
        vector_db: Chroma vector database
        bm25_index: BM25 index
        llm: Language model
        cache: Response cache
        chat_history: Conversation history
        question: User question
        ensemble_retriever: Optional pre-initialized EnsembleRetriever

    Returns:
        Grounded AI response
    """
    route = classify_query(question)
    log_step("ROUTE_DECISION", route=route)

    # Live tool results must not be served from the response cache.
    if route == "tool_call":
        tool_start = time.perf_counter()
        answer = call_time_tool()
        tool_latency_ms = int((time.perf_counter() - tool_start) * 1000)
        log_step("TOOL_CALL", tool="get_current_time", latency=f"{tool_latency_ms}ms")
        log_step("VECTOR_SEARCH", latency="0ms", hits=0, source="tool")
        log_step("BM25_SEARCH", latency="0ms", hits=0, source="tool")
        log_step("SEARCH_DONE", latency="0ms", hits=0, source="tool")
        log_step("LLM_RESPONSE", latency="0ms", source="tool")
        return answer

    # 1. Check cache first
    cache_enabled = True
    try:
        from src.admin.config_service import load_config

        cache_enabled = load_config().performance.cache_enabled
    except (OSError, ValueError):
        pass
    cached_answer = get_cached_answer(cache, question) if cache_enabled else None
    if cached_answer and not (
        route == "retrieval" and _is_low_confidence_cached_answer(cached_answer)
    ):
        log_step("CACHE_HIT", route=route)
        log_step("VECTOR_SEARCH", latency="0ms", hits=0, source="cache")
        log_step("BM25_SEARCH", latency="0ms", hits=0, source="cache")
        log_step("SEARCH_DONE", latency="0ms", hits=0, source="cache")
        log_step("LLM_RESPONSE", latency="0ms", source="cache")
        return cached_answer

    if cached_answer:
        log_step("CACHE_BYPASS", route=route, reason="low_confidence_answer")
    else:
        log_step("CACHE_MISS", route=route)

    # 2. Handle small talk
    if route == "small_talk":
        answer = active_response_text(
            "greeting_prompt",
            f"Hello! How can I assist you today? I am {ASSISTANT_NAME}.",
        )
        if cache_enabled:
            update_cache(cache, question, answer, route)
        log_step("VECTOR_SEARCH", latency="0ms", hits=0, source="route")
        log_step("BM25_SEARCH", latency="0ms", hits=0, source="route")
        log_step("SEARCH_DONE", latency="0ms", hits=0, source="route")
        log_step("LLM_RESPONSE", latency="0ms", source="route")
        return answer

    # Convert conversation history to LangChain BaseMessage objects
    lc_messages = to_langchain_messages(chat_history)

    # 3. Every knowledge question, including long pasted requests, must pass
    # through retrieval. This prevents a prompt-shaped user message from
    # bypassing the active PDF and causing an ungrounded answer.
    retriever = ensemble_retriever or create_ensemble_retriever(vector_db, bm25_index)
    docs = retriever.invoke(_retrieval_query(question, chat_history))

    if not docs:
        log_step("SEARCH_EMPTY", route=route)
        return (
            active_guardrail_text(
                active_response_text(
                    "no_answer_prompt",
                    "I don't have information about that in the knowledge base. "
                    "Please provide more details or rephrase your question.",
                )
            )
        )

    # 4. Generate answer using LCEL RAG chain
    rag_chain = build_rag_chain(llm)
    llm_start = time.perf_counter()
    raw_answer = rag_chain.invoke({
        "docs": docs,
        "question": question,
        "chat_history": lc_messages,
    })
    answer = raw_answer.strip() if isinstance(raw_answer, str) else str(raw_answer).strip()
    llm_latency_ms = int((time.perf_counter() - llm_start) * 1000)
    log_step("LLM_RESPONSE", latency=f"{llm_latency_ms}ms")

    # Guard: Never return an empty string to the user
    if not answer:
        log_step("LLM_EMPTY_RESPONSE", route=route)
        answer = (
            "I'm sorry, I was unable to generate a response. "
            "Please try rephrasing your question or provide more details."
        )

    if _is_low_confidence_cached_answer(answer):
        log_step("CACHE_SKIP", route=route, reason="low_confidence_answer")
    else:
        if cache_enabled:
            update_cache(cache, question, answer, route)

    return answer


def chat() -> None:
    """
    Run the CLI chat interface.

    Loads the knowledge base and allows interactive conversation.
    """
    from dotenv import load_dotenv

    load_dotenv()

    vector_db = load_vector_store()
    bm25_index = load_bm25_index(vector_db)
    ensemble = create_ensemble_retriever(vector_db, bm25_index)

    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name=CHAT_MODEL,
        temperature=LLM_TEMPERATURE_CHAT,
        max_tokens=LLM_MAX_TOKENS,
    )

    from src.storage.cache import load_cache

    cache = load_cache()
    chat_history: list[dict[str, str]] = []

    print("VAPS chatbot is ready. Type 'exit' to quit.")

    while True:
        question = input("\nYou: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Bot: Goodbye.")
            break

        try:
            total_start = time.perf_counter()
            log_step("QUERY_START", query_len=len(question))

            answer = answer_question(
                vector_db=vector_db,
                bm25_index=bm25_index,
                llm=llm,
                cache=cache,
                chat_history=chat_history,
                question=question,
                ensemble_retriever=ensemble,
            )
            chat_history.append({"role": "user", "content": question})
            chat_history.append({"role": "assistant", "content": answer})

            if len(chat_history) > MAX_HISTORY_MESSAGES:
                chat_history = chat_history[-MAX_HISTORY_MESSAGES:]
            print(f"Bot: {answer}")

            log_step("BOT_MESSAGE", answer_len=len(answer))

            total_latency_ms = int((time.perf_counter() - total_start) * 1000)
            log_step("TOTAL", latency=f"{total_latency_ms}ms")
        except Exception as exc:
            log_error("CHAT_LOOP_ERROR", exc, query_len=len(question))
            print(f"Bot: Request failed: {exc}")
