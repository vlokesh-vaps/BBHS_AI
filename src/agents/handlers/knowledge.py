"""Knowledge base handler delegating to hybrid RAG retrieval."""

from typing import Any
from langchain_chroma import Chroma

from src.agents.state import ActionItem, AgentResponse, SessionState
from src.config import ADMISSION_PHONE
from src.core.rag import answer_question
from src.core.retrievers import BM25Index, EnsembleRetriever


def handle_knowledge_query(
    question: str,
    session_state: SessionState,
    vector_db: Chroma,
    bm25_index: BM25Index,
    llm: Any,
    cache: dict[str, dict[str, object]],
    chat_history: list[dict[str, str]],
    ensemble_retriever: EnsembleRetriever | None = None,
    answer_question_fn: Any = None,
) -> AgentResponse:
    """Answer factual school questions using the verified RAG pipeline."""
    fn = answer_question_fn or answer_question
    rag_answer = fn(
        vector_db=vector_db,
        bm25_index=bm25_index,
        llm=llm,
        cache=cache,
        chat_history=chat_history,
        question=question,
        ensemble_retriever=ensemble_retriever,
    )

    grade = session_state.grade
    actions = [
        ActionItem(label="Admissions Process", action="quick_reply", value="Tell me about the admission process", icon="message"),
        ActionItem(
            label=f"Fee Enquiry ({grade})" if grade else "Fee Structure",
            action="quick_reply",
            value=f"What is the fee for {grade}?" if grade else "What is the fee structure?",
            icon="message",
        ),
        ActionItem(label="Campus Visit", action="quick_reply", value="How can I visit the school?", icon="calendar"),
        ActionItem(label=f"Call Admissions ({ADMISSION_PHONE})", action="tel", value=ADMISSION_PHONE, icon="phone"),
    ]

    return AgentResponse(answer=rag_answer, actions=actions, state=session_state)
