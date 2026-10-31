"""LangGraph StateGraph for BBHS Conversational Admissions Agent.

This module replaces imperative orchestrator branching with a clean,
declarative, and inspectable state graph using LangGraph.
All conversation state remains strictly in-memory per Invariant D3.
"""

import re
from typing import Any, TypedDict
from langchain_chroma import Chroma
from langgraph.graph import END, START, StateGraph

from src.agents.entity_extractor import (    extract_entities,    is_consent_affirmative,    is_consent_negative,)

from src.agents.handlers import (    handle_admissions_process,    handle_contact_request,    handle_fee_enquiry,
    handle_knowledge_query,    handle_lead_callback,    handle_out_of_scope,    handle_school_visit,
    handle_small_talk,    handle_time_query,)

from src.agents.state import (    ActionItem,    AgentIntent,    EnquiryStage,    SessionState,)
from src.config import (    GREETING_PATTERN,    TIME_QUERY_PATTERN,    normalize_question,)

from src.core.retrievers import BM25Index, EnsembleRetriever
from src.leads.service import extract_phone_from_text
from src.utils.logging import log_step

FEE_PATTERN = re.compile(r"\b(fee|fees|tuition|cost|price|charges?|how much)\b", re.IGNORECASE)
VISIT_PATTERN = re.compile(r"\b(visit|visiting|tour|campus tour|come to school|open day|appointment)\b", re.IGNORECASE)
ADMISSION_PATTERN = re.compile(
    r"\b(admission|admissions|admit|enrol|enroll|enrollment|apply|application|register|registration|age criteria|cutoff|eligibility)\b",
    re.IGNORECASE,
)
CONTACT_PATTERN = re.compile(
    r"\b(contact|phone|call|telephone|email|number|reach|address|location|reception|office)\b",
    re.IGNORECASE,
)
CALLBACK_PATTERN = re.compile(
    r"\b(call me|callback|call back|request a call|speak to someone|arrange a call|reach out to me)\b",
    re.IGNORECASE,
)

SCHOOL_TERMS = (
    "school", "baldwin", "bbhs", "timing", "timings", "hours", "curriculum",
    "icse", "syllabus", "subject", "facilities", "sports", "library",
    "uniform", "bus", "transport", "principal", "teacher", "hostel", "class", "grade"
)


def classify_intent(question: str, session_state: SessionState) -> AgentIntent:
    """Determine user intent based on the current turn and active conversation state."""
    normalized = normalize_question(question)
    if not normalized:
        return AgentIntent.SMALL_TALK

    # 1. Deterministic system tools
    if TIME_QUERY_PATTERN.search(normalized):
        return AgentIntent.TOOL_CALL

    # 2. Ongoing callback consent flow
    if session_state.pending_consent:
        if is_consent_negative(question):
            return AgentIntent.LEAD_CONSENT_DECLINED
        if is_consent_affirmative(question) or extract_phone_from_text(question):
            return AgentIntent.LEAD_CONSENT_GRANTED
        # If user asks another question instead of giving consent, exit pending state
        session_state.pending_consent = False

    # 3. Explicit callback request
    if CALLBACK_PATTERN.search(normalized):
        return AgentIntent.LEAD_CALLBACK_REQUEST

    # 4. Greetings and pleasantries
    if GREETING_PATTERN.search(normalized) or normalized in {"thanks", "thank you", "bye", "goodbye", "ok", "okay"}:
        return AgentIntent.SMALL_TALK

    # 5. Fee enquiries (e.g. "What is the Grade 8 fee?")
    if FEE_PATTERN.search(normalized):
        return AgentIntent.ADMISSION_FEE_ENQUIRY

    # 6. School visits & campus tours
    if VISIT_PATTERN.search(normalized):
        return AgentIntent.SCHOOL_VISIT

    # 7. Contact information (e.g. "admissions phone number", "who to call")
    if CONTACT_PATTERN.search(normalized):
        return AgentIntent.CONTACT_REQUEST

    # 8. Admissions process
    if ADMISSION_PATTERN.search(normalized):
        return AgentIntent.ADMISSION_PROCESS

    # 9. General school FAQ vs Out of scope
    if any(term in normalized for term in SCHOOL_TERMS) or len(normalized.split()) <= 4:
        return AgentIntent.SCHOOL_FAQ

    return AgentIntent.SCHOOL_FAQ


class AgentGraphState(TypedDict, total=False):
    """In-memory turn state passed through the LangGraph pipeline."""

    question: str
    chat_history: list[dict[str, str]]
    session_state: SessionState
    extracted_entities: dict[str, Any]
    intent: AgentIntent
    grade: str | None
    phone: str | None

    # Resources injected per-turn
    vector_db: Chroma | None
    bm25_index: BM25Index | None
    llm: Any
    cache: dict[str, dict[str, object]] | None
    ensemble_retriever: EnsembleRetriever | None
    answer_question_fn: Any

    # Final outputs
    response_answer: str
    response_actions: list[ActionItem]


# ---------------------------------------------------------------------------
# Graph Nodes
# ---------------------------------------------------------------------------

def node_extract_entities(state: AgentGraphState) -> dict[str, Any]:
    """Node: Extract student grade and phone entities from user query."""
    question = state["question"]
    session_state = state.get("session_state") or SessionState()
    extracted = extract_entities(question)

    if extracted.get("grade"):
        session_state.grade = extracted["grade"]
        session_state.stage = EnquiryStage.IDENTIFYING_GRADE

    return {
        "session_state": session_state,
        "extracted_entities": extracted,
        "grade": session_state.grade,
    }


def node_classify_intent(state: AgentGraphState) -> dict[str, Any]:
    """Node: Classify intent based on extracted entities and session state."""
    question = state["question"]
    session_state = state["session_state"]
    intent = classify_intent(question, session_state)
    session_state.last_intent = intent.value

    log_step(
        "AGENT_DECISION",
        intent=intent.value,
        grade=session_state.grade or "none",
        stage=session_state.stage.value,
    )
    return {"intent": intent}


def node_tool(state: AgentGraphState) -> dict[str, Any]:
    """Node: Handle system tools (e.g. current time query)."""
    res = handle_time_query(state["session_state"])
    return {"response_answer": res.answer, "response_actions": res.actions}


def node_small_talk(state: AgentGraphState) -> dict[str, Any]:
    """Node: Handle polite greetings and small talk."""
    res = handle_small_talk(state["question"], state["session_state"])
    return {"response_answer": res.answer, "response_actions": res.actions}


def node_lead(state: AgentGraphState) -> dict[str, Any]:
    """Node: Handle lead callback request, consent granting, or consent declining."""
    extracted = state.get("extracted_entities") or {}
    res = handle_lead_callback(
        state["question"],
        state["session_state"],
        extracted_phone=extracted.get("phone"),
    )
    return {"response_answer": res.answer, "response_actions": res.actions}


def node_contact(state: AgentGraphState) -> dict[str, Any]:
    """Node: Handle direct contact queries with clickable action pills."""
    res = handle_contact_request(state["question"], state["session_state"])
    return {"response_answer": res.answer, "response_actions": res.actions}


def node_visit(state: AgentGraphState) -> dict[str, Any]:
    """Node: Handle campus tour and school visit requests."""
    res = handle_school_visit(state["question"], state["session_state"])
    return {"response_answer": res.answer, "response_actions": res.actions}


def node_fee(state: AgentGraphState) -> dict[str, Any]:
    """Node: Handle fee inquiries with RAG retrieval and mandatory fallback."""
    session_state = state["session_state"]
    ensemble_retriever = state.get("ensemble_retriever")
    llm = state.get("llm")

    docs = []
    if ensemble_retriever:
        search_term = f"{session_state.grade or ''} fee structure tuition charges admission"
        try:
            docs = ensemble_retriever.invoke(search_term)
        except Exception:
            docs = []

    res = handle_fee_enquiry(state["question"], session_state, docs=docs, llm=llm)
    return {"response_answer": res.answer, "response_actions": res.actions}


def node_admission(state: AgentGraphState) -> dict[str, Any]:
    """Node: Handle admissions process, application forms, and eligibility."""
    session_state = state["session_state"]
    ensemble_retriever = state.get("ensemble_retriever")
    llm = state.get("llm")

    docs = []
    if ensemble_retriever:
        search_term = f"{session_state.grade or ''} admission application process eligibility"
        try:
            docs = ensemble_retriever.invoke(search_term)
        except Exception:
            docs = []

    res = handle_admissions_process(state["question"], session_state, docs=docs, llm=llm)
    return {"response_answer": res.answer, "response_actions": res.actions}


def node_out_of_scope(state: AgentGraphState) -> dict[str, Any]:
    """Node: Handle out-of-scope questions gracefully directing to school contacts."""
    res = handle_out_of_scope(state["question"], state["session_state"])
    return {"response_answer": res.answer, "response_actions": res.actions}


def node_knowledge(state: AgentGraphState) -> dict[str, Any]:
    """Node: Default factual RAG knowledge query against active documents."""
    question = state["question"]
    session_state = state["session_state"]
    vector_db = state.get("vector_db")
    bm25_index = state.get("bm25_index")
    llm = state.get("llm")
    cache = state.get("cache")
    chat_history = state.get("chat_history") or []
    ensemble_retriever = state.get("ensemble_retriever")
    answer_question_fn = state.get("answer_question_fn")

    if vector_db and bm25_index and llm and cache is not None:
        res = handle_knowledge_query(
            question=question,
            session_state=session_state,
            vector_db=vector_db,
            bm25_index=bm25_index,
            llm=llm,
            cache=cache,
            chat_history=chat_history,
            ensemble_retriever=ensemble_retriever,
            answer_question_fn=answer_question_fn,
        )
    else:
        res = handle_small_talk(question, session_state)

    return {"response_answer": res.answer, "response_actions": res.actions}


# ---------------------------------------------------------------------------
# Conditional Edge Router
# ---------------------------------------------------------------------------

def route_by_intent(state: AgentGraphState) -> str:
    """Route to corresponding graph node based on classified intent."""
    intent = state.get("intent")
    if intent == AgentIntent.TOOL_CALL:
        return "tool_node"
    if intent == AgentIntent.SMALL_TALK:
        return "small_talk_node"
    if intent in (
        AgentIntent.LEAD_CALLBACK_REQUEST,
        AgentIntent.LEAD_CONSENT_GRANTED,
        AgentIntent.LEAD_CONSENT_DECLINED,
    ):
        return "lead_node"
    if intent == AgentIntent.CONTACT_REQUEST:
        return "contact_node"
    if intent == AgentIntent.SCHOOL_VISIT:
        return "visit_node"
    if intent == AgentIntent.ADMISSION_FEE_ENQUIRY:
        return "fee_node"
    if intent == AgentIntent.ADMISSION_PROCESS:
        return "admission_node"
    if intent == AgentIntent.OUT_OF_SCOPE:
        return "out_of_scope_node"
    return "knowledge_node"


# ---------------------------------------------------------------------------
# Graph Construction & Compilation
# ---------------------------------------------------------------------------

def build_agent_graph() -> Any:
    """Build and compile the LangGraph StateGraph for the admissions assistant."""
    builder = StateGraph(AgentGraphState)

    # 1. Register nodes
    builder.add_node("extract_entities", node_extract_entities)
    builder.add_node("classify_intent", node_classify_intent)
    builder.add_node("tool_node", node_tool)
    builder.add_node("small_talk_node", node_small_talk)
    builder.add_node("lead_node", node_lead)
    builder.add_node("contact_node", node_contact)
    builder.add_node("visit_node", node_visit)
    builder.add_node("fee_node", node_fee)
    builder.add_node("admission_node", node_admission)
    builder.add_node("out_of_scope_node", node_out_of_scope)
    builder.add_node("knowledge_node", node_knowledge)

    # 2. Sequential entry flow
    builder.add_edge(START, "extract_entities")
    builder.add_edge("extract_entities", "classify_intent")

    # 3. Conditional dispatch based on classified intent
    builder.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "tool_node": "tool_node",
            "small_talk_node": "small_talk_node",
            "lead_node": "lead_node",
            "contact_node": "contact_node",
            "visit_node": "visit_node",
            "fee_node": "fee_node",
            "admission_node": "admission_node",
            "out_of_scope_node": "out_of_scope_node",
            "knowledge_node": "knowledge_node",
        },
    )

    # 4. Connect all handler nodes to terminal END
    for handler_node in [
        "tool_node",
        "small_talk_node",
        "lead_node",
        "contact_node",
        "visit_node",
        "fee_node",
        "admission_node",
        "out_of_scope_node",
        "knowledge_node",
    ]:
        builder.add_edge(handler_node, END)

    return builder.compile()


# Precompiled singleton graph instance for fast in-memory execution
admissions_graph = build_agent_graph()

