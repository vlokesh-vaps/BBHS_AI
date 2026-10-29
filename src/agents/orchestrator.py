"""Agent Orchestrator: Multi-intent classifier, contextual state manager, and handler dispatcher."""

import re
from typing import Any
from langchain_chroma import Chroma

from src.agents.entity_extractor import (
    extract_entities,
    extract_grade,
    is_consent_affirmative,
    is_consent_negative,
)
from src.agents.handlers import (
    handle_admissions_process,
    handle_contact_request,
    handle_fee_enquiry,
    handle_knowledge_query,
    handle_lead_callback,
    handle_out_of_scope,
    handle_school_visit,
    handle_small_talk,
    handle_time_query,
)
from src.agents.state import (
    ActionItem,
    AgentDecision,
    AgentIntent,
    AgentResponse,
    EnquiryStage,
    SessionState,
)
from src.config import (
    GREETING_PATTERN,
    TIME_QUERY_PATTERN,
    normalize_question,
)
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


class AgentOrchestrator:
    """Central orchestrator for the BBHS Conversational Admissions Agent."""

    def __init__(self) -> None:
        pass

    def classify_intent(self, question: str, session_state: SessionState) -> AgentIntent:
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
        # If question contains typical school terms:
        school_terms = (
            "school", "baldwin", "bbhs", "timing", "timings", "hours", "curriculum",
            "icse", "syllabus", "subject", "facilities", "sports", "library",
            "uniform", "bus", "transport", "principal", "teacher", "hostel", "class", "grade"
        )
        if any(term in normalized for term in school_terms) or len(normalized.split()) <= 4:
            return AgentIntent.SCHOOL_FAQ

        return AgentIntent.SCHOOL_FAQ

    def process_turn(
        self,
        question: str,
        chat_history: list[dict[str, str]],
        session_state: SessionState | None = None,
        vector_db: Chroma | None = None,
        bm25_index: BM25Index | None = None,
        llm: Any = None,
        cache: dict[str, dict[str, object]] | None = None,
        ensemble_retriever: EnsembleRetriever | None = None,
        answer_question_fn: Any = None,
    ) -> AgentResponse:
        """Process a conversational turn and return a grounded, contextual response with actions.

        Internal decision making is structured; no chain-of-thought is revealed to the user.
        """
        state = session_state or SessionState()

        # Extract entities from the user turn
        extracted = extract_entities(question)
        if extracted.get("grade"):
            state.grade = extracted["grade"]
            state.stage = EnquiryStage.IDENTIFYING_GRADE

        intent = self.classify_intent(question, state)
        state.last_intent = intent.value

        decision = AgentDecision(
            intent=intent,
            grade=state.grade,
            confidence=1.0,
            selected_handler=intent.value,
        )
        log_step(
            "AGENT_DECISION",
            intent=intent.value,
            grade=state.grade or "none",
            stage=state.stage.value,
        )

        # Dispatch based on classified intent
        if intent == AgentIntent.TOOL_CALL:
            return handle_time_query(state)

        if intent == AgentIntent.SMALL_TALK:
            return handle_small_talk(question, state)

        if intent in (AgentIntent.LEAD_CALLBACK_REQUEST, AgentIntent.LEAD_CONSENT_GRANTED, AgentIntent.LEAD_CONSENT_DECLINED):
            return handle_lead_callback(question, state, extracted_phone=extracted.get("phone"))

        if intent == AgentIntent.CONTACT_REQUEST:
            return handle_contact_request(question, state)

        if intent == AgentIntent.SCHOOL_VISIT:
            return handle_school_visit(question, state)

        if intent == AgentIntent.ADMISSION_FEE_ENQUIRY:
            # Perform RAG search for fee query first
            docs = []
            if ensemble_retriever:
                search_term = f"{state.grade or ''} fee structure tuition charges admission"
                try:
                    docs = ensemble_retriever.invoke(search_term)
                except Exception:
                    docs = []
            return handle_fee_enquiry(question, state, docs=docs, llm=llm)

        if intent == AgentIntent.ADMISSION_PROCESS:
            docs = []
            if ensemble_retriever:
                search_term = f"{state.grade or ''} admission application process eligibility"
                try:
                    docs = ensemble_retriever.invoke(search_term)
                except Exception:
                    docs = []
            return handle_admissions_process(question, state, docs=docs, llm=llm)

        if intent == AgentIntent.OUT_OF_SCOPE:
            return handle_out_of_scope(question, state)

        # Default: SCHOOL_FAQ (Knowledge Agent)
        if vector_db and bm25_index and llm and cache is not None:
            return handle_knowledge_query(
                question=question,
                session_state=state,
                vector_db=vector_db,
                bm25_index=bm25_index,
                llm=llm,
                cache=cache,
                chat_history=chat_history,
                ensemble_retriever=ensemble_retriever,
                answer_question_fn=answer_question_fn,
            )

        # Fallback when vector_db is not loaded
        return handle_small_talk(question, state)
