"""General conversation handler for greetings, small talk, time queries, and clarifications."""

from src.agents.state import ActionItem, AgentResponse, SessionState
from src.config import ADMISSION_PHONE, ASSISTANT_NAME
from src.core.tools import call_time_tool


def handle_time_query(session_state: SessionState) -> AgentResponse:
    """Handle deterministic current time query."""
    time_str = call_time_tool()
    return AgentResponse(answer=time_str, actions=[], state=session_state)


def handle_small_talk(question: str, session_state: SessionState) -> AgentResponse:
    """Handle greetings, polite pleasantries, and general introductions."""
    grade_context = f" for {session_state.grade}" if session_state.grade else ""
    answer = (
        f"Hello! Welcome to Baldwin Boys' High School. I am {ASSISTANT_NAME}, your admissions "
        f"and school guidance assistant{grade_context}. How can I assist you today? "
        "I can help you with admissions, fee enquiries, campus visits, school timings, "
        "facilities, or curriculum details."
    )
    actions = [
        ActionItem(label="Admissions Process", action="quick_reply", value="Tell me about the admission process", icon="message"),
        ActionItem(label="Fee Enquiry", action="quick_reply", value="What is the fee structure?", icon="message"),
        ActionItem(label="Campus Visit", action="quick_reply", value="How can I visit the school?", icon="calendar"),
        ActionItem(label="Call Admissions", action="tel", value=ADMISSION_PHONE, icon="phone"),
    ]
    return AgentResponse(answer=answer, actions=actions, state=session_state)


def handle_out_of_scope(question: str, session_state: SessionState) -> AgentResponse:
    """Provide a polite institutional boundary when the question is unrelated to the school."""
    answer = (
        "I am dedicated to helping parents and students with Baldwin Boys' High School. "
        "I would be glad to assist you with admissions, fees, curriculum, campus visits, "
        "facilities, or school timings. What would you like to explore?"
    )
    actions = [
        ActionItem(label="Admissions Process", action="quick_reply", value="Tell me about admissions", icon="message"),
        ActionItem(label="Fee Structure", action="quick_reply", value="Tell me about school fees", icon="message"),
        ActionItem(label="Contact Admissions", action="tel", value=ADMISSION_PHONE, icon="phone"),
    ]
    return AgentResponse(answer=answer, actions=actions, state=session_state)

