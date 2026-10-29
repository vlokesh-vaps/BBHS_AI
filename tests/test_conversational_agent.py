"""Comprehensive test suite for BBHS Conversational Admissions Agent."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.agents.entity_extractor import (
    extract_entities,
    extract_grade,
    is_consent_affirmative,
    is_consent_negative,
)
from src.agents.orchestrator import AgentOrchestrator
from src.agents.state import (
    ActionItem,
    AgentIntent,
    EnquiryStage,
    SessionState,
)
from src.api.app import BoundedSessionStore, app
from src.config import (
    ADMISSION_EMAIL,
    ADMISSION_PHONE,
    FEE_ENQUIRY_PHONE,
    ONLINE_APPLICATION_URL,
)
from src.leads.service import clean_phone_number, extract_phone_from_text, record_lead


# ============================================================================
# 1. Entity Extraction & Normalization Tests
# ============================================================================

def test_extract_grade_various_patterns():
    assert extract_grade("What is the Grade 8 fee?") == "Grade 8"
    assert extract_grade("i want to know fee for 3 th std") == "Grade 3"
    assert extract_grade("Looking for 10th standard admission") == "Grade 10"
    assert extract_grade("How to apply for class 5?") == "Grade 5"
    assert extract_grade("Admission in LKG") == "LKG"
    assert extract_grade("Enquiry for Pre-KG") == "Pre-KG"
    assert extract_grade("Class VIII admission syllabus") == "Grade 8"
    assert extract_grade("What about first standard?") == "Grade 1"
    assert extract_grade("Tell me about BBHS history") is None


def test_clean_and_extract_phone_number():
    assert clean_phone_number("9880012345") == "9880012345"
    assert clean_phone_number("+91 98800 12345") == "9880012345"
    assert clean_phone_number("09880012345") == "9880012345"
    assert clean_phone_number("12345") is None
    assert clean_phone_number("4880012345") is None  # Does not start with 6-9

    text = "You can call me on 9880012345 after 3 PM"
    assert extract_phone_from_text(text) == "9880012345"


def test_consent_affirmative_and_negative_detection():
    assert is_consent_affirmative("Yes, please have someone call me") is True
    assert is_consent_affirmative("Sure, my number is 9880012345") is True
    assert is_consent_negative("No thanks, do not call me") is True
    assert is_consent_negative("No need to call") is True
    assert is_consent_negative("Yes") is False


# ============================================================================
# 2. Mandatory Fee Fallback & KB Integration Tests
# ============================================================================

def test_fee_unavailable_triggers_mandatory_fallback():
    """Requirement 2: Fee unavailable in KB gives Admissions contact and 3 actions."""
    orchestrator = AgentOrchestrator()
    state = SessionState()
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = []  # No fee info in KB

    response = orchestrator.process_turn(
        question="What is the Grade 8 fee?",
        chat_history=[],
        session_state=state,
        ensemble_retriever=mock_retriever,
    )

    assert "**Enquiring about Grade 8 at BBHS?**" in response.answer
    assert FEE_ENQUIRY_PHONE in response.answer
    assert "8147883880" in response.answer
    assert "Learn about the admission process" in response.answer
    assert "Enquire about the fee structure" in response.answer
    assert "Explore a school visit" in response.answer
    assert state.grade == "Grade 8"

    # Action pills verification
    action_labels = [a.label for a in response.actions]
    assert any("Call Admissions" in label for label in action_labels)
    assert any("Learn about the admission process" in label for label in action_labels)
    assert any("Explore a school visit" in label for label in action_labels)


def test_fee_available_in_kb_answers_verified():
    """Requirement 1: Fee available in KB answers accurately with verified information."""
    orchestrator = AgentOrchestrator()
    state = SessionState()
    mock_retriever = MagicMock()
    doc = Document(
        page_content="Grade 8 Tuition Fee structure: Annual tuition is ₹85,000 payable in two installments.",
        metadata={"page": 12},
    )
    mock_retriever.invoke.return_value = [doc]

    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = "For Grade 8, the annual tuition fee is ₹85,000, payable in two installments."

    response = orchestrator.process_turn(
        question="What is the Grade 8 fee?",
        chat_history=[],
        session_state=state,
        ensemble_retriever=mock_retriever,
        llm=mock_llm,
    )

    assert "₹85,000" in response.answer
    assert state.grade == "Grade 8"
    assert response.actions is not None


# ============================================================================
# 3. Multi-Turn Grade Context Retention Tests
# ============================================================================

def test_grade_context_retention_across_followups():
    """Requirement 3: Grade 8 mentioned earlier is preserved in follow-up enquiry."""
    orchestrator = AgentOrchestrator()
    state = SessionState()

    # Turn 1: User mentions Grade 8 in general
    resp1 = orchestrator.process_turn(
        question="I am looking for Grade 8 admission",
        chat_history=[],
        session_state=state,
    )
    assert state.grade == "Grade 8"

    # Turn 2: User asks follow-up without repeating the grade
    resp2 = orchestrator.process_turn(
        question="What is the fee?",
        chat_history=[
            {"role": "user", "content": "I am looking for Grade 8 admission"},
            {"role": "assistant", "content": resp1.answer},
        ],
        session_state=state,
    )

    # Must retain Grade 8 and output Grade 8 fee fallback
    assert "**Enquiring about Grade 8 at BBHS?**" in resp2.answer
    assert "8147883880" in resp2.answer


# ============================================================================
# 4. School Visit & Contact Agent Tests
# ============================================================================

def test_school_visit_handler_never_invents_hours():
    """Requirement 4: School visit handler does not invent hours and offers verification."""
    orchestrator = AgentOrchestrator()
    state = SessionState(grade="Grade 8")

    response = orchestrator.process_turn(
        question="How can I visit the school?",
        chat_history=[],
        session_state=state,
    )

    assert "Baldwin Boys' High School" in response.answer
    assert "Grade 8" in response.answer
    assert "confirm" in response.answer.lower()
    assert ADMISSION_PHONE in response.answer
    # Must NOT hallucinate arbitrary visiting hours like 8:15 AM
    assert "8:15" not in response.answer
    assert any(a.action == "tel" for a in response.actions)


def test_contact_agent_provides_click_to_call():
    """Requirement 5: 'Call admissions' displays verified click-to-call action."""
    orchestrator = AgentOrchestrator()
    state = SessionState()

    response = orchestrator.process_turn(
        question="What is the admissions phone number?",
        chat_history=[],
        session_state=state,
    )

    assert ADMISSION_PHONE in response.answer
    assert ADMISSION_EMAIL in response.answer

    tel_actions = [a for a in response.actions if a.action == "tel"]
    assert len(tel_actions) > 0
    assert tel_actions[0].value == ADMISSION_PHONE


# ============================================================================
# 5. Consent-Based Lead Management Tests
# ============================================================================

def test_callback_declined_submits_no_lead(tmp_path):
    """Requirement 6: When callback consent is declined, do not submit lead data."""
    dest_file = tmp_path / "leads.jsonl"
    orchestrator = AgentOrchestrator()
    state = SessionState(pending_consent=True, grade="Grade 8")

    response = orchestrator.process_turn(
        question="No, please don't call me",
        chat_history=[],
        session_state=state,
    )

    assert "No callback request has been submitted" in response.answer
    assert state.stage == EnquiryStage.CONSENT_DECLINED
    assert not dest_file.exists()


def test_callback_consent_granted_submits_lead(tmp_path):
    """Requirement 7: Callback consent granted submits valid lead data to isolated store."""
    dest_file = tmp_path / "leads.jsonl"
    orchestrator = AgentOrchestrator()
    state = SessionState(pending_consent=True, grade="Grade 8")

    with patch("src.agents.handlers.lead.record_lead") as mock_record:
        mock_record.side_effect = lambda phone, grade, enquiry_type, consent_granted: record_lead(
            phone=phone,
            grade=grade,
            enquiry_type=enquiry_type,
            consent_granted=consent_granted,
            destination_file=dest_file,
        )

        response = orchestrator.process_turn(
            question="Yes sure, call me on 9880012345",
            chat_history=[],
            session_state=state,
        )

        assert "Callback Request Received" in response.answer
        assert "9880012345" in response.answer
        assert state.stage == EnquiryStage.LEAD_SUBMITTED

        # Verify isolated lead store
        assert dest_file.exists()
        lines = dest_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["phone"] == "9880012345"
        assert record["grade"] == "Grade 8"
        assert record["consent_granted"] is True
        # Verify Invariant D3: no user question or chat transcript in lead store
        assert "Yes sure, call me on 9880012345" not in lines[0]


# ============================================================================
# 6. Safety, Small Talk, and Out-of-Scope Tests
# ============================================================================

def test_small_talk_avoids_rag_retrieval():
    """Requirement 10: Greetings and small talk avoid unnecessary RAG search."""
    orchestrator = AgentOrchestrator()
    state = SessionState()
    mock_retriever = MagicMock()

    response = orchestrator.process_turn(
        question="Hello, good morning!",
        chat_history=[],
        session_state=state,
        ensemble_retriever=mock_retriever,
    )

    # Hybrid retriever must NOT be invoked for small talk
    mock_retriever.invoke.assert_not_called()
    assert "Baldwin Boys' High School" in response.answer
    assert len(response.actions) > 0


def test_time_tool_routing():
    orchestrator = AgentOrchestrator()
    state = SessionState()

    response = orchestrator.process_turn(
        question="what is the time now?",
        chat_history=[],
        session_state=state,
    )

    assert "The current time is" in response.answer


def test_missing_contact_config_never_hallucinates_numbers():
    """Requirement 8: Validated official number is always preserved; no fabricated numbers."""
    orchestrator = AgentOrchestrator()
    state = SessionState()

    response = orchestrator.process_turn(
        question="What is the admissions phone number?",
        chat_history=[],
        session_state=state,
    )

    assert "8147883880" in response.answer
    assert "9999999999" not in response.answer
    assert "080-12345678" not in response.answer


# ============================================================================
# 7. End-to-End FastAPI Webhook Integration with Actions
# ============================================================================

def test_fastapi_webhook_returns_actions_payload():
    from starlette.testclient import TestClient
    with TestClient(app) as client:
        res = client.post(
            "/webhook/chat",
            json={"message": "What is the Grade 8 fee?", "session_id": "test-agent-sess"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["session_id"] == "test-agent-sess"
        assert "**Enquiring about Grade 8 at BBHS?**" in data["answer"]
        assert "actions" in data
        assert isinstance(data["actions"], list)
        assert len(data["actions"]) >= 3

        # Click-to-call action presence
        actions = data["actions"]
        assert any(a["action"] == "tel" and a["value"] == "8147883880" for a in actions)

