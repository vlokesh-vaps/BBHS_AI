"""Fee enquiry handler with Knowledge Base search and mandatory institutional fallback."""

import re
from typing import Any
from langchain_core.documents import Document

from src.agents.state import ActionItem, AgentResponse, EnquiryStage, SessionState
from src.config import FEE_ENQUIRY_PHONE


def _has_verified_fee_in_docs(docs: list[Document], grade: str) -> bool:
    """Determine whether retrieved documents contain an explicit, verified fee breakdown for the grade."""
    if not docs:
        return False

    grade_normalized = grade.lower().replace("-", " ")
    grade_digits = re.findall(r"\d+", grade_normalized)
    digit_str = grade_digits[0] if grade_digits else None

    # Specific school fee terms (tuition, term fee, annual fee breakdown)
    fee_terms = ("tuition fee", "term fee", "annual fee", "annual tuition", "composite fee")
    # Amounts representing actual tuition (excluding small application/registration form charges like ₹700)
    substantial_currency_patterns = (
        r"₹\s*([1-9]\d{3,}|[1-9]\d{1,2},\d{3})",
        r"rs\.?\s*([1-9]\d{3,}|[1-9]\d{1,2},\d{3})",
    )

    for doc in docs:
        content = doc.page_content.lower()
        has_fee_term = any(term in content for term in fee_terms)
        has_substantial_currency = any(re.search(pat, content) for pat in substantial_currency_patterns)

        # Check grade mention in document chunk
        grade_mentioned = grade_normalized in content
        if not grade_mentioned and digit_str:
            grade_mentioned = bool(
                re.search(rf"\b(?:grade|class|std\.?|standard)\s*{digit_str}\b", content)
                or re.search(rf"\b{digit_str}\s*(?:st|nd|rd|th)\b", content)
            )

        if has_fee_term and has_substantial_currency and grade_mentioned:
            return True

    return False


def handle_fee_enquiry(
    question: str,
    session_state: SessionState,
    docs: list[Document] | None = None,
    llm: Any = None,
) -> AgentResponse:
    """Handle school fee enquiries for Baldwin Boys' High School.

    Follows the mandatory fallback requirement:
    - If a specific grade is identified and verified fee details exist in KB,
      answers accurately from context.
    - If verified fee figures are unavailable in the KB, never guesses amounts;
      instead, delivers the official Admissions team contact and 3 next-step actions.
    - If no grade is known, asks which class the parent is enquiring about.
    """
    grade = session_state.grade
    session_state.enquiry_type = "fee"

    # If grade is not known yet
    if not grade:
        session_state.stage = EnquiryStage.IDENTIFYING_GRADE
        answer = (
            "I'd be happy to help with the fee details for Baldwin Boys' High School. "
            "The fee structure varies by class. Which class or grade are you enquiring about?"
        )
        actions = [
            ActionItem(label="Pre-KG / LKG", action="quick_reply", value="What is the fee for LKG?", icon="message"),
            ActionItem(label="Grade 1", action="quick_reply", value="What is the Grade 1 fee?", icon="message"),
            ActionItem(label="Grade 5", action="quick_reply", value="What is the Grade 5 fee?", icon="message"),
            ActionItem(label="Grade 8", action="quick_reply", value="What is the Grade 8 fee?", icon="message"),
            ActionItem(label=f"Call Admissions ({FEE_ENQUIRY_PHONE})", action="tel", value=FEE_ENQUIRY_PHONE, icon="phone"),
        ]
        return AgentResponse(answer=answer, actions=actions, state=session_state)

    # If grade is known, check if verified fee details are available in the retrieved docs
    verified_available = False
    if docs:
        verified_available = _has_verified_fee_in_docs(docs, grade)

    if verified_available and llm:
        # Generate grounded response using LLM with RAG docs
        from src.core.rag import format_context_docs
        context = format_context_docs(docs)
        prompt = (
            f"Context:\n{context}\n\n"
            f"Question: What is the fee structure for {grade}?\n\n"
            "Answer using ONLY the provided context. State the verified fee breakdown clearly, "
            "and suggest contacting the Admissions office for payment terms or installment schedules."
        )
        try:
            llm_res = llm.invoke(prompt)
            answer_text = llm_res.content if hasattr(llm_res, "content") else str(llm_res)
            actions = [
                ActionItem(label="Admission Process", action="quick_reply", value=f"What is the admission process for {grade}?", icon="message"),
                ActionItem(label="Explore School Visit", action="quick_reply", value="How can I visit the school?", icon="calendar"),
                ActionItem(label=f"Contact Admissions ({FEE_ENQUIRY_PHONE})", action="tel", value=FEE_ENQUIRY_PHONE, icon="phone"),
            ]
            session_state.stage = EnquiryStage.INFORMATION_REQUESTED
            return AgentResponse(answer=answer_text.strip(), actions=actions, state=session_state)
        except Exception:
            pass  # Fall through to deterministic institutional fallback

    # Mandatory institutional fallback for unverified / missing fee breakdown
    session_state.stage = EnquiryStage.CONTACT_OFFERED
    session_state.contact_offered = True

    answer = (
        f"**Enquiring about {grade} at BBHS?**\n\n"
        f"We'd be happy to help you with the fee details for your child's admission to Baldwin Boys' High School.\n\n"
        f"**Contact our admissions team:** {FEE_ENQUIRY_PHONE} for the latest {grade} fee structure and applicable charges.\n\n"
        "**How would you like to proceed?**\n\n"
        "- Learn about the admission process\n"
        "- Enquire about the fee structure\n"
        "- Explore a school visit"
    )

    actions = [
        ActionItem(label=f"Call Admissions ({FEE_ENQUIRY_PHONE})", action="tel", value=FEE_ENQUIRY_PHONE, icon="phone"),
        ActionItem(label="Learn about the admission process", action="quick_reply", value=f"What is the admission process for {grade}?", icon="message"),
        ActionItem(label="Enquire about the fee structure", action="quick_reply", value=f"Can admissions contact me regarding {grade} fees?", icon="phone"),
        ActionItem(label="Explore a school visit", action="quick_reply", value="How can I visit the school?", icon="calendar"),
    ]

    return AgentResponse(answer=answer, actions=actions, state=session_state)
