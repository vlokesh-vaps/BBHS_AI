"""Lead management handler for consent-based admissions callback requests."""

from src.agents.entity_extractor import extract_grade, is_consent_affirmative, is_consent_negative
from src.agents.state import ActionItem, AgentResponse, EnquiryStage, SessionState
from src.config import ADMISSION_PHONE
from src.leads.service import clean_phone_number, extract_phone_from_text, record_lead


def handle_lead_callback(
    question: str,
    session_state: SessionState,
    extracted_phone: str | None = None,
) -> AgentResponse:
    """Manage the opt-in admission callback flow strictly with consent.

    Guarantees:
    - Never collects phone numbers aggressively or prematurely.
    - If user declines consent, records nothing and returns to standard assistance.
    - If user provides a valid 10-digit number with consent, records the callback
      and confirms.
    """
    phone = extracted_phone or extract_phone_from_text(question)
    grade = session_state.grade or extract_grade(question)

    # 1. User explicitly declines callback
    if is_consent_negative(question):
        session_state.pending_consent = False
        session_state.stage = EnquiryStage.CONSENT_DECLINED
        answer = (
            "Understood! No callback request has been submitted. "
            "How else may I help you with Baldwin Boys' High School today? "
            "You can continue asking questions about admissions, fees, timings, or facilities."
        )
        actions = [
            ActionItem(label="Admissions Info", action="quick_reply", value="Tell me about the admission process", icon="message"),
            ActionItem(label="Fee Details", action="quick_reply", value="What is the fee structure?", icon="message"),
            ActionItem(label=f"Call Directly ({ADMISSION_PHONE})", action="tel", value=ADMISSION_PHONE, icon="phone"),
        ]
        return AgentResponse(answer=answer, actions=actions, state=session_state)

    # 2. Valid phone number provided
    if phone:
        cleaned = clean_phone_number(phone)
        if cleaned:
            record = record_lead(
                phone=cleaned,
                grade=grade,
                enquiry_type=session_state.enquiry_type or "admission",
                consent_granted=True,
            )
            if record:
                session_state.phone = cleaned
                session_state.pending_consent = False
                session_state.stage = EnquiryStage.LEAD_SUBMITTED
                grade_mention = f" for {grade}" if grade else ""
                answer = (
                    f"**Callback Request Received**\n\n"
                    f"Thank you! Our Admissions team has noted your callback request{grade_mention}. "
                    f"An admissions counselor will contact you on **{cleaned}** during school working hours.\n\n"
                    "If you need immediate assistance, you are also welcome to call our office directly."
                )
                actions = [
                    ActionItem(label=f"Call Admissions Now ({ADMISSION_PHONE})", action="tel", value=ADMISSION_PHONE, icon="phone"),
                    ActionItem(label="Explore Campus Visit", action="quick_reply", value="How can I visit the school?", icon="calendar"),
                    ActionItem(label="Admission Process", action="quick_reply", value="Tell me about the admission process", icon="message"),
                ]
                return AgentResponse(answer=answer, actions=actions, state=session_state)

    # 3. User requested callback or pending consent, but no valid phone given yet
    session_state.pending_consent = True
    session_state.stage = EnquiryStage.CALLBACK_CONSENT_REQUESTED
    grade_phrase = f" for {grade}" if grade else ""
    answer = (
        f"We would be delighted to have our Admissions counselor call you{grade_phrase} to assist with your questions.\n\n"
        "Please reply with your **10-digit mobile number** to authorize a call from our admissions desk. "
        "Your contact details will only be used to answer your school enquiry."
    )
    actions = [
        ActionItem(label=f"Call Directly Instead ({ADMISSION_PHONE})", action="tel", value=ADMISSION_PHONE, icon="phone"),
        ActionItem(label="No, thank you", action="quick_reply", value="No, please don't call me", icon="x"),
    ]
    return AgentResponse(answer=answer, actions=actions, state=session_state)

