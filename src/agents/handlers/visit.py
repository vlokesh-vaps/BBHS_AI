"""School visit handler for prospective parent campus tours and appointments."""

from src.agents.state import ActionItem, AgentResponse, SessionState
from src.config import ADMISSION_EMAIL, ADMISSION_PHONE


def handle_school_visit(question: str, session_state: SessionState) -> AgentResponse:
    """Handle campus tour and in-person school visit enquiries safely.

    Strict grounding: visiting hours can vary depending on exams and school schedule,
    so parents are guided to confirm with Admissions without hallucinating fixed hours.
    """
    grade_note = f" for {session_state.grade}" if session_state.grade else ""
    answer = (
        f"We would be delighted to welcome you to Baldwin Boys' High School{grade_note}.\n\n"
        "Parents are welcome to visit our Richmond Town campus for admission enquiries "
        "and guided walk-throughs during working hours. Because timings may vary based on "
        "the academic calendar and examination schedules, we strongly recommend confirming "
        f"your visit with the Admissions team at **{ADMISSION_PHONE}** or **{ADMISSION_EMAIL}** "
        "before you arrive."
    )
    actions = [
        ActionItem(label=f"Call Admissions ({ADMISSION_PHONE})", action="tel", value=ADMISSION_PHONE, icon="phone"),
        ActionItem(label="Request a Callback", action="quick_reply", value="I would like a callback to schedule a visit", icon="phone"),
        ActionItem(label="Learn about Admissions", action="quick_reply", value="Tell me about the admission process", icon="message"),
    ]
    session_state.contact_offered = True
    return AgentResponse(answer=answer, actions=actions, state=session_state)

