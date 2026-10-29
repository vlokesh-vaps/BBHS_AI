"""Contact handler providing verified institutional communication channels."""

from src.agents.state import ActionItem, AgentResponse, SessionState
from src.config import (
    ADMISSION_EMAIL,
    ADMISSION_PHONE,
    GENERAL_ENQUIRY_EMAIL,
    GENERAL_ENQUIRY_PHONE,
)


def handle_contact_request(question: str, session_state: SessionState) -> AgentResponse:
    """Provide verified Baldwin Boys' High School contact information."""
    grade_prefix = f"Regarding {session_state.grade}: " if session_state.grade else ""
    answer = (
        f"{grade_prefix}You can reach the Baldwin Boys' High School Admissions and School Office "
        f"directly through the verified channels below:\n\n"
        f"- **Admissions Office:** {ADMISSION_PHONE}\n"
        f"- **Admissions Email:** {ADMISSION_EMAIL}\n"
        f"- **General Enquiries:** {GENERAL_ENQUIRY_PHONE} | {GENERAL_ENQUIRY_EMAIL}\n"
        f"- **Campus Location:** Richmond Town, Hosur Road, Bengaluru, Karnataka 560025\n\n"
        "Our office is open on working days during school hours. Feel free to call us or request a callback."
    )
    actions = [
        ActionItem(label=f"Call Admissions ({ADMISSION_PHONE})", action="tel", value=ADMISSION_PHONE, icon="phone"),
        ActionItem(label="Email Admissions", action="link", value=f"mailto:{ADMISSION_EMAIL}", icon="mail"),
        ActionItem(label="Request a Callback", action="quick_reply", value="Please arrange a callback from admissions", icon="phone"),
    ]
    session_state.contact_offered = True
    return AgentResponse(answer=answer, actions=actions, state=session_state)

