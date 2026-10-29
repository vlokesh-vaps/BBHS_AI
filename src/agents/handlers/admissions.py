"""Admissions process and portal guidance handler."""

from typing import Any
from langchain_core.documents import Document

from src.agents.state import ActionItem, AgentResponse, EnquiryStage, SessionState
from src.config import ADMISSION_PHONE, ONLINE_APPLICATION_URL


def handle_admissions_process(
    question: str,
    session_state: SessionState,
    docs: list[Document] | None = None,
    llm: Any = None,
) -> AgentResponse:
    """Guide prospective parents through the BBHS admission process."""
    grade = session_state.grade
    session_state.enquiry_type = "admission"
    grade_phrase = f" for {grade}" if grade else ""

    # If docs are present and LLM is available, generate grounded guidance
    if docs and llm:
        from src.core.rag import format_context_docs
        context = format_context_docs(docs)
        prompt = (
            f"Context:\n{context}\n\n"
            f"Question: How can a parent apply for admission{grade_phrase} at Baldwin Boys' High School?\n\n"
            "Provide clear, warm, step-by-step guidance based strictly on the context. Mention required documents "
            "or eligibility criteria if mentioned. State that admissions applications are submitted online, "
            "and suggest contacting the Admissions desk for verification or assistance."
        )
        try:
            llm_res = llm.invoke(prompt)
            answer_text = llm_res.content if hasattr(llm_res, "content") else str(llm_res)
            actions = [
                ActionItem(label="Apply Online (Portal)", action="link", value=ONLINE_APPLICATION_URL, icon="external-link"),
                ActionItem(
                    label=f"Fee Enquiry ({grade})" if grade else "Fee Structure",
                    action="quick_reply",
                    value=f"What is the fee for {grade}?" if grade else "What is the fee structure?",
                    icon="message",
                ),
                ActionItem(label="Explore Campus Visit", action="quick_reply", value="How can I visit the school?", icon="calendar"),
                ActionItem(label=f"Call Admissions ({ADMISSION_PHONE})", action="tel", value=ADMISSION_PHONE, icon="phone"),
            ]
            session_state.stage = EnquiryStage.INFORMATION_REQUESTED
            return AgentResponse(answer=answer_text.strip(), actions=actions, state=session_state)
        except Exception:
            pass

    # Default institutional admission guidance
    session_state.stage = EnquiryStage.INFORMATION_REQUESTED
    answer = (
        f"**Admission to Baldwin Boys' High School{grade_phrase}**\n\n"
        "Baldwin Boys' High School welcomes prospective students based on academic readiness "
        "and availability of seats.\n\n"
        "**Key Next Steps:**\n"
        "1. **Online Application:** Parents can register and submit the admission form directly "
        f"through the [BBHS Admissions Portal]({ONLINE_APPLICATION_URL}).\n"
        "2. **Documentation:** Keep the child's birth certificate, previous school report cards, "
        "and transfer certificate ready.\n"
        "3. **Interaction & Verification:** The admissions office will review your application and "
        "notify you regarding the student interaction date.\n\n"
        f"For any queries regarding eligibility or seat availability, call the Admissions team at **{ADMISSION_PHONE}**."
    )

    actions = [
        ActionItem(label="Apply Online (Portal)", action="link", value=ONLINE_APPLICATION_URL, icon="external-link"),
        ActionItem(
            label=f"Enquire about {grade} Fees" if grade else "Enquire about Fees",
            action="quick_reply",
            value=f"What is the fee for {grade}?" if grade else "What is the fee structure?",
            icon="message",
        ),
        ActionItem(label="Explore School Visit", action="quick_reply", value="How can I visit the school for admissions?", icon="calendar"),
        ActionItem(label=f"Call Admissions ({ADMISSION_PHONE})", action="tel", value=ADMISSION_PHONE, icon="phone"),
    ]

    return AgentResponse(answer=answer, actions=actions, state=session_state)

