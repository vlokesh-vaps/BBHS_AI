"""State definitions, structured decision models, and action contracts for conversational agents."""

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class AgentIntent(str, Enum):
    """Categorized user intent for conversational routing."""

    ADMISSION_FEE_ENQUIRY = "admission_fee_enquiry"
    ADMISSION_PROCESS = "admission_process"
    SCHOOL_VISIT = "school_visit"
    CONTACT_REQUEST = "contact_request"
    LEAD_CALLBACK_REQUEST = "lead_callback_request"
    LEAD_CONSENT_GRANTED = "lead_consent_granted"
    LEAD_CONSENT_DECLINED = "lead_consent_declined"
    SCHOOL_FAQ = "school_faq"
    TOOL_CALL = "tool_call"
    SMALL_TALK = "small_talk"
    OUT_OF_SCOPE = "out_of_scope"


class EnquiryStage(str, Enum):
    """Conversational lifecycle stage for an admission enquiry."""

    NEW_ENQUIRY = "new_enquiry"
    IDENTIFYING_GRADE = "identifying_grade"
    INFORMATION_REQUESTED = "information_requested"
    CONTACT_OFFERED = "contact_offered"
    CALLBACK_CONSENT_REQUESTED = "callback_consent_requested"
    LEAD_SUBMITTED = "lead_submitted"
    CONSENT_DECLINED = "consent_declined"
    COMPLETED = "completed"


class ActionItem(BaseModel):
    """Contextual interactive action rendered as a clickable pill in the UI."""

    label: str = Field(..., description="Display label for the action button")
    action: str = Field(..., description="Action type: 'tel', 'link', or 'quick_reply'")
    value: str = Field(..., description="Action payload, telephone URI, URL, or message text")
    icon: str | None = Field(default=None, description="Optional icon identifier: phone, external-link, message")


class SessionState(BaseModel):
    """Transient in-memory conversation state preserved across turns."""

    grade: str | None = Field(default=None, description="Active extracted student grade/class (e.g., Grade 8)")
    stage: EnquiryStage = Field(default=EnquiryStage.NEW_ENQUIRY, description="Current enquiry funnel stage")
    last_intent: str | None = Field(default=None, description="Previous turn's recognized intent")
    pending_consent: bool = Field(default=False, description="True if awaiting callback permission from parent")
    contact_offered: bool = Field(default=False, description="True if admissions contact has been displayed")
    phone: str | None = Field(default=None, description="Validated 10-digit phone number if consent provided")
    parent_name: str | None = Field(default=None, description="Parent or guardian name if provided")
    enquiry_type: str | None = Field(default=None, description="Type of enquiry: fee, admission, visit")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional ephemeral turn metadata")


class AgentDecision(BaseModel):
    """Structured internal decision record (strictly non-user-facing per privacy rules)."""

    intent: AgentIntent
    grade: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    selected_handler: str
    information_available: bool | None = None
    suggested_actions: list[ActionItem] = Field(default_factory=list)


class AgentResponse(BaseModel):
    """Standardized response produced by the agent orchestrator."""

    answer: str
    actions: list[ActionItem] = Field(default_factory=list)
    state: SessionState | None = None

