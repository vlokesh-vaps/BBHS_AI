"""Pydantic schemas for consent-based school admission leads."""

from pydantic import BaseModel, Field


class LeadRecord(BaseModel):
    """Structured record of a parent callback enquiry submitted with explicit consent."""

    lead_id: str = Field(..., description="Unique lead identifier")
    timestamp: str = Field(..., description="ISO 8601 creation timestamp")
    grade: str | None = Field(default=None, description="Target admission grade or class")
    phone: str = Field(..., description="Validated 10-digit parent phone number")
    enquiry_type: str = Field(default="admission", description="Type of enquiry (admission, fee, visit)")
    consent_granted: bool = Field(default=True, description="Explicit consent flag")
    consent_timestamp: str = Field(..., description="Timestamp when user gave consent")

