"""Pydantic models for API request/response validation."""

from pydantic import BaseModel, Field

from src.agents.state import ActionItem


class ChatRequest(BaseModel):
    """Request model for the chat endpoint."""

    message: str = Field(..., min_length=1, max_length=4000, description="User message")
    session_id: str = Field(
        default="default",
        min_length=1,
        max_length=128,
        description="In-memory conversation session ID",
    )


class ChatResponse(BaseModel):
    """Response model for the chat endpoint."""

    session_id: str = Field(..., description="Conversation session ID")
    answer: str = Field(..., description="AI response")
    actions: list[ActionItem] = Field(
        default_factory=list,
        description="Optional list of interactive action pills for the parent",
    )
