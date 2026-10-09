"""Pydantic models for API request/response validation."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request model for the chat endpoint."""

    message: str = Field(..., min_length=1, description="User message")
    session_id: str = Field(default="default", min_length=1, description="Conversation session ID")


class ChatResponse(BaseModel):
    """Response model for the chat endpoint."""

    session_id: str = Field(..., description="Conversation session ID")
    answer: str = Field(..., description="AI response")

