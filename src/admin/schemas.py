"""Typed contracts for the AI Control Center API."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelConfig(BaseModel):
    """Model provider settings without secret values."""

    primary_provider: str = Field(default="groq", min_length=1, max_length=50)
    primary_model: str = Field(default="", min_length=1, max_length=200)
    fallback_provider: str = Field(default="", max_length=50)
    fallback_model: str = Field(default="", max_length=200)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1, le=32768)
    request_timeout: int = Field(default=60, ge=1, le=600)
    primary_enabled: bool = True
    fallback_enabled: bool = False

    @model_validator(mode="after")
    def validate_fallback(self) -> "ModelConfig":
        if self.fallback_enabled and not self.fallback_model:
            raise ValueError("fallback_model is required when fallback is enabled")
        return self


class PromptConfig(BaseModel):
    """Editable prompt text, versioned for administrator visibility."""

    system_prompt: str = Field(..., min_length=1, max_length=20000)
    rag_prompt: str = Field(..., min_length=1, max_length=20000)
    no_answer_prompt: str = Field(..., min_length=1, max_length=4000)
    greeting_prompt: str = Field(..., min_length=1, max_length=4000)
    context_instructions: str = Field(..., min_length=1, max_length=10000)
    version: int = Field(default=1, ge=1)
    updated_at: datetime | None = None


class RAGConfig(BaseModel):
    """Retrieval settings supported by the existing hybrid retriever."""

    vector_top_k: int = Field(default=4, ge=1, le=20)
    bm25_top_k: int = Field(default=4, ge=1, le=20)
    final_top_k: int = Field(default=6, ge=1, le=20)
    similarity_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    chunk_size: int = Field(default=500, ge=100, le=5000)
    chunk_overlap: int = Field(default=80, ge=0, le=1000)
    retrieval_strategy: str = Field(default="hybrid", pattern="^hybrid$")

    @model_validator(mode="after")
    def validate_chunking(self) -> "RAGConfig":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class PerformanceConfig(BaseModel):
    """Runtime limits that protect latency and prompt size."""

    cache_enabled: bool = True
    semantic_cache_threshold: float = Field(default=0.82, ge=0.0, le=1.0)
    max_history_messages: int = Field(default=6, ge=0, le=50)
    max_history_chars: int = Field(default=4000, ge=0, le=50000)
    request_timeout: int = Field(default=60, ge=1, le=600)


class GuardrailConfig(BaseModel):
    """Pure-RAG safety settings."""

    strict_rag_mode: bool = True
    allow_general_questions: bool = False
    hallucination_protection: bool = True
    minimum_retrieval_score: float = Field(default=0.0, ge=0.0, le=1.0)
    out_of_scope_response: str = Field(
        default="I don't have information about that in the knowledge base. Please provide more details or rephrase your question.",
        min_length=1,
        max_length=4000,
    )


class AdminConfig(BaseModel):
    """Complete persisted Control Center configuration."""

    model_config = ConfigDict(validate_assignment=True)

    models: ModelConfig
    prompts: PromptConfig
    rag: RAGConfig
    performance: PerformanceConfig
    guardrails: GuardrailConfig
    version: int = Field(default=1, ge=1)
    updated_at: datetime | None = None


class AdminConfigUpdate(BaseModel):
    """Partial update contract used by PUT /admin/config."""

    models: ModelConfig | None = None
    prompts: PromptConfig | None = None
    rag: RAGConfig | None = None
    performance: PerformanceConfig | None = None
    guardrails: GuardrailConfig | None = None


class ConfigValidationRequest(BaseModel):
    """Configuration payload used by the validation endpoint."""

    config: AdminConfig


class ConfigTestRequest(BaseModel):
    """Non-persistent RAG test request."""

    question: str = Field(..., min_length=1, max_length=4000)
    config: AdminConfig | None = None


class AdminStatus(BaseModel):
    """Health component status."""

    status: str
    detail: str | None = None
    model_config = ConfigDict(extra="allow")


class ProvidersResponse(BaseModel):
    """Provider/model choices exposed to the UI."""

    providers: list[dict[str, Any]]


class KBActiveInfo(BaseModel):
    """Metadata about the currently active knowledge-base PDF."""

    display_name: str = ""
    size_bytes: int = 0
    pages: int = 0
    chunks: int = 0
    uploaded_at: str | None = None
    sha256: str = ""


class KBRebuildStatus(BaseModel):
    """Current or most recent knowledge-base rebuild job state."""

    state: str = "idle"  # idle|validating|building|swapping|done|failed
    message: str = ""
    started_at: str | None = None
    finished_at: str | None = None


class KBStatusResponse(BaseModel):
    """Full knowledge-base status contract."""

    active: KBActiveInfo
    rebuild: KBRebuildStatus


class DocumentRecord(BaseModel):
    """Complete document record in the multi-source Knowledge Base repository."""

    id: str
    title: str
    content_type: str  # pdf, docx, notice, faq, web_content
    category: str
    school_campus: str
    academic_year: str
    language: str = "English"
    owner: str
    version: str = "1.0"
    status: str = "Draft"  # Draft, Review, Approved, Published, Disabled, Archived
    effective_date: str | None = None
    expiry_date: str | None = None
    file_name: str | None = None
    source_url: str | None = None
    content_text: str | None = None
    chunks_count: int = 0
    size_bytes: int = 0
    rejection_reason: str | None = None
    created_at: str
    updated_at: str
    reviewed_by: str | None = None
    reviewed_at: str | None = None


class DocumentCreateRequest(BaseModel):
    """Request payload for creating a new document or text-based content."""

    title: str = Field(..., min_length=1, max_length=200)
    content_type: str = Field(default="pdf", pattern="^(pdf|docx|notice|faq|web_content)$")
    category: str = Field(default="General", min_length=1, max_length=100)
    school_campus: str = Field(default="All Campuses", min_length=1, max_length=100)
    academic_year: str = Field(default="2026-2027", min_length=1, max_length=50)
    language: str = Field(default="English", min_length=1, max_length=50)
    owner: str = Field(default="Staff Member", min_length=1, max_length=100)
    version: str = Field(default="1.0", min_length=1, max_length=20)
    effective_date: str | None = None
    expiry_date: str | None = None
    source_url: str | None = None
    content_text: str | None = None
    submit_for_review: bool = False


class DocumentUpdateRequest(BaseModel):
    """Payload for updating document metadata or text content."""

    title: str | None = None
    category: str | None = None
    school_campus: str | None = None
    academic_year: str | None = None
    language: str | None = None
    version: str | None = None
    effective_date: str | None = None
    expiry_date: str | None = None
    content_text: str | None = None
    source_url: str | None = None


class WorkflowActionRequest(BaseModel):
    """Workflow state change request (submit, approve, reject, publish, disable, archive)."""

    action: str = Field(
        ...,
        pattern="^(submit_for_review|approve|reject|publish|disable|archive)$",
    )
    comment: str | None = None
    actor: str | None = None
