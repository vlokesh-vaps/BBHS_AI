"""FastAPI routes for the internal AI Control Center."""

import os
import time
import secrets
import hashlib
import hmac
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from langchain_groq import ChatGroq

from src.admin.config_service import load_config, update_config, validate_config
from src.admin.health_service import check_health
from src.admin.provider_service import supported_providers
from src.admin.schemas import (
    AdminConfig,
    AdminConfigUpdate,
    ConfigTestRequest,
    ConfigValidationRequest,
    ProvidersResponse,
)
from src.config import (
    ADMIN_COOKIE_SECURE,
    ADMIN_PASSWORD,
    ADMIN_SESSION_TTL_SECONDS,
    ADMIN_TOKEN,
    ADMIN_USERNAME,
)
from src.core.rag import answer_question
from src.core.retrievers import create_ensemble_retriever
from src.utils.logging import log_error, log_step

basic_security = HTTPBasic(auto_error=False)
SESSION_COOKIE_NAME = "admin_session"


def _session_secret() -> str:
    """Use the configured credential material to sign the short-lived session."""
    return ADMIN_PASSWORD or ADMIN_TOKEN


def _session_signature(timestamp: str) -> str:
    """Create a tamper-resistant signature for a session timestamp."""
    return hmac.new(
        _session_secret().encode("utf-8"), timestamp.encode("utf-8"), hashlib.sha256
    ).hexdigest()


def _valid_session(value: str | None) -> bool:
    """Validate the signed session cookie and fixed login lifetime."""
    if not value or not _session_secret():
        return False
    try:
        timestamp, signature = value.split(".", maxsplit=1)
        issued_at = int(timestamp)
    except (ValueError, AttributeError):
        return False
    expected_signature = _session_signature(timestamp)
    age = int(time.time()) - issued_at
    return (
        0 <= age < ADMIN_SESSION_TTL_SECONDS
        and secrets.compare_digest(signature, expected_signature)
    )


def require_admin_access(
    request: Request,
    response: Response,
    credentials: HTTPBasicCredentials | None = Depends(basic_security),
    x_admin_token: str | None = Header(default=None),
) -> None:
    """Protect admin routes with a fixed five-minute signed login session."""
    existing_session = request.cookies.get(SESSION_COOKIE_NAME)
    if existing_session:
        if _valid_session(existing_session):
            return
        # Continue to the configured credential check so a browser carrying
        # an expired cookie can transparently establish a fresh session.
        # Invalid cookies without valid credentials still receive 401 below.

    if ADMIN_USERNAME and ADMIN_PASSWORD:
        valid_user = secrets.compare_digest(credentials.username, ADMIN_USERNAME) if credentials else False
        valid_password = secrets.compare_digest(credentials.password, ADMIN_PASSWORD) if credentials else False
        if not (valid_user and valid_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Admin username or password is incorrect",
                headers={"WWW-Authenticate": "Basic"},
            )
    elif ADMIN_TOKEN and not secrets.compare_digest(x_admin_token or "", ADMIN_TOKEN):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid admin token")

    if _session_secret():
        issued_at = str(int(time.time()))
        response.set_cookie(
            SESSION_COOKIE_NAME,
            f"{issued_at}.{_session_signature(issued_at)}",
            max_age=ADMIN_SESSION_TTL_SECONDS,
            httponly=True,
            secure=ADMIN_COOKIE_SECURE,
            samesite="lax",
        )


router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(require_admin_access)],
)


def _reload_runtime(app: Any, config: AdminConfig) -> None:
    """Apply validated runtime settings without changing the public chat contract."""
    if (
        getattr(app.state, "vector_db", None) is not None
        and getattr(app.state, "bm25_index", None) is not None
    ):
        app.state.ensemble_retriever = create_ensemble_retriever(
            app.state.vector_db,
            app.state.bm25_index,
            vector_top_k=config.rag.vector_top_k,
            bm25_top_k=config.rag.bm25_top_k,
            final_top_k=config.rag.final_top_k,
            similarity_threshold=max(
                config.rag.similarity_threshold,
                config.guardrails.minimum_retrieval_score,
            ),
        )
    if config.models.primary_provider == "groq" and config.models.primary_enabled:
        app.state.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name=config.models.primary_model,
            temperature=config.models.temperature,
            max_tokens=config.models.max_tokens,
            timeout=config.models.request_timeout,
        )


@router.get("/config", response_model=AdminConfig)
def get_admin_config() -> AdminConfig:
    """Return validated configuration without secrets."""
    try:
        return load_config()
    except ValueError as exc:
        raise HTTPException(status_code=500, detail="Admin configuration is invalid") from exc


@router.put("/config", response_model=AdminConfig)
def put_admin_config(payload: AdminConfigUpdate, request: Request) -> AdminConfig:
    """Validate, atomically save, and apply a partial configuration update."""
    try:
        config = update_config(payload)
        _reload_runtime(request.app, config)
        log_step("ADMIN_CONFIG_UPDATED", version=config.version)
        return config
    except (OSError, ValueError) as exc:
        log_error("ADMIN_CONFIG_UPDATE_FAILED", exc)
        raise HTTPException(status_code=422, detail="Configuration validation failed") from exc


@router.post("/config/validate", response_model=AdminConfig)
def validate_admin_config(payload: ConfigValidationRequest) -> AdminConfig:
    """Validate a proposed complete configuration without persisting it."""
    try:
        return validate_config(payload.config)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Configuration validation failed") from exc


@router.post("/config/test")
def test_admin_config(payload: ConfigTestRequest, request: Request) -> dict[str, object]:
    """Execute a real retrieval/LLM test in memory without persisting settings."""
    started = time.perf_counter()
    try:
        config = validate_config(payload.config) if payload.config else load_config()
        vector_db = getattr(request.app.state, "vector_db", None)
        bm25_index = getattr(request.app.state, "bm25_index", None)
        if vector_db is None or bm25_index is None:
            raise HTTPException(
                status_code=503,
                detail="The knowledge base is not loaded. Upload and rebuild a PDF first.",
            )
        ensemble = create_ensemble_retriever(
            vector_db,
            bm25_index,
            vector_top_k=config.rag.vector_top_k,
            bm25_top_k=config.rag.bm25_top_k,
            final_top_k=config.rag.final_top_k,
            similarity_threshold=max(
                config.rag.similarity_threshold,
                config.guardrails.minimum_retrieval_score,
            ),
        )
        docs = ensemble.invoke(payload.question)
        answer = answer_question(
            vector_db=vector_db,
            bm25_index=bm25_index,
            llm=request.app.state.llm,
            cache={},
            chat_history=[],
            question=payload.question,
            ensemble_retriever=ensemble,
        )
        return {
            "test_configuration": True,
            "not_applied_to_production": True,
            "provider": config.models.primary_provider,
            "model": config.models.primary_model,
            "retrieved_chunks": len(docs),
            "retrieved_pages": [doc.metadata.get("page", "unknown") for doc in docs],
            "response": answer,
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }
    except HTTPException:
        raise
    except Exception as exc:
        log_error("ADMIN_CONFIG_TEST_FAILED", exc)
        raise HTTPException(status_code=503, detail="Configuration test could not be completed") from exc


@router.get("/providers", response_model=ProvidersResponse)
def get_providers() -> ProvidersResponse:
    """Return backend-supported providers and model identifiers."""
    return ProvidersResponse(providers=supported_providers())


@router.get("/health")
def get_admin_health(request: Request) -> dict[str, object]:
    """Return real, failure-tolerant component health information."""
    try:
        config = load_config()
    except ValueError:
        config = None
    return check_health(request.app.state, config)
