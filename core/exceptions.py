"""
Module: core.exceptions
Layer:  Infrastructure
Owner:  core/

Purpose:
    Custom exception hierarchy for support-research-agent.

    Why a custom hierarchy instead of bare ValueError / RuntimeError?
      - Callers can catch specific exception types (never bare `except Exception`)
      - Each exception carries structured context — provider name, user ID, action
      - The hierarchy mirrors clean architecture layers:
          SupportAgentError  ← base (never raise directly)
            ConfigurationError    ← Infrastructure: bad settings at startup
            ProviderError         ← Infrastructure: LLM/API call failure
            RetrievalError        ← Service: RAG pipeline failure
            AuthenticationError   ← Service: JWT invalid / expired
            AuthorizationError    ← Service: user lacks required role
            ValidationError       ← Service: node input validation failure

    Three-layer error handling (applied in every node):
      Layer 1: @with_retry — handles transient errors (rate limits, timeouts)
      Layer 2: catch specific SupportAgentError subclasses — handle expected failures
      Layer 3: let unexpected errors propagate — don't swallow unknown errors

Public API:
    SupportAgentError    — base class (never raise directly)
    ConfigurationError   — missing/invalid settings at startup
    ProviderError        — LLM provider failure after retries
    RetrievalError       — RAG retrieval failure
    AuthenticationError  — JWT validation failure
    AuthorizationError   — RBAC permission denied
    ValidationError      — node input validation failure

LangGraph Concepts demonstrated:
    - Authentication (Concept 109) — AuthenticationError
    - Authorization (Concept 110)  — AuthorizationError
"""
from __future__ import annotations


class SupportAgentError(Exception):
    """
    Base exception for all application errors.

    Never raise this directly — raise a specific subclass.
    Catching SupportAgentError catches all application errors;
    catching a subclass catches only that category.
    """


class ConfigurationError(SupportAgentError):
    """
    Raised when settings are missing or invalid at startup.

    This should crash the process immediately — running with bad config
    is worse than not running at all.

    Example:
        if not settings.groq_api_key:
            raise ConfigurationError("GROQ_API_KEY is required")
    """


class ProviderError(SupportAgentError):
    """
    Raised when an LLM provider call fails after all retry attempts.

    Carries structured context: which provider, which model, the root cause.
    Nodes catch this and return an error signal for conditional routing.

    Args:
        provider: Provider name (e.g. "groq", "huggingface").
        model:    Model ID that failed.
        cause:    The underlying exception from the provider SDK.

    Example:
        except RetryError as e:
            raise ProviderError("groq", settings.groq_model_router, e) from e
    """

    def __init__(self, provider: str, model: str, cause: Exception) -> None:
        super().__init__(
            f"Provider '{provider}' model '{model}' failed after retries: {cause}"
        )
        self.provider = provider
        self.model = model
        self.cause = cause


class RetrievalError(SupportAgentError):
    """
    Raised when RAG retrieval fails for reasons beyond a missing index.

    Distinct from returning empty results — RetrievalError means the
    retrieval system itself is broken (FAISS index corrupted, embedding
    model unavailable, etc.).

    Example:
        if not store.is_loaded:
            raise RetrievalError("FAISS index not loaded — run python -m rag.ingest")
    """


class AuthenticationError(SupportAgentError):
    """
    Raised when JWT validation fails.

    Callers should return HTTP 401 (FastAPI) or exit the CLI session.
    Never log the raw token — only log the failure reason.

    Args:
        reason: Human-readable explanation (e.g. "token expired", "invalid signature").

    Example:
        raise AuthenticationError("token expired")
    """

    def __init__(self, reason: str) -> None:
        super().__init__(f"Authentication failed: {reason}")
        self.reason = reason


class AuthorizationError(SupportAgentError):
    """
    Raised when an authenticated user attempts an action they are not permitted.

    Carries the user ID, the attempted action, and the required role.
    Return HTTP 403 (FastAPI) or log and block in the graph node.

    Args:
        user_id:       The user attempting the action.
        action:        The specific action being blocked (e.g. "approve_refund").
        required_role: The role that would allow this action.

    Example:
        raise AuthorizationError("user-123", "approve_refund", "billing_admin")
    """

    def __init__(self, user_id: str, action: str, required_role: str) -> None:
        super().__init__(
            f"User '{user_id}' lacks role '{required_role}' "
            f"required for action '{action}'"
        )
        self.user_id = user_id
        self.action = action
        self.required_role = required_role


class ValidationError(SupportAgentError):
    """
    Raised when a node receives invalid or missing input.

    Use this instead of plain ValueError inside nodes so callers can
    distinguish application validation failures from Python type errors.

    Args:
        field:   The state field or parameter that failed validation.
        reason:  What the validation failure was.

    Example:
        if not state.get("messages"):
            raise ValidationError("messages", "list is empty")
    """

    def __init__(self, field: str, reason: str) -> None:
        super().__init__(f"Validation failed for '{field}': {reason}")
        self.field = field
        self.reason = reason
