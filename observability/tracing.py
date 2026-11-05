"""
Module: observability.tracing
Layer:  Service (observability)
Owner:  observability/

Purpose:
    LangSmith tracing setup for support-research-agent.

    LangSmith traces every LLM call, tool invocation, and graph node
    execution. In production, this is how you debug hallucinations,
    measure latency, and audit what prompts were sent to the LLM.

    Setup: configure_tracing() is called ONCE at startup in main.py,
    before any graph runs. It sets the required environment variables
    that LangChain/LangGraph read automatically.

    Tracing is OPTIONAL — the app runs normally without a LangSmith key.
    When tracing is disabled, all functions are no-ops.

    Tenant tagging:
      Every run is tagged with tenant_id so traces from different
      enterprise customers are filterable in the LangSmith UI.
      This is the multi-tenancy observability pattern.

Public API:
    configure_tracing(settings)      — call once at app startup
    get_run_metadata(tenant_id, ...) — builds per-run metadata dict
    is_tracing_enabled()             — returns bool

Dependencies:
    core.settings  — langsmith_api_key, langchain_tracing_v2, langchain_project

LangGraph Concepts demonstrated:
    - LangSmith Tracing (Concept 116) — tagged per tenant
    - Graph Observability (Concept 117) — startup configuration
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def configure_tracing(settings=None) -> None:
    """
    Configure LangSmith tracing for the current process.

    Sets the environment variables LangChain reads automatically:
      LANGCHAIN_TRACING_V2=true
      LANGCHAIN_API_KEY=<key>
      LANGCHAIN_PROJECT=<project>

    Call this ONCE at app startup, before any graph or LLM call.
    All subsequent LangChain calls in this process will be traced.

    No-op if LANGSMITH_API_KEY is not configured (tracing stays off).

    Args:
        settings: Settings instance. Defaults to get_settings().
    """
    from core.settings import get_settings
    s = settings or get_settings()

    if not s.langsmith_api_key:
        logger.debug("tracing: LANGSMITH_API_KEY not set — tracing disabled")
        return

    if not s.langchain_tracing_v2:
        logger.debug("tracing: LANGCHAIN_TRACING_V2=false — tracing disabled")
        return

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = s.langsmith_api_key.get_secret_value()
    os.environ["LANGCHAIN_PROJECT"] = s.langchain_project

    logger.info(
        "tracing: enabled project=%s",
        s.langchain_project,
    )


def get_run_metadata(
    tenant_id: str = "default",
    user_id: str = "unknown",
    ticket_type: str | None = None,
    thread_id: str | None = None,
) -> dict:
    """
    Builds per-run metadata dict for LangSmith tagging.

    Pass this as `metadata` in the RunnableConfig to tag traces with
    tenant, user, and ticket context. The LangSmith UI lets you filter
    and group traces by any metadata field.

    Args:
        tenant_id:   Tenant identifier for multi-tenant filtering.
        user_id:     Authenticated user making the request.
        ticket_type: Classification result (billing/technical/general/research).
        thread_id:   LangGraph thread ID for correlation.

    Returns:
        Dict ready to pass as config["metadata"].

    Example:
        config = {
            "configurable": {"thread_id": session_id},
            "metadata": get_run_metadata(tenant_id="acme", user_id="u-123"),
        }
    """
    metadata: dict = {
        "tenant_id": tenant_id,
        "user_id": user_id,
        "app": "support-research-agent",
    }
    if ticket_type:
        metadata["ticket_type"] = ticket_type
    if thread_id:
        metadata["thread_id"] = thread_id
    return metadata


def is_tracing_enabled() -> bool:
    """
    Returns True if LangSmith tracing is currently active.

    Checks the process environment — accurate after configure_tracing() runs.

    Returns:
        True if LANGCHAIN_TRACING_V2=true in the current environment.
    """
    return os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true"
