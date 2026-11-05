"""
Module: api.routes
Layer:  Delivery
Owner:  api/

Purpose:
    FastAPI delivery layer for support-research-agent.

    This is the OUTERMOST layer for the HTTP API — it handles HTTP I/O
    and delegates to the graph. No business logic lives here.

    Clean architecture rules (same as main.py):
      - Receives HTTP requests, validates auth, calls graph
      - Returns HTTP responses — never knows about nodes or state internals
      - All auth happens HERE before the graph is invoked
      - All graph errors are translated to HTTP status codes here

    Endpoints:
      POST /tickets         — submit a support ticket (streams response)
      GET  /health          — liveness check
      GET  /metrics/{id}    — session metrics for a thread (debug)
      POST /auth/token      — issue a JWT for testing (dev only)

    Authentication flow:
      1. Client sends: Authorization: Bearer <jwt>
      2. get_current_user() FastAPI dependency validates the token
      3. AuthContext is injected into every endpoint handler
      4. AuthContext is passed in config["configurable"] to the graph

    Streaming:
      POST /tickets returns StreamingResponse — the graph streams tokens
      via astream(stream_mode="messages") and we forward them as SSE.

Public API:
    create_app()  — factory: returns a configured FastAPI application instance

Dependencies:
    fastapi          — HTTP framework
    security.auth    — build_auth_context, validate_jwt, create_jwt
    security.rbac    — require_permission
    core.settings    — all configuration
    core.factories   — get_checkpointer
    graph            — build_graph
    observability    — configure_tracing, configure_metrics

LangGraph Concepts demonstrated:
    - Enterprise Deployment (Concept 120) — FastAPI delivery layer
    - Authentication (Concept 109)        — JWT Bearer dependency
    - Authorization (Concept 110)         — require_permission per endpoint
    - LangSmith Tracing (Concept 116)     — metadata tagging per request
"""
from __future__ import annotations

import logging
import uuid

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from core.exceptions import AuthenticationError, AuthorizationError
from core.settings import get_settings
from security.auth import AuthContext, build_auth_context
from security.rbac import require_permission

logger = logging.getLogger(__name__)


# ── Request / Response models ─────────────────────────────────────────────────

class TicketRequest(BaseModel):
    """Incoming support ticket."""
    ticket: str
    stream: bool = True


class HealthResponse(BaseModel):
    """Liveness response."""
    status: str
    provider: str
    tracing: bool


class TokenRequest(BaseModel):
    """Test JWT issuance request (dev only)."""
    user_id: str
    tenant_id: str
    roles: list[str] = ["agent"]


class TokenResponse(BaseModel):
    """Issued JWT response."""
    access_token: str
    token_type: str = "bearer"


# ── Auth dependency ───────────────────────────────────────────────────────────

async def get_current_user(
    authorization: str | None = Header(default=None),
) -> AuthContext:
    """
    FastAPI dependency: validates the Bearer JWT and returns AuthContext.

    Called automatically by FastAPI for every endpoint that declares it
    as a Depends() parameter. If validation fails, raises HTTP 401.

    In dev mode (no JWT_SECRET): accepts requests with no token and
    returns dev defaults. This allows local testing without a token.

    Args:
        authorization: The raw Authorization header value.

    Returns:
        AuthContext with user_id, tenant_id, roles.

    Raises:
        HTTPException 401: If the token is present but invalid.
    """
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()

    try:
        return build_auth_context(token=token)
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        ) from e


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """
    Factory: creates and configures the FastAPI application.

    Why a factory function instead of a module-level app object?
      - Tests can call create_app() to get a fresh instance with clean state
      - Settings and logging are configured at call time, not import time
      - Follows the same factory pattern as get_checkpointer() and get_rag_graph()

    Returns:
        Configured FastAPI application instance.
    """
    from core.logging import configure_logging
    from observability.metrics import configure_metrics
    from observability.tracing import configure_tracing

    configure_logging()
    configure_tracing()
    configure_metrics()

    settings = get_settings()

    app = FastAPI(
        title="Support Research Agent",
        description="Production multi-agent customer support system",
        version="1.0.0",
        docs_url="/docs" if settings.graph_debug else None,  # disable Swagger in prod
        redoc_url=None,
    )

    # ── Lifespan: compile graph once at startup ───────────────────────────────
    # Build the graph once and store on app.state — reused for every request.
    # This avoids per-request graph compilation (expensive).
    @app.on_event("startup")
    async def startup() -> None:
        from core.factories import get_checkpointer
        from graph import build_graph

        checkpointer = get_checkpointer()
        app.state.graph = build_graph(
            checkpointer=checkpointer,
            debug=settings.graph_debug,
        )
        logger.info("api: graph compiled and ready")

    # ── Routes ────────────────────────────────────────────────────────────────

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Liveness check. No auth required."""
        from observability.tracing import is_tracing_enabled
        return HealthResponse(
            status="ok",
            provider=settings.llm_provider,
            tracing=is_tracing_enabled(),
        )

    @app.post("/tickets")
    async def submit_ticket(
        request: TicketRequest,
        auth: AuthContext = Depends(get_current_user),
    ) -> StreamingResponse:
        """
        Submit a support ticket and stream the response.

        Requires: tickets:respond permission (role: agent or higher).
        Returns:  Server-Sent Events stream of response tokens.
        """
        # RBAC check: requires tickets:respond permission
        try:
            require_permission(
                roles=auth["roles"],
                user_id=auth["user_id"],
                action="submit_ticket",
                required_permission="tickets:respond",
            )
        except AuthorizationError as e:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e)) from e

        thread_id = str(uuid.uuid4())

        from observability.tracing import get_run_metadata

        config = {
            "configurable": {
                "thread_id": thread_id,
                "user_id": auth["user_id"],
                "tenant_id": auth["tenant_id"],
                "roles": auth["roles"],
            },
            "recursion_limit": settings.graph_recursion_limit,
            "metadata": get_run_metadata(
                tenant_id=auth["tenant_id"],
                user_id=auth["user_id"],
                thread_id=thread_id,
            ),
        }

        initial_state = _build_initial_state(
            ticket=request.ticket,
            user_id=auth["user_id"],
            tenant_id=auth["tenant_id"],
            roles=auth["roles"],
        )

        if request.stream:
            return StreamingResponse(
                _stream_tokens(app.state.graph, initial_state, config),
                media_type="text/event-stream",
            )

        # Non-streaming: collect full response
        result = await app.state.graph.ainvoke(initial_state, config=config)
        final_response = result.get("final_response", "No response generated.")
        return {"thread_id": thread_id, "response": final_response}

    @app.get("/metrics/{thread_id}")
    async def get_metrics(
        thread_id: str,
        auth: AuthContext = Depends(get_current_user),
    ) -> dict:
        """Returns session metrics. Only available in debug mode."""
        if not settings.graph_debug:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Metrics endpoint not available in production mode",
            )
        from observability.metrics import get_session_metrics
        return get_session_metrics(thread_id)

    @app.post("/auth/token", response_model=TokenResponse)
    async def issue_token(request: TokenRequest) -> TokenResponse:
        """
        Issue a test JWT. DEV ONLY — disabled when JWT_SECRET is not set.

        In production, JWTs are issued by your identity provider (Auth0, Okta, etc.).
        This endpoint exists for local development and integration testing only.
        """
        if settings.jwt_secret is None:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="Token issuance not available: JWT_SECRET not configured",
            )
        from security.auth import create_jwt
        token = create_jwt(
            user_id=request.user_id,
            tenant_id=request.tenant_id,
            roles=request.roles,
        )
        logger.info(
            "api: issued test token user_id=%s tenant=%s",
            request.user_id,
            request.tenant_id,
        )
        return TokenResponse(access_token=token)

    return app


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_initial_state(
    ticket: str,
    user_id: str = "unknown",
    tenant_id: str = "default",
    roles: list[str] | None = None,
) -> dict:
    """Builds the initial AgentState for a graph invocation from the API."""
    return {
        "messages": [HumanMessage(content=ticket)],
        "ticket_type": None,
        "ticket_metadata": None,
        "classification_confidence": 0.0,
        "thread_id": None,
        "retrieved_memories": [],
        "streaming_enabled": True,
        "active_agent": None,
        "agent_results": [],
        "query": None,
        "retrieved_docs": [],
        "retrieval_grade": None,
        "hallucination_check": None,
        "pending_approval": None,
        "approval_decision": None,
        "escalation_level": 0,
        "authenticated": True,
        "rbac_roles": roles or ["agent"],
        "audit_events": [],
        "final_response": None,
        "error": None,
    }


async def _stream_tokens(graph, initial_state: dict, config: dict):
    """
    Async generator: streams response tokens as Server-Sent Events.

    Runs two passes (same pattern as main.py._stream_response):
      Pass 1: stream_mode="updates" — collect metadata, run all nodes
      Pass 2: stream_mode="messages" — yield token chunks as SSE

    Yields:
        SSE-formatted strings: "data: <token>\n\n"
    """
    # Pass 1: run the graph
    async for _ in graph.astream(initial_state, config=config, stream_mode="updates"):
        pass  # let the graph run; we don't forward node progress via SSE

    # Pass 2: stream tokens
    async for chunk, metadata in graph.astream(
        {},
        config=config,
        stream_mode="messages",
    ):
        node = metadata.get("langgraph_node", "")
        if node == "respond" and hasattr(chunk, "content") and chunk.content:
            yield f"data: {chunk.content}\n\n"

    yield "data: [DONE]\n\n"
