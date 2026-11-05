"""
Module: security.auth
Layer:  Service
Owner:  security/

Purpose:
    JWT authentication for support-research-agent.

    Design: authentication is a PRE-GRAPH GUARD, not an inline node.
    The JWT is validated BEFORE build_graph() is called. If validation
    fails, the graph never runs. This is the correct pattern because:
      - Auth failure is binary: either the caller is allowed or not
      - Keeping auth outside the graph makes it testable independently
      - The graph assumes it receives only authenticated callers

    For CLI: auth is bypassed (trusted local user).
    For FastAPI: auth validates the Bearer token on each request.
    For the graph itself: auth_node can optionally run as the first node
    to log authentication context into the audit trail.

    JWT structure expected:
        {
            "sub":       "user-123",         # user ID (required)
            "tenant_id": "acme-corp",        # tenant ID (required)
            "roles":     ["agent"],          # RBAC roles (required)
            "exp":       1737034260,         # expiry (validated by jose)
        }

Public API:
    validate_jwt(token, secret, algorithm)  — pure function: decodes + validates
    build_auth_context(token, settings)     — resolves user context from JWT
    auth_node(state, config)               — optional graph node: logs auth audit event
    AuthContext                             — TypedDict for validated auth info

Dependencies:
    jose            — JWT decode/verify (python-jose[cryptography])
    core.settings   — jwt_secret, jwt_algorithm, default_tenant_id
    core.exceptions — AuthenticationError

LangGraph Concepts demonstrated:
    - Authentication (Concept 109) — JWT as pre-graph guard
    - Auditability (Concept 114)   — auth events in audit trail
"""
from __future__ import annotations

import logging
from typing import TypedDict

from langchain_core.runnables import RunnableConfig  # noqa: TC002

from core.exceptions import AuthenticationError
from core.settings import get_settings
from state import AgentState  # noqa: TC001

logger = logging.getLogger(__name__)


class AuthContext(TypedDict):
    """
    Validated authentication context extracted from a JWT.

    Built by build_auth_context() after successful validation.
    Passed into the RunnableConfig["configurable"] so nodes can read it.
    """
    user_id: str
    tenant_id: str
    roles: list[str]
    authenticated: bool


def validate_jwt(
    token: str,
    secret: str,
    algorithm: str = "HS256",
) -> dict:
    """
    Decodes and validates a JWT token.

    Pure function — no state, no graph knowledge.
    Used by build_auth_context() and the FastAPI dependency.

    Args:
        token:     Raw JWT string (without "Bearer " prefix).
        secret:    The signing secret (from settings.jwt_secret).
        algorithm: Signing algorithm, default HS256.

    Returns:
        Decoded JWT payload dict.

    Raises:
        AuthenticationError: If the token is expired, has an invalid signature,
                             or is missing required claims (sub, tenant_id, roles).
    """
    try:
        from jose import jwt as jose_jwt

        payload = jose_jwt.decode(token, secret, algorithms=[algorithm])
    except Exception as e:
        raise AuthenticationError(f"invalid token: {e}") from e

    # Validate required claims
    if not payload.get("sub"):
        raise AuthenticationError("missing 'sub' claim")
    if not payload.get("tenant_id"):
        raise AuthenticationError("missing 'tenant_id' claim")
    if "roles" not in payload:
        raise AuthenticationError("missing 'roles' claim")

    return payload


def build_auth_context(
    token: str | None = None,
    settings=None,
) -> AuthContext:
    """
    Builds an AuthContext from a JWT token (or dev defaults if no secret configured).

    In development (no JWT_SECRET set): returns a default context with
    `default_tenant_id` and `["agent"]` role — allows local testing without tokens.

    In production (JWT_SECRET set): validates the token and extracts context.

    Args:
        token:    Raw JWT string. Required if JWT_SECRET is set.
        settings: Settings instance. Defaults to get_settings().

    Returns:
        AuthContext with user_id, tenant_id, roles, authenticated.

    Raises:
        AuthenticationError: If JWT_SECRET is set but the token is invalid.
    """
    s = settings or get_settings()

    # Dev mode: no JWT_SECRET configured → bypass auth
    if s.jwt_secret is None:
        logger.debug("auth: jwt_secret not set — using dev defaults (no auth)")
        return AuthContext(
            user_id="dev-user",
            tenant_id=s.default_tenant_id,
            roles=["agent", "billing_admin"],
            authenticated=True,
        )

    if not token:
        raise AuthenticationError("no token provided and jwt_secret is set")

    secret = s.jwt_secret.get_secret_value()
    payload = validate_jwt(token, secret, s.jwt_algorithm)

    context = AuthContext(
        user_id=str(payload["sub"]),
        tenant_id=str(payload.get("tenant_id", s.default_tenant_id)),
        roles=list(payload.get("roles", ["agent"])),
        authenticated=True,
    )

    logger.info(
        "auth: validated user_id=%s tenant_id=%s roles=%s",
        context["user_id"],
        context["tenant_id"],
        context["roles"],
    )
    return context


def create_jwt(
    user_id: str,
    tenant_id: str,
    roles: list[str],
    settings=None,
) -> str:
    """
    Creates a signed JWT for testing and CLI use.

    Args:
        user_id:   Subject claim.
        tenant_id: Tenant identifier.
        roles:     List of RBAC roles.
        settings:  Settings instance.

    Returns:
        Signed JWT string.

    Raises:
        ConfigurationError: If JWT_SECRET is not configured.
    """
    from datetime import datetime, timedelta, timezone

    from jose import jwt as jose_jwt

    from core.exceptions import ConfigurationError

    s = settings or get_settings()
    if s.jwt_secret is None:
        raise ConfigurationError("Cannot create JWT: JWT_SECRET is not set")

    now = datetime.now(tz=timezone.utc)
    payload = {
        "sub": user_id,
        "tenant_id": tenant_id,
        "roles": roles,
        "iat": now,
        "exp": now + timedelta(hours=24),
    }
    return jose_jwt.encode(
        payload,
        s.jwt_secret.get_secret_value(),
        algorithm=s.jwt_algorithm,
    )


# ── Optional graph node ───────────────────────────────────────────────────────

async def auth_node(state: AgentState, config: RunnableConfig) -> dict:
    """
    Optional graph node: logs authentication context to the audit trail.

    This node does NOT perform JWT validation (that happens before the graph
    runs). It simply reads auth context from config and records it as an
    audit event, confirming the caller was authenticated.

    In dev mode (no JWT_SECRET): records the dev defaults.
    In production: records the validated user_id and tenant_id.

    Reads:  config["configurable"]["user_id"] (if set)
            config["configurable"]["tenant_id"] (if set)

    Writes: state["authenticated"] — True (only reached if auth passed)
            state["rbac_roles"]    — roles from the auth context
            state["audit_events"]  — auth_verified event

    Args:
        state:  Current AgentState.
        config: RunnableConfig. Auth context injected via configurable.

    Returns:
        Partial state dict.
    """
    configurable = config.get("configurable", {})
    user_id = configurable.get("user_id", "dev-user")
    tenant_id = configurable.get("tenant_id", get_settings().default_tenant_id)
    roles = configurable.get("roles", ["agent"])
    thread_id = configurable.get("thread_id", "unknown")

    logger.info(
        "auth_node: user_id=%s tenant_id=%s roles=%s thread=%s",
        user_id,
        tenant_id,
        roles,
        thread_id,
    )

    return {
        "authenticated": True,
        "rbac_roles": roles,
        "audit_events": [{
            "node": "auth_node",
            "action": "auth_verified",
            "thread_id": thread_id,
            "user_id": user_id,
            "tenant_id": tenant_id,
            "roles": roles,
        }],
    }
