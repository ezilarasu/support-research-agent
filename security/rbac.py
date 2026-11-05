"""
Module: security.rbac
Layer:  Service
Owner:  security/

Purpose:
    Role-Based Access Control for support-research-agent.

    RBAC answers one question: "Does this user have permission to do X?"
    Roles are granted by the auth layer (JWT or dev defaults).
    Permissions are mapped from roles by this module (config-driven).

    Design: role-to-permission mapping lives in a dict constant —
    making it code (reviewable, testable, version-controlled) rather
    than database rows (opaque, deployment-dependent).

    Roles in this system:
      agent           — basic support agent: read and respond to tickets
      billing_admin   — can approve/reject refunds
      supervisor      — can escalate tickets, view all tenants
      readonly        — read-only access, no responses

    Policy node:
      `rbac_check_node` can be added to the graph as an early gate node.
      It reads roles from state (populated by auth_node) and the
      required_action from config, then fails fast on unauthorized access.

Public API:
    ROLE_PERMISSIONS           — role → set[permission] mapping (config)
    has_permission(roles, perm) — pure function: checks permission
    require_permission(roles, user_id, action, required_perm) — raises on denial
    rbac_check_node(state, config) — optional graph node: enforces RBAC

Dependencies:
    core.exceptions  — AuthorizationError
    state            — AgentState

LangGraph Concepts demonstrated:
    - Authorization (Concept 110) — AuthorizationError raised on denial
    - RBAC (Concept 111)          — role-to-permission mapping
    - Multi-Tenant Graphs (Concept 112) — tenant_id in every audit event
    - Governance (Concept 113)    — policy-as-code, not database rows
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig  # noqa: TC002

from core.exceptions import AuthorizationError
from state import AgentState  # noqa: TC001

logger = logging.getLogger(__name__)


# ── Role-to-permission mapping (policy as code) ───────────────────────────────
#
# Why a dict constant instead of a database?
#   - Reviewable: every permission change is a code review, not a DB migration
#   - Testable: unit tests import this directly — no fixtures needed
#   - Auditable: git history shows every permission change with author + reason
#   - Deployable: zero runtime DB dependency for authorization decisions
#
# Permission naming convention: "<resource>:<action>"
# e.g. "tickets:read", "refunds:approve", "tenants:list"

ROLE_PERMISSIONS: dict[str, set[str]] = {
    "agent": {
        "tickets:read",
        "tickets:respond",
        "kb:search",
        "web:search",
    },
    "billing_admin": {
        "tickets:read",
        "tickets:respond",
        "kb:search",
        "web:search",
        "refunds:approve",   # ← billing_node HITL gate checks this
        "refunds:reject",
        "billing:view",
    },
    "supervisor": {
        "tickets:read",
        "tickets:respond",
        "tickets:escalate",
        "kb:search",
        "web:search",
        "tenants:list",      # ← can view tickets across tenants
        "agents:manage",
    },
    "readonly": {
        "tickets:read",
        "kb:search",
    },
}


# ── Pure functions ────────────────────────────────────────────────────────────

def has_permission(roles: list[str], permission: str) -> bool:
    """
    Returns True if any of the given roles grants the specified permission.

    Pure function — no I/O, no state. Always use this over direct dict lookups
    so that unknown roles silently return False instead of raising KeyError.

    Args:
        roles:      List of RBAC roles assigned to the user.
        permission: Permission string to check (e.g. "refunds:approve").

    Returns:
        True if at least one role grants this permission. False otherwise.

    Examples:
        has_permission(["agent"], "tickets:respond")  → True
        has_permission(["agent"], "refunds:approve")  → False
        has_permission(["agent", "billing_admin"], "refunds:approve") → True
        has_permission(["unknown_role"], "tickets:read") → False
    """
    return any(permission in ROLE_PERMISSIONS.get(role, set()) for role in roles)


def get_permissions(roles: list[str]) -> set[str]:
    """
    Returns the union of all permissions granted by the given roles.

    Useful for debugging and audit logging — shows everything a user can do.

    Args:
        roles: List of RBAC roles.

    Returns:
        Set of all permissions the user has. Empty set for unknown roles.
    """
    permissions: set[str] = set()
    for role in roles:
        permissions |= ROLE_PERMISSIONS.get(role, set())
    return permissions


def require_permission(
    roles: list[str],
    user_id: str,
    action: str,
    required_permission: str,
) -> None:
    """
    Asserts that the user has the required permission. Raises on denial.

    Use this at the start of sensitive node logic — it's a guard, not a
    conditional. If the user doesn't have permission, execution stops.

    Args:
        roles:               The user's current roles (from state["rbac_roles"]).
        user_id:             The user ID (from config["configurable"]["user_id"]).
        action:              Human-readable action description for error message.
        required_permission: The permission string required for this action.

    Raises:
        AuthorizationError: If no role grants the required permission.

    Example:
        require_permission(
            roles=state["rbac_roles"],
            user_id=config["configurable"].get("user_id", "unknown"),
            action="approve_refund",
            required_permission="refunds:approve",
        )
    """
    if not has_permission(roles, required_permission):
        logger.warning(
            "rbac_denied: user_id=%s action=%s required=%s roles=%s",
            user_id,
            action,
            required_permission,
            roles,
        )
        raise AuthorizationError(user_id, action, required_permission)

    logger.debug(
        "rbac_allowed: user_id=%s action=%s permission=%s",
        user_id,
        action,
        required_permission,
    )


# ── Optional graph node ───────────────────────────────────────────────────────

async def rbac_check_node(
    state: AgentState,
    config: RunnableConfig,
    required_permission: str = "tickets:respond",
) -> dict:
    """
    Optional graph node: enforces RBAC at the start of a graph path.

    Add this as the first node after auth_node to block unauthorized
    users from proceeding. Sets state["error"] on failure — the graph
    should route to a response node that presents the error to the caller.

    WHY a node and not a pre-graph check?
      Pre-graph checks (in main.py / FastAPI) catch auth failures before
      the graph starts. rbac_check_node provides FINE-GRAINED control:
      different graph paths can require different permissions, and the
      check is recorded in the audit trail automatically.

    Reads:  state["rbac_roles"]  — set by auth_node
            config["configurable"]["user_id"]
            config["configurable"]["tenant_id"]

    Writes: state["audit_events"] — RBAC check result (always)
            state["error"]        — set only on authorization failure

    Args:
        state:               Current AgentState.
        config:              RunnableConfig.
        required_permission: The permission to check. Default: tickets:respond.

    Returns:
        Partial state dict. Contains "error" only on failure.
    """
    configurable = config.get("configurable", {})
    user_id = configurable.get("user_id", "unknown")
    tenant_id = configurable.get("tenant_id", "unknown")
    thread_id = configurable.get("thread_id", "unknown")
    roles = state.get("rbac_roles", [])

    allowed = has_permission(roles, required_permission)

    audit_event: dict[str, Any] = {
        "node": "rbac_check_node",
        "action": "rbac_check",
        "thread_id": thread_id,
        "user_id": user_id,
        "tenant_id": tenant_id,
        "roles": roles,
        "permission_checked": required_permission,
        "allowed": allowed,
    }

    if allowed:
        logger.info(
            "rbac_check_node: ALLOWED user_id=%s tenant=%s permission=%s",
            user_id,
            tenant_id,
            required_permission,
        )
        return {"audit_events": [audit_event]}

    logger.warning(
        "rbac_check_node: DENIED user_id=%s tenant=%s permission=%s roles=%s",
        user_id,
        tenant_id,
        required_permission,
        roles,
    )
    error_msg = (
        f"Access denied: user '{user_id}' requires permission '{required_permission}'"
    )
    return {
        "error": error_msg,
        "audit_events": [audit_event],
    }
