"""
Module: security
Layer:  Service
Owner:  security/

Purpose:
    Security primitives for support-research-agent.
    Covers: compliance checks (PII detection), JWT auth, RBAC.

Public API:
    compliance_gate_node    — PII detection node; flags tickets with PII
    detect_pii              — pure function: returns detected PII types
    validate_jwt            — pure function: decodes + validates JWT
    build_auth_context      — resolves AuthContext from token or dev defaults
    create_jwt              — creates a signed JWT for testing
    auth_node               — optional graph node: logs auth audit event
    has_permission          — pure function: checks RBAC permission
    require_permission      — raises AuthorizationError on denial
    rbac_check_node         — optional graph node: enforces RBAC
    ROLE_PERMISSIONS        — role → set[permission] config dict
"""
from security.auth import AuthContext, auth_node, build_auth_context, create_jwt, validate_jwt
from security.compliance import compliance_gate_node, detect_pii
from security.rbac import ROLE_PERMISSIONS, has_permission, rbac_check_node, require_permission

__all__ = [
    "AuthContext",
    "auth_node",
    "build_auth_context",
    "create_jwt",
    "validate_jwt",
    "compliance_gate_node",
    "detect_pii",
    "ROLE_PERMISSIONS",
    "has_permission",
    "rbac_check_node",
    "require_permission",
]
