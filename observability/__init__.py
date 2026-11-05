"""
Module: observability
Layer:  Infrastructure
Owner:  observability/

Purpose:
    Cross-cutting observability concerns: hooks, tracing, metrics, audit.
    Decouples logging/metrics/tracing from node business logic.

Public API:
    node_with_hooks     — wraps a node with pre/post hook execution
    register_pre_hook   — registers a function called before every node
    register_post_hook  — registers a function called after every node
    configure_hooks     — registers the built-in latency + audit hooks

LangGraph Concepts demonstrated:
    - Hooks (Concept 19) — Observer pattern for node observability
"""
from observability.audit import (
    AuditEvent,
    audit_trail_node,
    create_audit_event,
    format_audit_trail,
)
from observability.hooks import (
    configure_hooks,
    node_with_hooks,
    register_post_hook,
    register_pre_hook,
)
from observability.metrics import configure_metrics, get_session_metrics, record_node_latency
from observability.tracing import configure_tracing, get_run_metadata, is_tracing_enabled

__all__ = [
    "node_with_hooks",
    "register_pre_hook",
    "register_post_hook",
    "configure_hooks",
    # audit
    "AuditEvent",
    "audit_trail_node",
    "create_audit_event",
    "format_audit_trail",
    # tracing
    "configure_tracing",
    "get_run_metadata",
    "is_tracing_enabled",
    # metrics
    "configure_metrics",
    "get_session_metrics",
    "record_node_latency",
]
