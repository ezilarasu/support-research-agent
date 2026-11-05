"""
Module: observability.audit
Layer:  Service (observability)
Owner:  observability/

Purpose:
    Immutable audit trail helpers for support-research-agent.

    Every sensitive action in the system — approvals, escalations, memory
    writes, PII detections — appends to state["audit_events"]. This module
    provides the factory function and formatting utilities.

    Why append-only?
      The audit_events field uses operator.add as its reducer, which means
      it only ever accumulates entries. No node can overwrite or delete prior
      events. This is a fundamental compliance property — audit trails must
      be tamper-evident within a session.

    Relationship to checkpointer:
      The checkpointer (SQLite/Postgres) persists the full state including
      audit_events after every node. This means the audit trail is durable
      even if the process crashes mid-run.

Public API:
    AuditEvent             — TypedDict for a single audit event
    create_audit_event     — factory: builds a typed AuditEvent dict
    format_audit_trail     — pure fn: human-readable audit trail string
    audit_trail_node       — LangGraph node: adds a summary audit event

LangGraph Concepts demonstrated:
    - Audit Trails (Concept 70)
    - operator.add reducer (Concept 11) — audit_events accumulates
    - Durable Execution (Concept 33) — events survive checkpoint replay
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, TypedDict

from langchain_core.runnables import RunnableConfig  # noqa: TC002

from state import AgentState  # noqa: TC001

logger = logging.getLogger(__name__)


# ── TypedDict ─────────────────────────────────────────────────────────────────

class AuditEvent(TypedDict, total=False):
    """
    A single audit trail entry.

    Fields marked with total=False are optional — not all events have all
    fields. Required fields (node, action, thread_id, timestamp) are always
    present because create_audit_event() always sets them.
    """
    node: str           # Which node created this event
    action: str         # What happened: "ticket_received", "refund_approved", etc.
    thread_id: str      # Which session this event belongs to
    timestamp: str      # ISO 8601 UTC timestamp (always set by create_audit_event)
    # Optional context fields (present only when relevant):
    amount: float       # For financial events
    decision: str       # For approval events
    pii_types: list     # For compliance events
    level: int          # For escalation events
    user_id: str        # For user-attributed events
    details: dict       # Catch-all for event-specific data


# ── Pure Functions ────────────────────────────────────────────────────────────

def create_audit_event(
    node: str,
    action: str,
    thread_id: str,
    **extra: Any,
) -> AuditEvent:
    """
    Factory: creates a typed AuditEvent with a UTC timestamp.

    Always sets node, action, thread_id, and timestamp.
    Additional keyword args are merged as optional fields.

    Pure function — no I/O, no side effects.

    Args:
        node:      Name of the LangGraph node creating this event.
        action:    Verb describing what happened (snake_case).
        thread_id: Session identifier for grouping related events.
        **extra:   Optional additional fields merged into the event.

    Returns:
        AuditEvent dict ready to append to state["audit_events"].

    Examples:
        create_audit_event("billing_node", "refund_approved", "t-001",
                          amount=750.0, decision="approved")
        → {"node": "billing_node", "action": "refund_approved",
           "thread_id": "t-001", "timestamp": "2026-...", "amount": 750.0, ...}
    """
    event: AuditEvent = {
        "node": node,
        "action": action,
        "thread_id": thread_id,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        **extra,  # type: ignore[misc]
    }
    return event


def format_audit_trail(events: list[dict]) -> str:
    """
    Formats audit_events into a human-readable string for display.

    Pure function — used by CLI and logging. No I/O.

    Args:
        events: List of AuditEvent dicts from state["audit_events"].

    Returns:
        Multi-line string, one event per line.
        Empty string if events is empty.

    Example output:
        [billing_node] refund_approved — amount=750.0 decision=approved
        [memory_write_node] memory_written — user_id=user-123
    """
    if not events:
        return ""

    lines: list[str] = []
    for event in events:
        node = event.get("node", "unknown")
        action = event.get("action", "unknown")
        # Format optional fields as key=value pairs
        extras = {
            k: v for k, v in event.items()
            if k not in {"node", "action", "thread_id", "timestamp"}
        }
        extra_str = " ".join(f"{k}={v}" for k, v in extras.items()) if extras else ""
        line = f"[{node}] {action}"
        if extra_str:
            line += f" — {extra_str}"
        lines.append(line)

    return "\n".join(lines)


# ── Node ──────────────────────────────────────────────────────────────────────

async def audit_trail_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    """
    Audit trail summary node — writes a session-end summary event.

    This node is typically placed at the END of a graph (or subgraph)
    to record that the session completed successfully. The running audit
    trail in state["audit_events"] is already complete — this adds a
    final "session_completed" event with summary statistics.

    Reads:
        state["audit_events"]     — existing events to summarise
        state["final_response"]   — whether a response was generated
        state["escalation_level"] — for summary context

    Writes:
        state["audit_events"]  — appends one session summary event

    Args:
        state:  Current AgentState.
        config: RunnableConfig with thread_id.

    Returns:
        Partial state dict with one new audit event appended.
    """
    from agents.nodes import _get_thread_id

    thread_id = _get_thread_id(config)
    existing_events = state.get("audit_events", [])
    final_response = state.get("final_response")
    escalation_level = state.get("escalation_level", 0)

    summary_event = create_audit_event(
        node="audit_trail_node",
        action="session_completed",
        thread_id=thread_id,
        total_events=len(existing_events),
        response_generated=final_response is not None,
        escalation_level=escalation_level,
        ticket_type=state.get("ticket_type"),
    )

    logger.info(
        "audit_trail_node: session_complete thread=%s events=%d escalation=%d",
        thread_id,
        len(existing_events) + 1,
        escalation_level,
    )

    return {"audit_events": [summary_event]}
