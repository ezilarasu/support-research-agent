"""
Module: workflows.sla
Layer:  Graph (workflow)
Owner:  workflows/

Purpose:
    SLA (Service Level Agreement) tracking for support-research-agent.

    Every support ticket has an expected resolution time based on its priority.
    This module computes deadlines, detects breaches, and provides a node that
    writes SLA metadata to state for downstream nodes to use.

    SLA tiers:
      critical: 1 hour   — system outages, data loss
      high:     4 hours  — major feature broken
      medium:   24 hours — normal issues
      low:      72 hours — cosmetic, documentation

    Design:
      SLA logic is entirely in pure functions for testability.
      The sla_tracking_node just reads ticket_metadata and calls them.
      No LLM calls — pure datetime arithmetic.

Public API:
    SLAConfig              — TypedDict for SLA configuration
    compute_sla_deadline   — pure: deadline ISO string from priority + created_at
    is_sla_breached        — pure: True if deadline has passed
    sla_tracking_node      — LangGraph node: writes SLA info to audit_events

LangGraph Concepts demonstrated:
    - SLA Workflows (Concept 69)
    - Escalation Chains (Concept 68) — SLA breach triggers auto-escalation
    - Workflow State Machines (Concept 65) — SLA as lifecycle constraint
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import TypedDict

from langchain_core.runnables import RunnableConfig  # noqa: TC002

from state import AgentState  # noqa: TC001

logger = logging.getLogger(__name__)


# ── SLA Configuration ─────────────────────────────────────────────────────────

_SLA_HOURS: dict[str, int] = {
    "critical": 1,
    "high": 4,
    "medium": 24,
    "low": 72,
}
"""Resolution time in hours per priority level."""

_DEFAULT_PRIORITY: str = "medium"
"""Used when ticket_metadata is None or priority is missing."""


class SLAConfig(TypedDict):
    """
    SLA metadata written to audit events by sla_tracking_node.
    Downstream nodes (escalate_node, respond_node) can read this from
    audit_events to adapt behaviour based on SLA status.
    """
    priority: str
    deadline: str        # ISO 8601 datetime string
    created_at: str      # ISO 8601 datetime string
    breached: bool       # True if current time > deadline
    hours_remaining: float  # Negative if breached


# ── Pure Functions ────────────────────────────────────────────────────────────

def compute_sla_deadline(
    priority: str,
    created_at: datetime | None = None,
) -> str:
    """
    Computes the SLA deadline for a ticket.

    Pure function — no I/O, deterministic given the same inputs.
    The created_at default of "now" is only used when no timestamp is
    available (e.g., test fixtures that don't set ticket_metadata).

    Args:
        priority:   One of "critical", "high", "medium", "low".
                    Unknown values default to "medium".
        created_at: When the ticket was created. Defaults to now (UTC).

    Returns:
        ISO 8601 string representing the SLA deadline.

    Examples:
        compute_sla_deadline("critical")  → "+1 hour from now" as ISO string
        compute_sla_deadline("low")       → "+72 hours from now" as ISO string
    """
    hours = _SLA_HOURS.get(priority, _SLA_HOURS[_DEFAULT_PRIORITY])
    base = created_at or datetime.now(tz=timezone.utc)

    # Ensure timezone-aware
    if base.tzinfo is None:
        base = base.replace(tzinfo=timezone.utc)

    deadline = base + timedelta(hours=hours)
    return deadline.isoformat()


def is_sla_breached(deadline_str: str) -> bool:
    """
    Returns True if the SLA deadline has already passed.

    Pure function — compares deadline against current UTC time.

    Args:
        deadline_str: ISO 8601 deadline string from compute_sla_deadline().

    Returns:
        True if the current time is past the deadline.

    Raises:
        ValueError: If deadline_str is not a valid ISO 8601 string.
    """
    deadline = datetime.fromisoformat(deadline_str)
    # Normalise to UTC if no timezone info
    if deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=timezone.utc)
    return datetime.now(tz=timezone.utc) > deadline


def hours_until_deadline(deadline_str: str) -> float:
    """
    Returns hours remaining until the SLA deadline.

    Negative value means the SLA is already breached.
    Pure function — useful for prioritisation logic.

    Args:
        deadline_str: ISO 8601 deadline string.

    Returns:
        Float hours remaining (negative if breached).
    """
    deadline = datetime.fromisoformat(deadline_str)
    if deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=timezone.utc)
    delta = deadline - datetime.now(tz=timezone.utc)
    return delta.total_seconds() / 3600


# ── Node ──────────────────────────────────────────────────────────────────────

async def sla_tracking_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    """
    SLA tracking node — computes deadline and detects breaches.

    Reads ticket_metadata for priority and created_at. If ticket_metadata
    is None (common in tests), defaults to medium priority and now as
    created_at — still computes a valid SLA.

    SLA breach triggers an elevated audit event. Future: could trigger
    automatic escalation via should_escalate() router.

    Reads:
        state["ticket_metadata"]  — priority and created_at (optional)

    Writes:
        state["audit_events"]  — SLA computation event including breach status

    Args:
        state:  Current AgentState.
        config: RunnableConfig with thread_id.

    Returns:
        Partial state dict with audit_events.
    """
    from agents.nodes import _get_thread_id

    thread_id = _get_thread_id(config)
    metadata = state.get("ticket_metadata")

    # Extract priority and created_at from ticket metadata
    priority = _DEFAULT_PRIORITY
    created_at: datetime | None = None

    if metadata:
        priority = metadata.get("priority", _DEFAULT_PRIORITY)
        raw_created = metadata.get("created_at")
        if raw_created:
            try:
                created_at = datetime.fromisoformat(raw_created)
            except ValueError:
                logger.warning(
                    "sla_tracking_node: invalid created_at=%r, using now",
                    raw_created,
                )

    # Compute deadline and check breach
    deadline_str = compute_sla_deadline(priority, created_at)
    breached = is_sla_breached(deadline_str)
    remaining = hours_until_deadline(deadline_str)

    if breached:
        logger.warning(
            "sla_tracking_node: SLA BREACHED thread=%s priority=%s",
            thread_id,
            priority,
        )
    else:
        logger.info(
            "sla_tracking_node: thread=%s priority=%s hours_remaining=%.1f",
            thread_id,
            priority,
            remaining,
        )

    sla_event: dict = {
        "node": "sla_tracking_node",
        "action": "sla_breach_detected" if breached else "sla_computed",
        "thread_id": thread_id,
        "priority": priority,
        "deadline": deadline_str,
        "breached": breached,
        "hours_remaining": round(remaining, 2),
    }

    return {"audit_events": [sla_event]}
