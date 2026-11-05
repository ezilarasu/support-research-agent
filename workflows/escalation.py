"""
Module: workflows.escalation
Layer:  Graph (workflow)
Owner:  workflows/

Purpose:
    Escalation workflow for support-research-agent.

    When a ticket cannot be resolved at the current support level — or when
    a human operator sends Command(goto="escalate_node") during a HITL review
    — this module handles the escalation logic.

    Escalation levels:
      L1 (escalation_level=0) — Frontline agent (default)
      L2 (escalation_level=1) — Senior agent / specialist
      L3 (escalation_level=2) — Engineering / management

    Why Command(goto) instead of a normal conditional edge?
      A conditional edge is decided by the graph at runtime based on state.
      Command(goto) is decided by the HUMAN during a HITL interrupt — they
      choose to escalate rather than approve/reject. It's the human overriding
      the normal graph flow, which is exactly what Command(goto) enables.

      Normal flow:
        billing_node → interrupt() → resume("approved") → respond → END

      Escalated flow:
        billing_node → interrupt() → Command(goto="escalate_node") → respond → END

Public API:
    escalate_node              — LangGraph node: increments level, writes audit
    should_escalate            — pure router fn: checks if escalation needed
    EscalationLevel            — int constants: L1=0, L2=1, L3=2
    _build_escalation_message  — pure helper: context-aware escalation prompt

LangGraph Concepts demonstrated:
    - Command(goto) (Concept 58)
    - Agent Handoffs (Concept 59)
    - Escalation Flows (Concept 62)
    - Escalation Chains (Concept 68)
"""
from __future__ import annotations

import logging

from langchain_core.runnables import RunnableConfig  # noqa: TC002

from state import AgentState  # noqa: TC001

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

class EscalationLevel:
    """
    Escalation level constants.

    Using a class of integer constants rather than an Enum because:
      - The state field is a plain int (JSON-serialisable, no Enum overhead)
      - Comparison with >= works naturally: escalation_level >= EscalationLevel.L3
      - Adding levels later doesn't break existing comparisons
    """
    L1: int = 0   # Frontline — default, all tickets start here
    L2: int = 1   # Senior agent / specialist
    L3: int = 2   # Engineering / management — highest escalation

    MAX: int = L3  # Convenience: cap escalation at L3, never go higher


_ESCALATION_MESSAGES: dict[int, str] = {
    EscalationLevel.L1: "Ticket assigned to frontline support.",
    EscalationLevel.L2: "Escalated to senior agent — requires specialist review.",
    EscalationLevel.L3: "Critical escalation — engineering and management notified.",
}
"""Human-readable escalation context for each level."""


# ── Pure Functions ────────────────────────────────────────────────────────────

def should_escalate(state: AgentState) -> str:
    """
    Router function for the conditional edge after a handler node.

    Returns "escalate_node" if escalation is needed, "respond" otherwise.
    Used as the router_fn in GraphBuilder.conditional_edge() when the graph
    has an escalation path.

    IMPORTANT: Pure synchronous function — never async, never LLM calls.

    Reads:
        state["escalation_level"]  — set by escalate_node or HITL operator
        state["error"]             — triggers escalation on unrecoverable errors
        state["approval_decision"] — "escalated" decision also triggers this

    Returns:
        "escalate_node" or "respond"
    """
    escalation_level = state.get("escalation_level", 0)
    error = state.get("error")
    approval_decision = state.get("approval_decision")

    if approval_decision == "escalated":
        return "escalate_node"
    if error and escalation_level < EscalationLevel.MAX:
        return "escalate_node"
    if escalation_level >= EscalationLevel.L2:
        return "escalate_node"
    return "respond"


def _build_escalation_message(
    current_level: int,
    ticket_type: str | None,
    reason: str = "operator requested escalation",
) -> str:
    """
    Builds the escalation context message injected into state.

    Pure function — tells the respond_node why this ticket was escalated
    so it can adapt its response accordingly.

    Args:
        current_level: The NEW escalation level after increment.
        ticket_type:   Category of the ticket (billing/technical/etc.)
        reason:        Why escalation occurred.

    Returns:
        Human-readable escalation context string.
    """
    base = _ESCALATION_MESSAGES.get(current_level, "Escalated to support team.")
    return f"{base} Reason: {reason}. Ticket type: {ticket_type or 'unknown'}."


# ── Node ──────────────────────────────────────────────────────────────────────

async def escalate_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    """
    Escalation node — increments escalation_level and records the event.

    This node is reached via two paths:
      1. should_escalate() router returns "escalate_node" (automatic)
      2. Human operator sends Command(goto="escalate_node") during HITL

    After escalation, respond_node uses the elevated escalation_level to
    adapt its response — acknowledging the escalation and setting expectations.

    Reads:
        state["escalation_level"]  — current level (0, 1, or 2)
        state["ticket_type"]       — for audit context
        state["approval_decision"] — "escalated" if HITL-triggered

    Writes:
        state["escalation_level"]  — incremented by 1 (capped at MAX)
        state["audit_events"]      — escalation event

    Args:
        state:  Current AgentState.
        config: RunnableConfig with thread_id.

    Returns:
        Partial state dict with escalation_level and audit_events.
    """
    from agents.nodes import _get_thread_id  # avoid circular import at module level

    thread_id = _get_thread_id(config)
    current_level = state.get("escalation_level", 0)
    ticket_type = state.get("ticket_type")
    approval_decision = state.get("approval_decision")

    # Increment level, but never exceed MAX
    new_level = min(current_level + 1, EscalationLevel.MAX)

    # Determine why escalation happened for the audit trail
    if approval_decision == "escalated":
        reason = "operator escalated via HITL"
    elif state.get("error"):
        reason = f"unrecoverable error: {state.get('error', 'unknown')}"
    else:
        reason = "automatic escalation threshold reached"

    logger.warning(
        "escalate_node: thread=%s level=%d→%d reason=%s",
        thread_id,
        current_level,
        new_level,
        reason,
    )

    return {
        "escalation_level": new_level,
        "audit_events": [{
            "node": "escalate_node",
            "action": "ticket_escalated",
            "thread_id": thread_id,
            "previous_level": current_level,
            "new_level": new_level,
            "reason": reason,
            "ticket_type": ticket_type,
        }],
    }
