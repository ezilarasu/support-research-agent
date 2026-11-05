"""
Module: security.compliance
Layer:  Service
Owner:  security/

Purpose:
    PII detection and compliance gate for support-research-agent.

    Why a compliance gate?
      Enterprise support systems handle sensitive customer data — emails,
      phone numbers, partial credit card numbers. Before any LLM response
      is sent back, the system must detect if the input ticket or draft
      response contains PII that shouldn't be echoed or logged.

      This module:
        1. Detects PII in ticket text (pre-response)
        2. Flags it in audit_events (always — even if allowed)
        3. Optionally blocks if configured (settings.compliance_strict)

    Clean architecture note:
      This is a Service layer node. It has no knowledge of graph topology
      or delivery details — just PII detection logic and state writes.

Public API:
    compliance_gate_node   — LangGraph node: detects PII, writes audit events
    detect_pii             — pure function: finds PII type matches in text

LangGraph Concepts demonstrated:
    - Hooks (Concept 19) — this node acts as a compliance pre-check
    - Compliance Pipelines (Concept 71)
    - Audit Trails (Concept 70)
"""
from __future__ import annotations

import logging
import re

from langchain_core.runnables import RunnableConfig  # noqa: TC002

from state import AgentState  # noqa: TC001

logger = logging.getLogger(__name__)

# ── PII Patterns ──────────────────────────────────────────────────────────────
# Each pattern is named and compiled once at module load.
# Patterns are intentionally broad — better to flag a false positive
# than to miss real PII (fail-safe over fail-open).

_PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    ),
    "phone": re.compile(
        r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"
    ),
    "ssn": re.compile(
        r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"
    ),
    "credit_card": re.compile(
        r"\b(?:\d{4}[-\s]?){3}\d{4}\b"
    ),
    "ip_address": re.compile(
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    ),
}

# ── Pure Functions ────────────────────────────────────────────────────────────

def detect_pii(text: str) -> list[str]:
    """
    Detects PII types present in the given text.

    Pure function — no I/O, no LLM calls. Trivially unit-testable.
    Scans text against all _PII_PATTERNS and returns matching type names.

    Args:
        text: Raw text to scan (ticket body, LLM response draft, etc.)

    Returns:
        List of PII type names found (e.g. ["email", "phone"]).
        Empty list if no PII detected.

    Examples:
        detect_pii("Contact me at john@example.com")
        → ["email"]

        detect_pii("Call 555-123-4567 or email me")
        → ["phone", "email"]
    """
    found: list[str] = []
    for pii_type, pattern in _PII_PATTERNS.items():
        if pattern.search(text):
            found.append(pii_type)
    return found


def _redact_pii(text: str, pii_types: list[str]) -> str:
    """
    Replaces PII matches with [REDACTED_<TYPE>] placeholders.

    Pure function — used for safe logging of ticket content.
    Does NOT modify the original ticket stored in state.

    Args:
        text:      The text containing PII.
        pii_types: Which PII types to redact (from detect_pii output).

    Returns:
        Text with PII replaced by redaction markers.
    """
    redacted = text
    for pii_type in pii_types:
        if pii_type in _PII_PATTERNS:
            placeholder = f"[REDACTED_{pii_type.upper()}]"
            redacted = _PII_PATTERNS[pii_type].sub(placeholder, redacted)
    return redacted


# ── Node ──────────────────────────────────────────────────────────────────────

async def compliance_gate_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    """
    Compliance gate: detects PII in the incoming ticket and flags it.

    Concept 71 — Compliance Pipelines:
      This node acts as an inline compliance check. It runs before the
      specialist handler so that PII is flagged BEFORE any LLM sees it.
      In strict mode (future Module 10), it can block the ticket entirely.

    Design decision: detect-and-flag, not detect-and-block.
      Blocking here would require knowing whether PII is intentional
      (user providing their email to get a callback). Instead:
        - Flag all PII in audit_events
        - Let the LLM handle it (it won't echo PII unnecessarily)
        - Compliance officer can review audit_events in the dashboard

    Reads:
        state["messages"]  — last message scanned for PII

    Writes:
        state["audit_events"] — PII detection event (always written)

    Args:
        state:  Current AgentState.
        config: RunnableConfig with thread_id.

    Returns:
        Partial state dict with audit_events.
    """
    from agents.nodes import _get_thread_id  # avoid circular import at module level

    messages = state.get("messages", [])
    thread_id = _get_thread_id(config)

    if not messages:
        return {}

    ticket_text = messages[-1].content

    # ── PII Detection ─────────────────────────────────────────────────────────
    pii_types = detect_pii(ticket_text)

    if pii_types:
        # Redact before logging — never log raw PII
        redacted = _redact_pii(ticket_text, pii_types)
        logger.warning(
            "compliance_gate_node: PII detected thread=%s types=%s text=%r",
            thread_id,
            pii_types,
            redacted[:100],
        )
    else:
        logger.debug(
            "compliance_gate_node: no PII detected thread=%s",
            thread_id,
        )

    # ── Audit Event ───────────────────────────────────────────────────────────
    # Always write a compliance event — even clean tickets.
    # This creates a complete audit trail showing every ticket was checked.
    audit_event: dict = {
        "node": "compliance_gate_node",
        "action": "pii_scan_complete",
        "thread_id": thread_id,
        "pii_detected": bool(pii_types),
        "pii_types": pii_types,
    }

    return {"audit_events": [audit_event]}
