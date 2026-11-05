"""
Module: workflows.approval
Layer:  Graph (workflow subgraph)
Owner:  workflows/

Purpose:
    Approval workflow helpers for human-in-the-loop refund decisions.

    When billing_node detects a refund above the approval threshold, it calls
    interrupt() to pause the graph and surface the approval request to a human
    operator. This module provides:
      1. Pure helper functions for detecting and building approval requests
      2. The ApprovalRequest TypedDict
      3. Constants that drive the approval threshold (settings-overridable)

    Why interrupt() over a dedicated approval subgraph?
      For a single-step approval (approve/reject/escalate), interrupt() inside
      the node is simpler than a subgraph. A subgraph is better for multi-step
      approval chains (Concept 67) where multiple reviewers must sign off in
      sequence. See Concept 61 for the subgraph pattern.

    The canonical HITL round-trip:
      ┌─ billing_node detects large refund ─────────────────────────────────┐
      │  1. interrupt(payload)   → graph pauses, saves state to SQLite      │
      │  2. caller sees next=("billing_node",) on get_state()               │
      │  3. human reviews payload, sends Command(resume="approved")         │
      │  4. interrupt() returns "approved" — billing_node continues         │
      │  5. billing_node writes approval_decision to state                  │
      └─────────────────────────────────────────────────────────────────────┘

Public API:
    ApprovalRequest            — TypedDict for the interrupt payload
    check_approval_required    — pure function: is amount above threshold?
    build_approval_payload     — pure function: builds the interrupt payload dict
    _extract_refund_amount     — pure helper: finds $ amount in ticket text

LangGraph Concepts demonstrated:
    - Interrupt (Concept 52)
    - Command(resume) (Concept 57)
    - Approval Workflows (Concept 60)
    - Approval Chains (Concept 67)
"""
from __future__ import annotations

import logging
import re
from typing import TypedDict

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_REFUND_APPROVAL_THRESHOLD: float = 500.0
"""Refund amounts above this require human approval. Configurable per deployment."""

_AMOUNT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\$\s*(\d+(?:\.\d{1,2})?)"),          # $750 or $750.00
    re.compile(r"(\d+(?:\.\d{1,2})?)\s*dollars?", re.IGNORECASE),  # 750 dollars
    re.compile(r"refund\s+(?:of\s+)?(\d+(?:\.\d{1,2})?)", re.IGNORECASE),  # refund of 750
    re.compile(r"(\d+(?:\.\d{1,2})?)\s*(?:USD|usd)"),  # 750 USD
]
"""Dollar amount patterns searched in order. First match wins."""


# ── TypedDicts ────────────────────────────────────────────────────────────────

class ApprovalRequest(TypedDict):
    """
    Structured payload passed to interrupt() for a refund approval.

    This is what the human operator sees in their approval UI or CLI prompt.
    Must contain enough context to make an informed decision without needing
    to look up the full ticket.
    """
    message: str           # Human-readable description of what needs approval
    amount: float          # Dollar amount of the refund requested
    ticket_type: str       # Always "billing" for refund approvals
    action: str            # Always "approve_refund" — identifies the approval type
    thread_id: str         # For the operator to look up the full ticket if needed
    reason: str            # Why the approval is needed (from ticket text summary)


# ── Pure Helper Functions ─────────────────────────────────────────────────────

def _extract_refund_amount(text: str) -> float | None:
    """
    Extracts the first dollar amount found in ticket text.

    Searches patterns in order: $XXX, XXX dollars, refund of XXX, XXX USD.
    Case-insensitive. Handles optional decimal places.

    Pure function — no I/O, fully unit-testable.

    Args:
        text: Raw ticket text from the customer.

    Returns:
        Float amount if found, None if no dollar amount detected.

    Examples:
        _extract_refund_amount("I need a refund of $750")    → 750.0
        _extract_refund_amount("Charge me 299.99 dollars")   → 299.99
        _extract_refund_amount("My billing is wrong")        → None
    """
    for pattern in _AMOUNT_PATTERNS:
        match = pattern.search(text)
        if match:
            try:
                return float(match.group(1).replace(",", ""))
            except (ValueError, IndexError):
                continue
    return None


def check_approval_required(amount: float | None) -> bool:
    """
    Returns True if the amount requires human approval.

    Pure function — the single source of truth for the threshold check.
    Testing this independently of billing_node keeps threshold logic
    testable without mocking the full graph.

    Args:
        amount: Dollar amount, or None if no amount detected.

    Returns:
        True if amount is not None and >= _REFUND_APPROVAL_THRESHOLD.
    """
    if amount is None:
        return False
    return amount >= _REFUND_APPROVAL_THRESHOLD


def build_approval_payload(
    amount: float,
    thread_id: str,
    reason: str = "customer refund request",
) -> ApprovalRequest:
    """
    Builds the structured payload passed to interrupt().

    This is what the human operator sees. Must be self-contained —
    the operator should not need additional context to make a decision.

    Pure function — no I/O.

    Args:
        amount:    Dollar amount of the refund.
        thread_id: Current graph thread for audit trail.
        reason:    Brief description of why the refund was requested.

    Returns:
        ApprovalRequest dict ready to pass to interrupt().

    Example:
        payload = build_approval_payload(750.0, "thread-abc-123")
        decision = interrupt(payload)   # pauses graph
    """
    return ApprovalRequest(
        message=f"Refund of ${amount:.2f} requires approval. Approve or reject?",
        amount=amount,
        ticket_type="billing",
        action="approve_refund",
        thread_id=thread_id,
        reason=reason,
    )
