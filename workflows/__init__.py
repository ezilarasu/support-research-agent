"""
Module: workflows
Layer:  Graph (workflow subgraphs and helpers)
Owner:  workflows/

Purpose:
    Business process workflows for support-research-agent.
    Covers: approval flows, escalation chains, SLA tracking.

Public API:
    ApprovalRequest          — TypedDict for interrupt() payloads
    check_approval_required  — pure fn: is refund above threshold?
    build_approval_payload   — pure fn: builds ApprovalRequest dict
    EscalationLevel          — int constants: L1=0, L2=1, L3=2
    escalate_node            — LangGraph node: increments escalation_level
    should_escalate          — pure router fn: "escalate_node" | "respond"
    compute_sla_deadline     — pure fn: deadline from priority + created_at
    is_sla_breached          — pure fn: True if deadline passed
    sla_tracking_node        — LangGraph node: writes SLA audit event
"""
from workflows.approval import (
    ApprovalRequest,
    _extract_refund_amount,
    build_approval_payload,
    check_approval_required,
)
from workflows.escalation import (
    EscalationLevel,
    escalate_node,
    should_escalate,
)
from workflows.sla import (
    SLAConfig,
    compute_sla_deadline,
    hours_until_deadline,
    is_sla_breached,
    sla_tracking_node,
)

__all__ = [
    # approval
    "ApprovalRequest",
    "check_approval_required",
    "build_approval_payload",
    "_extract_refund_amount",
    # escalation
    "EscalationLevel",
    "escalate_node",
    "should_escalate",
    # sla
    "SLAConfig",
    "compute_sla_deadline",
    "is_sla_breached",
    "hours_until_deadline",
    "sla_tracking_node",
]
