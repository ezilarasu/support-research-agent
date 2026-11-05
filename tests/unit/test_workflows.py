"""
Unit tests for Module 7 workflow modules:
  - workflows/escalation.py  — EscalationLevel, escalate_node, should_escalate
  - workflows/sla.py         — compute_sla_deadline, is_sla_breached, sla_tracking_node
  - observability/audit.py   — create_audit_event, format_audit_trail, audit_trail_node
  - security/compliance.py   — detect_pii, compliance_gate_node
  - state.py                 — TicketStatus enum

Run:
    .venv/bin/python -m pytest tests/unit/test_workflows.py -v
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from langchain_core.messages import HumanMessage

# ══════════════════════════════════════════════════════════════════════════════
# TestTicketStatus
# ══════════════════════════════════════════════════════════════════════════════

class TestTicketStatus:
    """TicketStatus enum — Concept 65: Workflow State Machines."""

    def test_ticket_status_values_are_strings(self):
        """str-Enum means values serialise as plain strings (JSON-safe)."""
        from state import TicketStatus
        assert TicketStatus.PENDING == "pending"
        assert TicketStatus.AWAITING_APPROVAL == "awaiting_approval"
        assert TicketStatus.RESOLVED == "resolved"

    def test_ticket_status_comparison(self):
        """str-Enum compares equal to plain strings."""
        from state import TicketStatus
        assert TicketStatus.APPROVED == "approved"
        assert TicketStatus.ESCALATED != "resolved"

    def test_all_expected_statuses_defined(self):
        from state import TicketStatus
        expected = {
            "pending", "in_progress", "awaiting_approval",
            "approved", "rejected", "escalated", "resolved",
        }
        actual = {s.value for s in TicketStatus}
        assert actual == expected


# ══════════════════════════════════════════════════════════════════════════════
# TestEscalationLevel
# ══════════════════════════════════════════════════════════════════════════════

class TestEscalationLevel:
    """EscalationLevel int constants."""

    def test_levels_are_ints(self):
        from workflows.escalation import EscalationLevel
        assert EscalationLevel.L1 == 0
        assert EscalationLevel.L2 == 1
        assert EscalationLevel.L3 == 2

    def test_max_equals_l3(self):
        from workflows.escalation import EscalationLevel
        assert EscalationLevel.MAX == EscalationLevel.L3

    def test_comparison_works(self):
        """integer >= comparison works naturally."""
        from workflows.escalation import EscalationLevel
        assert EscalationLevel.L3 >= EscalationLevel.L2
        assert not (EscalationLevel.L1 >= EscalationLevel.L2)


# ══════════════════════════════════════════════════════════════════════════════
# TestShouldEscalate
# ══════════════════════════════════════════════════════════════════════════════

class TestShouldEscalate:
    """should_escalate() — pure router function for escalation conditional edge."""

    def _make_state(self, escalation_level=0, error=None, approval_decision=None):
        return {
            "escalation_level": escalation_level,
            "error": error,
            "approval_decision": approval_decision,
        }

    def test_returns_respond_at_l1(self):
        from workflows.escalation import should_escalate
        state = self._make_state(escalation_level=0)
        assert should_escalate(state) == "respond"

    def test_returns_escalate_node_at_l2(self):
        from workflows.escalation import should_escalate
        state = self._make_state(escalation_level=1)
        assert should_escalate(state) == "escalate_node"

    def test_returns_escalate_node_when_error(self):
        from workflows.escalation import should_escalate
        state = self._make_state(error="LLM failed after retries")
        assert should_escalate(state) == "escalate_node"

    def test_returns_escalate_node_when_decision_escalated(self):
        from workflows.escalation import should_escalate
        state = self._make_state(approval_decision="escalated")
        assert should_escalate(state) == "escalate_node"

    def test_returns_respond_when_approved(self):
        """approved decision does not trigger should_escalate."""
        from workflows.escalation import should_escalate
        state = self._make_state(approval_decision="approved")
        assert should_escalate(state) == "respond"


# ══════════════════════════════════════════════════════════════════════════════
# TestEscalateNode
# ══════════════════════════════════════════════════════════════════════════════

class TestEscalateNode:
    """escalate_node — increments escalation_level and writes audit event."""

    @pytest.mark.asyncio
    async def test_increments_level(self, base_state, runnable_config):
        from workflows.escalation import escalate_node
        state = dict(base_state)
        state["escalation_level"] = 0

        result = await escalate_node(state, runnable_config)
        assert result["escalation_level"] == 1

    @pytest.mark.asyncio
    async def test_caps_at_max(self, base_state, runnable_config):
        """escalation_level never exceeds EscalationLevel.MAX (2)."""
        from workflows.escalation import EscalationLevel, escalate_node
        state = dict(base_state)
        state["escalation_level"] = EscalationLevel.MAX  # already at max

        result = await escalate_node(state, runnable_config)
        assert result["escalation_level"] == EscalationLevel.MAX

    @pytest.mark.asyncio
    async def test_writes_audit_event(self, base_state, runnable_config):
        from workflows.escalation import escalate_node
        state = dict(base_state)
        state["escalation_level"] = 0

        result = await escalate_node(state, runnable_config)

        assert "audit_events" in result
        assert len(result["audit_events"]) == 1
        event = result["audit_events"][0]
        assert event["node"] == "escalate_node"
        assert event["action"] == "ticket_escalated"
        assert event["previous_level"] == 0
        assert event["new_level"] == 1

    @pytest.mark.asyncio
    async def test_hitl_escalation_reason(self, base_state, runnable_config):
        """When approval_decision='escalated', reason reflects HITL escalation."""
        from workflows.escalation import escalate_node
        state = dict(base_state)
        state["escalation_level"] = 0
        state["approval_decision"] = "escalated"

        result = await escalate_node(state, runnable_config)
        event = result["audit_events"][0]
        assert "operator" in event["reason"].lower() or "hitl" in event["reason"].lower()

    @pytest.mark.asyncio
    async def test_returns_partial_dict(self, base_state, runnable_config):
        from workflows.escalation import escalate_node
        state = dict(base_state)
        result = await escalate_node(state, runnable_config)

        assert "messages" not in result
        assert "ticket_type" not in result


# ══════════════════════════════════════════════════════════════════════════════
# TestSLAFunctions
# ══════════════════════════════════════════════════════════════════════════════

class TestSLAFunctions:
    """Pure SLA helper functions — deterministic datetime arithmetic."""

    def test_compute_sla_deadline_critical(self):
        """Critical priority → 1 hour SLA."""
        from workflows.sla import compute_sla_deadline
        now = datetime.now(tz=timezone.utc)
        deadline_str = compute_sla_deadline("critical", created_at=now)
        deadline = datetime.fromisoformat(deadline_str)
        diff_hours = (deadline - now).total_seconds() / 3600
        assert abs(diff_hours - 1.0) < 0.01

    def test_compute_sla_deadline_low(self):
        """Low priority → 72 hour SLA."""
        from workflows.sla import compute_sla_deadline
        now = datetime.now(tz=timezone.utc)
        deadline_str = compute_sla_deadline("low", created_at=now)
        deadline = datetime.fromisoformat(deadline_str)
        diff_hours = (deadline - now).total_seconds() / 3600
        assert abs(diff_hours - 72.0) < 0.01

    def test_compute_sla_deadline_unknown_priority_defaults_medium(self):
        """Unknown priority defaults to medium (24h)."""
        from workflows.sla import compute_sla_deadline
        now = datetime.now(tz=timezone.utc)
        deadline_str = compute_sla_deadline("unknown_priority", created_at=now)
        deadline = datetime.fromisoformat(deadline_str)
        diff_hours = (deadline - now).total_seconds() / 3600
        assert abs(diff_hours - 24.0) < 0.01

    def test_is_sla_breached_past_deadline(self):
        """A deadline in the past is breached."""
        from workflows.sla import is_sla_breached
        past = (datetime.now(tz=timezone.utc) - timedelta(hours=2)).isoformat()
        assert is_sla_breached(past) is True

    def test_is_sla_breached_future_deadline(self):
        """A deadline in the future is not breached."""
        from workflows.sla import is_sla_breached
        future = (datetime.now(tz=timezone.utc) + timedelta(hours=24)).isoformat()
        assert is_sla_breached(future) is False

    def test_hours_until_deadline_positive(self):
        """Future deadline → positive hours remaining."""
        from workflows.sla import hours_until_deadline
        future = (datetime.now(tz=timezone.utc) + timedelta(hours=5)).isoformat()
        hours = hours_until_deadline(future)
        assert 4.9 < hours < 5.1

    def test_hours_until_deadline_negative_when_breached(self):
        """Past deadline → negative hours remaining."""
        from workflows.sla import hours_until_deadline
        past = (datetime.now(tz=timezone.utc) - timedelta(hours=3)).isoformat()
        hours = hours_until_deadline(past)
        assert hours < 0


# ══════════════════════════════════════════════════════════════════════════════
# TestSLATrackingNode
# ══════════════════════════════════════════════════════════════════════════════

class TestSLATrackingNode:
    """sla_tracking_node — reads metadata, writes SLA audit event."""

    @pytest.mark.asyncio
    async def test_writes_sla_audit_event(self, base_state, runnable_config):
        from workflows.sla import sla_tracking_node
        result = await sla_tracking_node(base_state, runnable_config)

        assert "audit_events" in result
        assert len(result["audit_events"]) == 1
        event = result["audit_events"][0]
        assert event["node"] == "sla_tracking_node"
        assert "deadline" in event
        assert "breached" in event

    @pytest.mark.asyncio
    async def test_respects_ticket_metadata_priority(self, base_state, runnable_config):
        """Uses priority from ticket_metadata when available."""
        from workflows.sla import sla_tracking_node
        state = dict(base_state)
        state["ticket_metadata"] = {
            "ticket_id": "t-001",
            "tenant_id": "test",
            "user_id": "u-001",
            "priority": "critical",
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
            "sla_deadline": "",
        }

        result = await sla_tracking_node(state, runnable_config)
        event = result["audit_events"][0]
        assert event["priority"] == "critical"
        # Critical = 1 hour, so hours_remaining should be close to 1
        assert abs(event["hours_remaining"] - 1.0) < 0.1

    @pytest.mark.asyncio
    async def test_no_metadata_uses_default_priority(self, base_state, runnable_config):
        """Without ticket_metadata, defaults to medium priority."""
        from workflows.sla import sla_tracking_node
        result = await sla_tracking_node(base_state, runnable_config)
        event = result["audit_events"][0]
        assert event["priority"] == "medium"

    @pytest.mark.asyncio
    async def test_returns_partial_dict(self, base_state, runnable_config):
        from workflows.sla import sla_tracking_node
        result = await sla_tracking_node(base_state, runnable_config)
        assert "messages" not in result
        assert "ticket_type" not in result


# ══════════════════════════════════════════════════════════════════════════════
# TestAuditHelpers
# ══════════════════════════════════════════════════════════════════════════════

class TestAuditHelpers:
    """create_audit_event and format_audit_trail pure functions."""

    def test_create_audit_event_required_fields(self):
        from observability.audit import create_audit_event
        event = create_audit_event("billing_node", "refund_approved", "t-001")
        assert event["node"] == "billing_node"
        assert event["action"] == "refund_approved"
        assert event["thread_id"] == "t-001"
        assert "timestamp" in event

    def test_create_audit_event_timestamp_is_iso(self):
        from observability.audit import create_audit_event
        event = create_audit_event("n", "a", "t")
        # Should parse without error
        datetime.fromisoformat(event["timestamp"])

    def test_create_audit_event_extra_fields_included(self):
        from observability.audit import create_audit_event
        event = create_audit_event(
            "billing_node", "refund_approved", "t-001",
            amount=750.0, decision="approved"
        )
        assert event["amount"] == 750.0
        assert event["decision"] == "approved"

    def test_format_audit_trail_empty(self):
        from observability.audit import format_audit_trail
        assert format_audit_trail([]) == ""

    def test_format_audit_trail_single_event(self):
        from observability.audit import format_audit_trail
        events = [{"node": "billing_node", "action": "ticket_received", "thread_id": "t1"}]
        result = format_audit_trail(events)
        assert "billing_node" in result
        assert "ticket_received" in result

    def test_format_audit_trail_multiple_events(self):
        from observability.audit import format_audit_trail
        events = [
            {"node": "n1", "action": "a1", "thread_id": "t1"},
            {"node": "n2", "action": "a2", "thread_id": "t1"},
        ]
        result = format_audit_trail(events)
        lines = result.strip().split("\n")
        assert len(lines) == 2

    def test_format_audit_trail_includes_extra_fields(self):
        from observability.audit import format_audit_trail
        events = [{"node": "n", "action": "a", "thread_id": "t", "amount": 750.0}]
        result = format_audit_trail(events)
        assert "750" in result


# ══════════════════════════════════════════════════════════════════════════════
# TestAuditTrailNode
# ══════════════════════════════════════════════════════════════════════════════

class TestAuditTrailNode:
    """audit_trail_node — appends a session summary event."""

    @pytest.mark.asyncio
    async def test_writes_session_completed_event(self, base_state, runnable_config):
        from observability.audit import audit_trail_node
        result = await audit_trail_node(base_state, runnable_config)

        assert "audit_events" in result
        assert len(result["audit_events"]) == 1
        event = result["audit_events"][0]
        assert event["action"] == "session_completed"
        assert event["node"] == "audit_trail_node"

    @pytest.mark.asyncio
    async def test_records_total_events_count(self, base_state, runnable_config):
        from observability.audit import audit_trail_node
        state = dict(base_state)
        state["audit_events"] = [
            {"node": "n1", "action": "a1"},
            {"node": "n2", "action": "a2"},
        ]

        result = await audit_trail_node(state, runnable_config)
        event = result["audit_events"][0]
        assert event["total_events"] == 2

    @pytest.mark.asyncio
    async def test_records_escalation_level(self, base_state, runnable_config):
        from observability.audit import audit_trail_node
        state = dict(base_state)
        state["escalation_level"] = 2

        result = await audit_trail_node(state, runnable_config)
        assert result["audit_events"][0]["escalation_level"] == 2

    @pytest.mark.asyncio
    async def test_returns_partial_dict(self, base_state, runnable_config):
        from observability.audit import audit_trail_node
        result = await audit_trail_node(base_state, runnable_config)
        assert "messages" not in result
        assert "ticket_type" not in result


# ══════════════════════════════════════════════════════════════════════════════
# TestPIIDetection
# ══════════════════════════════════════════════════════════════════════════════

class TestPIIDetection:
    """detect_pii — pure function for PII pattern matching."""

    def test_detects_email(self):
        from security.compliance import detect_pii
        result = detect_pii("Contact me at user@example.com please")
        assert "email" in result

    def test_detects_phone(self):
        from security.compliance import detect_pii
        result = detect_pii("Call me at 555-123-4567")
        assert "phone" in result

    def test_detects_multiple_types(self):
        from security.compliance import detect_pii
        result = detect_pii("Email user@example.com or call 555-123-4567")
        assert "email" in result
        assert "phone" in result

    def test_clean_text_returns_empty(self):
        from security.compliance import detect_pii
        result = detect_pii("My invoice shows the wrong amount")
        assert result == []

    def test_detects_credit_card(self):
        from security.compliance import detect_pii
        result = detect_pii("My card 4111-1111-1111-1111 was charged twice")
        assert "credit_card" in result

    def test_empty_string_returns_empty(self):
        from security.compliance import detect_pii
        assert detect_pii("") == []


# ══════════════════════════════════════════════════════════════════════════════
# TestComplianceGateNode
# ══════════════════════════════════════════════════════════════════════════════

class TestComplianceGateNode:
    """compliance_gate_node — PII detection node."""

    @pytest.mark.asyncio
    async def test_clean_ticket_writes_audit_event(self, base_state, runnable_config):
        from security.compliance import compliance_gate_node
        state = dict(base_state)
        state["messages"] = [HumanMessage(content="My billing is wrong")]

        result = await compliance_gate_node(state, runnable_config)

        assert "audit_events" in result
        event = result["audit_events"][0]
        assert event["pii_detected"] is False
        assert event["pii_types"] == []

    @pytest.mark.asyncio
    async def test_pii_ticket_flags_in_audit(self, base_state, runnable_config):
        from security.compliance import compliance_gate_node
        state = dict(base_state)
        state["messages"] = [HumanMessage(content="Email me at user@example.com")]

        result = await compliance_gate_node(state, runnable_config)

        event = result["audit_events"][0]
        assert event["pii_detected"] is True
        assert "email" in event["pii_types"]

    @pytest.mark.asyncio
    async def test_empty_messages_returns_empty(self, base_state, runnable_config):
        from security.compliance import compliance_gate_node
        state = dict(base_state)
        state["messages"] = []

        result = await compliance_gate_node(state, runnable_config)
        assert result == {}

    @pytest.mark.asyncio
    async def test_returns_partial_dict(self, base_state, runnable_config):
        from security.compliance import compliance_gate_node
        state = dict(base_state)
        state["messages"] = [HumanMessage(content="billing question")]
        result = await compliance_gate_node(state, runnable_config)
        assert "messages" not in result
        assert "ticket_type" not in result

    @pytest.mark.asyncio
    async def test_always_writes_compliance_event(self, base_state, runnable_config):
        """Even clean tickets get a compliance audit event (complete audit trail)."""
        from security.compliance import compliance_gate_node
        state = dict(base_state)
        state["messages"] = [HumanMessage(content="no pii here")]

        result = await compliance_gate_node(state, runnable_config)

        assert len(result["audit_events"]) == 1
        assert result["audit_events"][0]["action"] == "pii_scan_complete"
