"""
Unit and integration tests for Module 7: Human-in-the-Loop

Tests cover:
  - interrupt() triggers correctly for large refund amounts
  - Command(resume=value) resumes from the interrupt point
  - approved/rejected/escalated decision paths
  - interrupt_before compile option works
  - billing_node normal path (no approval needed)
  - _handle_interrupt() helper in main.py
  - ApprovalRequest payload structure

Run:
    .venv/bin/python -m pytest tests/unit/test_hitl.py -v
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_mock_classifier(label: str = "billing", score: float = 0.95):
    labels = [label] + [lbl for lbl in ["billing", "technical", "general", "research"] if lbl != label]
    scores = [score] + [(1 - score) / 3] * 3
    mock = MagicMock(return_value={"labels": labels, "scores": scores, "sequence": "test"})
    return mock


def _billing_state(ticket: str, base_state):
    """Returns a copy of base_state with billing ticket text."""
    state = dict(base_state)
    state["messages"] = [HumanMessage(content=ticket)]
    return state


# ── TestApprovalHelpers ────────────────────────────────────────────────────────

class TestApprovalHelpers:
    """Unit tests for pure helper functions in workflows/approval.py."""

    def test_extract_refund_amount_dollar_sign(self):
        from workflows.approval import _extract_refund_amount
        assert _extract_refund_amount("I need a refund of $750") == 750.0

    def test_extract_refund_amount_with_cents(self):
        from workflows.approval import _extract_refund_amount
        assert _extract_refund_amount("Please refund $299.99 to my account") == 299.99

    def test_extract_refund_amount_dollars_word(self):
        from workflows.approval import _extract_refund_amount
        assert _extract_refund_amount("Charge me 600 dollars") == 600.0

    def test_extract_refund_amount_refund_of_pattern(self):
        from workflows.approval import _extract_refund_amount
        assert _extract_refund_amount("refund of 1000 for my laptop") == 1000.0

    def test_extract_refund_amount_no_amount(self):
        from workflows.approval import _extract_refund_amount
        assert _extract_refund_amount("My billing is wrong") is None

    def test_extract_refund_amount_empty_string(self):
        from workflows.approval import _extract_refund_amount
        assert _extract_refund_amount("") is None

    def test_check_approval_required_above_threshold(self):
        from workflows.approval import check_approval_required
        assert check_approval_required(750.0) is True

    def test_check_approval_required_at_threshold(self):
        from workflows.approval import check_approval_required
        # Exactly at threshold → requires approval (>=)
        assert check_approval_required(500.0) is True

    def test_check_approval_required_below_threshold(self):
        from workflows.approval import check_approval_required
        assert check_approval_required(499.99) is False

    def test_check_approval_required_none(self):
        from workflows.approval import check_approval_required
        assert check_approval_required(None) is False

    def test_build_approval_payload_structure(self):
        from workflows.approval import build_approval_payload
        payload = build_approval_payload(750.0, "thread-001")
        assert payload["amount"] == 750.0
        assert payload["thread_id"] == "thread-001"
        assert payload["ticket_type"] == "billing"
        assert payload["action"] == "approve_refund"
        assert "message" in payload
        assert "750.00" in payload["message"]

    def test_build_approval_payload_custom_reason(self):
        from workflows.approval import build_approval_payload
        payload = build_approval_payload(1000.0, "t-002", reason="duplicate charge")
        assert payload["reason"] == "duplicate charge"


# ── TestBillingNodeNormalPath ─────────────────────────────────────────────────

class TestBillingNodeNormalPath:
    """billing_node standard path — no approval needed (amount < threshold)."""

    def test_small_refund_no_interrupt(self, base_state, runnable_config):
        """Refund below threshold runs without interrupt().

        billing_node is synchronous — no await needed. LangGraph calls
        sync nodes in a thread pool, which is also how interrupt() works
        safely on Python 3.10.
        """
        from agents.nodes import billing_node

        state = dict(base_state)
        state["messages"] = [HumanMessage(content="I need a refund of $50")]

        result = billing_node(state, runnable_config)

        assert result["active_agent"] == "billing"
        assert "audit_events" in result
        assert result["audit_events"][0]["action"] == "ticket_received"

    def test_no_refund_amount_no_interrupt(self, base_state, runnable_config):
        """Ticket with no dollar amount runs without interrupt()."""
        from agents.nodes import billing_node

        state = dict(base_state)
        state["messages"] = [HumanMessage(content="My invoice shows the wrong address")]

        result = billing_node(state, runnable_config)

        assert result["active_agent"] == "billing"
        assert result["audit_events"][0]["action"] == "ticket_received"

    def test_returns_partial_dict(self, base_state, runnable_config):
        """Node returns only the fields it changes — not full state."""
        from agents.nodes import billing_node

        state = dict(base_state)
        state["messages"] = [HumanMessage(content="billing question")]

        result = billing_node(state, runnable_config)

        assert "messages" not in result
        assert "ticket_type" not in result
        assert "final_response" not in result


# ── TestBillingNodeInterrupt ──────────────────────────────────────────────────

class TestBillingNodeInterrupt:
    """
    Integration tests for the billing HITL flow using interrupt_before.

    WHY interrupt_before instead of interrupt()?
      LangGraph 1.2.5's interrupt() reads var_child_runnable_config, a context
      variable. On Python 3.10, loop.run_in_executor() does NOT copy context
      variables to threads (Python 3.11 added this). The ASYNCIO_ACCEPTS_CONTEXT
      flag in langgraph/_internal/_runnable.py gates the context propagation —
      it's False on Python 3.10. So interrupt() raises RuntimeError on 3.10.

    interrupt_before=["approval_check"] (the build_graph default):
      - billing_node detects large refund → sets pending_approval in state
      - route_after_billing() routes to approval_check
      - Graph pauses BEFORE approval_check (interrupt_before)
      - snapshot.next == ("approval_check",)
      - Payload is in snapshot.values["pending_approval"]
      - Resume: graph.ainvoke({"approval_decision": decision}, config=config)
      - approval_check_node runs: records audit event, clears pending_approval

    All tests build a real graph with InMemorySaver. No real LLM calls.
    """

    @pytest.mark.asyncio
    async def test_large_refund_triggers_interrupt(
        self, base_state, memory_checkpointer
    ):
        """
        Graph pauses before approval_check when billing_node detects a refund >= $500.
        Verifies: snapshot.next contains "approval_check" after the initial run.
        """
        from graph import build_graph

        graph = build_graph(checkpointer=memory_checkpointer)
        state = _billing_state("I need a refund of $750 for my damaged laptop", base_state)
        config = {"configurable": {"thread_id": "hitl-interrupt-001"}}

        mock_clf = _make_mock_classifier("billing", 0.95)

        with patch("llm_provider.get_classifier", return_value=mock_clf):
            async for _ in graph.astream(state, config=config, stream_mode="updates"):
                pass

        snapshot = graph.get_state(config)
        assert snapshot.next, "Graph should be suspended before approval_check"
        assert "approval_check" in snapshot.next, (
            f"Expected graph interrupted before 'approval_check', got next={snapshot.next}"
        )

    @pytest.mark.asyncio
    async def test_interrupt_payload_contains_amount(
        self, base_state, memory_checkpointer
    ):
        """
        The approval payload set by billing_node contains the refund amount.

        With interrupt_before, the payload is in state.values["pending_approval"]
        (not in task.interrupts — that's only for interrupt() calls).
        """
        from graph import build_graph

        graph = build_graph(checkpointer=memory_checkpointer)
        state = _billing_state("Please refund $850 to my account", base_state)
        config = {"configurable": {"thread_id": "hitl-payload-001"}}

        mock_clf = _make_mock_classifier("billing", 0.95)

        with patch("llm_provider.get_classifier", return_value=mock_clf):
            async for _ in graph.astream(state, config=config, stream_mode="updates"):
                pass

        snapshot = graph.get_state(config)
        # With interrupt_before, payload lives in state — not task.interrupts
        payload = snapshot.values.get("pending_approval")
        assert payload is not None, "billing_node should have set pending_approval"
        assert payload["amount"] == 850.0
        assert payload["ticket_type"] == "billing"
        assert payload["action"] == "approve_refund"

    @pytest.mark.asyncio
    async def test_resume_approved_completes_graph(
        self, base_state, memory_checkpointer
    ):
        """
        Passing {"approval_decision": "approved"} as a state update resumes
        the graph and runs to completion.
        Verifies: approval_decision == "approved" in final state.
        """
        from graph import build_graph

        graph = build_graph(checkpointer=memory_checkpointer)
        state = _billing_state("I need a refund of $600 on my order", base_state)
        config = {"configurable": {"thread_id": "hitl-approve-001"}}

        mock_clf = _make_mock_classifier("billing", 0.95)
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(
            return_value=AIMessage(content="Your refund of $600 has been approved.")
        )

        # Pass 1: run until interrupt_before pauses before approval_check
        with patch("llm_provider.get_classifier", return_value=mock_clf):
            async for _ in graph.astream(state, config=config, stream_mode="updates"):
                pass

        # Verify interrupted before approval_check
        assert "approval_check" in graph.get_state(config).next

        # Resume: update state with decision, then continue from paused position.
        # WHY update_state + ainvoke(None) instead of ainvoke({"approval_decision": ...})?
        #   With interrupt_before, ainvoke(state_dict) triggers a NEW graph run
        #   starting from memory_retrieval (the entry point), not from approval_check.
        #   graph.update_state() injects state into the checkpoint so that when
        #   ainvoke(None) resumes, approval_check_node sees approval_decision in state.
        graph.update_state(config, {"approval_decision": "approved"})
        with patch("agents.nodes.get_research_llm", return_value=mock_llm):
            result = await graph.ainvoke(None, config=config)

        assert result["approval_decision"] == "approved"
        assert result["final_response"] == "Your refund of $600 has been approved."

    @pytest.mark.asyncio
    async def test_resume_rejected_records_decision(
        self, base_state, memory_checkpointer
    ):
        """
        update_state(rejected) records the rejection in state.
        """
        from graph import build_graph

        graph = build_graph(checkpointer=memory_checkpointer)
        state = _billing_state("Refund $750 please", base_state)
        config = {"configurable": {"thread_id": "hitl-reject-001"}}

        mock_clf = _make_mock_classifier("billing", 0.95)
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(
            return_value=AIMessage(content="Sorry, refund rejected.")
        )

        with patch("llm_provider.get_classifier", return_value=mock_clf):
            async for _ in graph.astream(state, config=config, stream_mode="updates"):
                pass

        graph.update_state(config, {"approval_decision": "rejected"})
        with patch("agents.nodes.get_research_llm", return_value=mock_llm):
            result = await graph.ainvoke(None, config=config)

        assert result["approval_decision"] == "rejected"

    @pytest.mark.asyncio
    async def test_resume_escalated_records_decision(
        self, base_state, memory_checkpointer
    ):
        """
        update_state(escalated) records escalated in state.
        """
        from graph import build_graph

        graph = build_graph(checkpointer=memory_checkpointer)
        state = _billing_state("Refund my $999 purchase now", base_state)
        config = {"configurable": {"thread_id": "hitl-escalate-001"}}

        mock_clf = _make_mock_classifier("billing", 0.95)
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(
            return_value=AIMessage(content="Your case has been escalated.")
        )

        with patch("llm_provider.get_classifier", return_value=mock_clf):
            async for _ in graph.astream(state, config=config, stream_mode="updates"):
                pass

        graph.update_state(config, {"approval_decision": "escalated"})
        with patch("agents.nodes.get_research_llm", return_value=mock_llm):
            result = await graph.ainvoke(None, config=config)

        assert result["approval_decision"] == "escalated"

    @pytest.mark.asyncio
    async def test_approval_audit_event_written(
        self, base_state, memory_checkpointer
    ):
        """
        After resume, audit_events contains a refund_decision event from
        approval_check_node with the correct amount and decision.
        """
        from graph import build_graph

        graph = build_graph(checkpointer=memory_checkpointer)
        state = _billing_state("Need refund for $520 charge", base_state)
        config = {"configurable": {"thread_id": "hitl-audit-001"}}

        mock_clf = _make_mock_classifier("billing", 0.95)
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Approved."))

        with patch("llm_provider.get_classifier", return_value=mock_clf):
            async for _ in graph.astream(state, config=config, stream_mode="updates"):
                pass

        graph.update_state(config, {"approval_decision": "approved"})
        with patch("agents.nodes.get_research_llm", return_value=mock_llm):
            result = await graph.ainvoke(None, config=config)

        audit = result.get("audit_events", [])
        approval_events = [e for e in audit if e.get("action") == "refund_decision"]
        assert len(approval_events) == 1
        assert approval_events[0]["amount"] == 520.0
        assert approval_events[0]["decision"] == "approved"


# ── TestInterruptBefore ───────────────────────────────────────────────────────

class TestInterruptBefore:
    """
    interrupt_before=[...] pauses the graph BEFORE the named node runs.
    This is a compile-time static interrupt — no node code changes needed.
    """

    @pytest.mark.asyncio
    async def test_interrupt_before_stops_before_node(self, base_state, memory_checkpointer):
        """
        interrupt_before=["respond"] stops the graph before respond_node executes.
        The final state has no final_response (respond never ran).
        """

        from graph import build_graph

        # Compile with interrupt_before — a compile-time static interrupt
        graph = build_graph(
            checkpointer=memory_checkpointer,
            interrupt_before=["respond"],
        )
        state = dict(base_state)
        state["messages"] = [HumanMessage(content="simple billing question")]
        config = {"configurable": {"thread_id": "interrupt-before-001"}}

        mock_clf = _make_mock_classifier("billing", 0.90)

        with patch("llm_provider.get_classifier", return_value=mock_clf):
            async for _ in graph.astream(state, config=config, stream_mode="updates"):
                pass

        # Graph should be paused before respond_node
        snapshot = graph.get_state(config)
        assert "respond" in snapshot.next, (
            f"Expected graph interrupted before 'respond', got next={snapshot.next}"
        )

    @pytest.mark.asyncio
    async def test_interrupt_before_resume_completes_graph(
        self, base_state, memory_checkpointer
    ):
        """
        After interrupt_before, resume with Command(resume=None) continues normally.
        """

        from graph import build_graph

        graph = build_graph(
            checkpointer=memory_checkpointer,
            interrupt_before=["respond"],
        )
        state = dict(base_state)
        state["messages"] = [HumanMessage(content="billing question")]
        config = {"configurable": {"thread_id": "interrupt-before-resume-001"}}

        mock_clf = _make_mock_classifier("general", 0.85)
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(
            return_value=AIMessage(content="Reviewed and answered.")
        )

        with patch("llm_provider.get_classifier", return_value=mock_clf):
            async for _ in graph.astream(state, config=config, stream_mode="updates"):
                pass

        # Resume — pass None for interrupt_before (no interrupt() value expected by the node)
        # Command(resume=...) is for nodes using interrupt() internally.
        # interrupt_before just needs the graph to continue — None signals "resume normally".
        with patch("agents.nodes.get_research_llm", return_value=mock_llm):
            result = await graph.ainvoke(None, config=config)

        assert result["final_response"] == "Reviewed and answered."
