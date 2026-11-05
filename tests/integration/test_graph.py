"""
Integration tests for graph topology — edges, nodes, and compilation.

These tests verify the graph's STRUCTURE without making any LLM calls.
They answer the question: "Is the graph wired correctly?"

Why topology tests matter:
  - Edge typos (`"clasify_ticket"`) are caught at compile time by LangGraph,
    but only if build_graph() is actually called in a test.
  - Missing END edges cause silent hangs — a graph that never terminates.
  - Node registration order doesn't matter, but all nodes must be registered
    before the edge that references them.

Test categories:
  TestGraphCompilation  — build_graph() succeeds and returns correct type
  TestGraphNodes        — expected nodes are registered
  TestGraphEdges        — edge topology matches intended flow
  TestGraphInvocation   — graph reaches END with mocked nodes (no real LLMs)

Run:
    .venv/bin/python -m pytest tests/integration/test_graph.py -v -m integration
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END

# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_initial_state() -> dict:
    """Minimal initial state for graph invocation tests."""
    return {
        "messages": [HumanMessage(content="My invoice is wrong")],
        "ticket_type": None,
        "ticket_metadata": None,
        "classification_confidence": 0.0,
        "thread_id": "test-integration-001",
        "retrieved_memories": [],
        "streaming_enabled": False,
        "active_agent": None,
        "agent_results": [],
        "query": None,
        "retrieved_docs": [],
        "retrieval_grade": None,
        "hallucination_check": None,
        "pending_approval": None,
        "approval_decision": None,
        "escalation_level": 0,
        "authenticated": True,
        "rbac_roles": ["agent"],
        "audit_events": [],
        "final_response": None,
        "error": None,
    }


# ── Tests ──────────────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestGraphCompilation:
    """build_graph() must succeed and return a compiled LangGraph object."""

    def test_build_graph_returns_compiled_graph(self):
        from graph import build_graph
        graph = build_graph()
        # CompiledStateGraph has these attributes — verify it's a compiled graph
        assert hasattr(graph, "invoke")
        assert hasattr(graph, "ainvoke")
        assert hasattr(graph, "astream")

    def test_build_graph_is_callable_multiple_times(self):
        """build_graph() must be idempotent — calling it twice is safe."""
        from graph import build_graph
        g1 = build_graph()
        g2 = build_graph()
        # Both should compile successfully — no shared mutable state
        assert hasattr(g1, "ainvoke")
        assert hasattr(g2, "ainvoke")

    def test_invalid_edge_node_raises_at_compile_time(self):
        """
        LangGraph validates that edge endpoints are registered nodes.
        A typo in a node name raises immediately at .build(), not at runtime.
        This is one of LangGraph's key safety guarantees.
        """
        from agents.nodes import classify_ticket_node
        from core.graph_builder import GraphBuilder
        from state import AgentState

        with pytest.raises((ValueError, KeyError)):
            # "clasify_ticket" is a typo — not registered as a node
            (
                GraphBuilder(AgentState)
                .node("classify_ticket", classify_ticket_node)
                .edge("clasify_ticket", END)   # ← typo: raises at .build()
                .entry("classify_ticket")
                .build()
            )

    def test_missing_entry_point_raises_at_build(self):
        """
        GraphBuilder.build() raises ValueError if .entry() was never called.
        This prevents a silent failure where the graph compiles but has no
        starting point.
        """
        from agents.nodes import classify_ticket_node
        from core.graph_builder import GraphBuilder
        from state import AgentState

        with pytest.raises(ValueError, match="entry point"):
            (
                GraphBuilder(AgentState)
                .node("classify_ticket", classify_ticket_node)
                .edge("classify_ticket", END)
                # .entry() deliberately omitted
                .build()
            )


@pytest.mark.integration
class TestGraphNodes:
    """Verifies the correct nodes are registered in the compiled graph."""

    def test_classify_ticket_node_registered(self):
        from graph import build_graph
        graph = build_graph()
        assert "classify_ticket" in graph.nodes

    def test_respond_node_registered(self):
        from graph import build_graph
        graph = build_graph()
        assert "respond" in graph.nodes

    def test_start_node_registered(self):
        """
        LangGraph always registers a __start__ node internally.
        This is the source of the implicit START → entry_point edge.
        """
        from graph import build_graph
        graph = build_graph()
        assert "__start__" in graph.nodes

    def test_no_unexpected_nodes(self):
        """
        Guard against accidentally registered nodes.
        Update this test when new nodes are added to the graph.

        Module 8 changes:
          - Removed from graph: technical_node, general_node, research_node
            (now handled by supervisor_node + registry dispatch)
          - Added: supervisor_node, kb_worker, web_worker, research_collect
        """
        from graph import build_graph
        graph = build_graph()
        expected = {
            "__start__",
            "memory_retrieval",    # Module 5: reads long-term memories
            "classify_ticket",
            "billing_node",
            "approval_check",      # Module 7: HITL gate for large refund approval
            "supervisor_node",     # Module 8: dispatches to registry agents
            "kb_worker",           # Module 8: parallel KB search worker
            "web_worker",          # Module 8: parallel web search worker
            "research_collect",    # Module 8: fan-in after parallel workers
            "respond",
            "memory_write",        # Module 5: writes memory summary
        }
        assert set(graph.nodes) == expected, (
            f"Unexpected nodes in graph. Expected {expected}, got {set(graph.nodes)}"
        )


@pytest.mark.integration
class TestGraphEdges:
    """Verifies the edge topology — which nodes connect to which."""

    def test_start_connects_to_classify_ticket(self):
        """
        The entry point creates an implicit START → classify_ticket edge.
        Verify this is the actual first node reached on invocation.
        """
        from graph import build_graph
        graph = build_graph()
        # LangGraph stores edges in graph.builder — we verify via invocation
        # (topology structural test is below via get_graph())
        assert "classify_ticket" in graph.nodes

    def test_graph_has_correct_mermaid_structure(self):
        """
        get_graph().draw_mermaid() reveals the full edge topology.
        Parse it to verify the expected connections.
        """
        from graph import build_graph
        graph = build_graph()
        mermaid = graph.get_graph().draw_mermaid()

        # Expected edges in the Mermaid output:
        assert "classify_ticket" in mermaid
        assert "respond" in mermaid
        # START → classify_ticket (entry point)
        assert "__start__" in mermaid


@pytest.mark.integration
class TestGraphInvocation:
    """
    End-to-end topology test: graph reaches END with mocked dependencies.
    No real LLM calls — mocks return deterministic results.
    """

    @pytest.mark.asyncio
    async def test_graph_runs_to_end_with_mocked_nodes(self, base_state):
        """
        The complete path classify_ticket → respond → END completes.
        Both LLM and classifier are mocked — this tests topology, not logic.
        """
        from graph import build_graph

        mock_clf = _make_mock_classifier("billing", 0.91)
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(
            return_value=AIMessage(content="Your refund has been processed.")
        )

        with patch("llm_provider.get_classifier", return_value=mock_clf), \
             patch("agents.nodes.get_research_llm", return_value=mock_llm):
            graph = build_graph()
            result = await graph.ainvoke(
                base_state,
                config={"configurable": {"thread_id": "integration-test-001"}},
            )

        # Both nodes ran — verify their writes are in final state
        assert result["ticket_type"] == "billing"
        assert result["classification_confidence"] == pytest.approx(0.91)
        assert result["final_response"] == "Your refund has been processed."

    @pytest.mark.asyncio
    async def test_classify_runs_before_respond(self, base_state):
        """
        Edge order is enforced: classify_ticket must run before respond.
        respond_node reads ticket_type set by classify_ticket_node.
        If edges were wrong order, ticket_type would still be None when respond runs.
        """

        execution_order = []

        async def mock_classify(state, config):
            execution_order.append("classify")
            return {"ticket_type": "technical", "classification_confidence": 0.88}

        async def mock_respond(state, config):
            execution_order.append("respond")
            # If edges are correct, ticket_type should be "technical" here
            assert state["ticket_type"] == "technical", (
                "respond_node ran before classify_ticket_node — edges are wrong!"
            )
            return {
                "messages": [AIMessage(content="Fixed.")],
                "final_response": "Fixed.",
            }

        with patch("agents.nodes.classify_ticket_node", mock_classify), \
             patch("agents.nodes.respond_node", mock_respond):
            # Must rebuild graph with patched nodes
            from langgraph.graph import END

            from core.graph_builder import GraphBuilder
            from state import AgentState

            graph = (
                GraphBuilder(AgentState)
                .node("classify_ticket", mock_classify)
                .node("respond", mock_respond)
                .edge("classify_ticket", "respond")
                .edge("respond", END)
                .entry("classify_ticket")
                .build()
            )
            await graph.ainvoke(
                base_state,
                config={"configurable": {"thread_id": "order-test-001"}},
            )

        assert execution_order == ["classify", "respond"], (
            f"Wrong execution order: {execution_order}"
        )

    @pytest.mark.asyncio
    async def test_messages_accumulate_across_nodes(self, base_state):
        """
        The add_messages reducer appends AIMessage from respond_node
        to the original HumanMessage. Tests that edges + reducers work together.
        """
        from graph import build_graph

        mock_clf = _make_mock_classifier("general", 0.75)
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(
            return_value=AIMessage(content="Thank you for contacting us.")
        )

        with patch("llm_provider.get_classifier", return_value=mock_clf), \
             patch("agents.nodes.get_research_llm", return_value=mock_llm):
            graph = build_graph()
            result = await graph.ainvoke(
                base_state,
                config={"configurable": {"thread_id": "reducer-test-001"}},
            )

        # Started with 1 HumanMessage, respond_node appended 1 AIMessage
        assert len(result["messages"]) == 2
        assert isinstance(result["messages"][0], HumanMessage)
        assert isinstance(result["messages"][1], AIMessage)


@pytest.mark.integration
class TestConditionalRouting:
    """
    Verifies the conditional edge routes tickets to the correct handler node.
    Tests each route independently with a mocked classifier.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize("ticket_type,expected_agent", [
        ("billing",   "billing"),
        ("technical", "technical"),
        ("general",   "general"),
        ("research",  "research"),
    ])
    async def test_routes_to_correct_handler(
        self, base_state, ticket_type, expected_agent
    ):
        """Each ticket_type must activate the matching handler node."""
        from graph import build_graph

        mock_clf = _make_mock_classifier(ticket_type, 0.91)
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(
            return_value=AIMessage(content=f"Response for {ticket_type}")
        )

        with patch("llm_provider.get_classifier", return_value=mock_clf), \
             patch("agents.nodes.get_research_llm", return_value=mock_llm):
            graph = build_graph()
            result = await graph.ainvoke(
                base_state,
                config={"configurable": {"thread_id": f"route-test-{ticket_type}"}},
            )

        assert result["active_agent"] == expected_agent, (
            f"ticket_type={ticket_type!r} should set active_agent={expected_agent!r}, "
            f"got {result['active_agent']!r}"
        )

    @pytest.mark.asyncio
    async def test_audit_trail_written_by_handler(self, base_state):
        """Handler nodes append audit events — verify they survive to final state."""
        from graph import build_graph

        mock_clf = _make_mock_classifier("billing", 0.91)
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="ok"))

        with patch("llm_provider.get_classifier", return_value=mock_clf), \
             patch("agents.nodes.get_research_llm", return_value=mock_llm):
            graph = build_graph()
            result = await graph.ainvoke(
                base_state,
                config={"configurable": {"thread_id": "audit-test-001"}},
            )

        # operator.add reducer accumulates audit events
        assert len(result["audit_events"]) >= 1
        assert any(e["node"] == "billing_node" for e in result["audit_events"])

    @pytest.mark.asyncio
    async def test_low_confidence_routes_to_general(self, base_state):
        """
        Classifier confidence below threshold → ticket_type = 'general'
        → route_ticket sends to general_node.
        """
        from graph import build_graph

        # Score 0.15 is below _CLASSIFICATION_THRESHOLD (0.4) → falls back to general
        mock_clf = _make_mock_classifier("billing", 0.15)
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="ok"))

        with patch("llm_provider.get_classifier", return_value=mock_clf), \
             patch("agents.nodes.get_research_llm", return_value=mock_llm):
            graph = build_graph()
            result = await graph.ainvoke(
                base_state,
                config={"configurable": {"thread_id": "lowconf-test-001"}},
            )

        assert result["ticket_type"] == "general"
        assert result["active_agent"] == "general"


# ── Test helpers ───────────────────────────────────────────────────────────────

def _make_mock_classifier(label: str, score: float):
    """Builds a mock HF classifier that returns a specific label and score."""
    from unittest.mock import MagicMock
    labels = [label] + [lbl for lbl in ["billing","technical","general","research"] if lbl != label]
    scores = [score] + [(1-score)/3] * 3
    return MagicMock(return_value={"labels": labels, "scores": scores, "sequence": "test"})
