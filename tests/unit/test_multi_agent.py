"""
Unit tests for Module 8: Multi-Agent Systems

Tests cover:
  - BaseAgent contract: all specialist agents implement the ABC correctly
  - Registry: register, get_agent, list_agents, clear + re-register
  - BillingAgent: delegates to billing_node (sync detection, HITL)
  - TechnicalAgent: sets active_agent + audit_events
  - GeneralAgent: sets active_agent + audit_events (catch-all)
  - ResearchAgent: extracts query, sets active_agent = "research"
  - supervisor_node: dispatches to correct registry agent
  - route_after_supervisor: "respond" for tech/general, Send list for research
  - kb_worker_node: returns agent_results + audit_events
  - web_worker_node: returns agent_results + audit_events (Tavily fallback)
  - research_collect_node: fan-in counts results, writes audit event
  - ROUTE_KEYS: technical/general/research route to supervisor_node

Run:
    .venv/bin/python -m pytest tests/unit/test_multi_agent.py -v
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from langchain_core.messages import HumanMessage

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig

# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_state(**overrides) -> dict:
    """Minimal AgentState dict for multi-agent tests."""
    from langchain_core.messages import HumanMessage
    base = {
        "messages": [HumanMessage(content="My printer is not working")],
        "ticket_type": "technical",
        "ticket_metadata": None,
        "classification_confidence": 0.85,
        "thread_id": "ma-test-001",
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
    base.update(overrides)
    return base


def _make_config(thread_id: str = "ma-test-001") -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id, "user_id": "test-user"}}


# ══════════════════════════════════════════════════════════════════════════════
# REGISTRY TESTS — Concept 74: Supervisor Pattern / Registry
# ══════════════════════════════════════════════════════════════════════════════

class TestRegistry:
    """Registry correctly stores and resolves agent factories."""

    def test_all_specialist_agents_registered(self):
        """
        After importing the agents package, all 4 specialist agents
        must be registered. This verifies @register_agent side effects work.
        """
        from agents import ensure_agents_registered
        from agents.registry import list_agents

        ensure_agents_registered()
        registered = list_agents()

        for expected in ("billing", "technical", "general", "research"):
            assert expected in registered, f"Agent '{expected}' not registered"

    def test_get_agent_returns_correct_type(self):
        """get_agent() returns the right class for each name."""
        from agents import ensure_agents_registered
        from agents.billing_agent import BillingAgent
        from agents.general_agent import GeneralAgent
        from agents.registry import get_agent
        from agents.research_agent import ResearchAgent
        from agents.technical_agent import TechnicalAgent

        ensure_agents_registered()

        assert isinstance(get_agent("billing"),   BillingAgent)
        assert isinstance(get_agent("technical"), TechnicalAgent)
        assert isinstance(get_agent("general"),   GeneralAgent)
        assert isinstance(get_agent("research"),  ResearchAgent)

    def test_get_agent_returns_fresh_instances(self):
        """Each get_agent() call returns a new object — no shared state."""
        from agents import ensure_agents_registered
        from agents.registry import get_agent

        ensure_agents_registered()

        a1 = get_agent("technical")
        a2 = get_agent("technical")
        assert a1 is not a2, "get_agent() must return a new instance each call"

    def test_get_agent_raises_for_unknown_name(self):
        """get_agent() raises KeyError for unregistered names."""
        from agents.registry import get_agent

        with pytest.raises(KeyError, match="No agent registered"):
            get_agent("nonexistent_agent_xyz")

    def test_ensure_agents_registered_is_idempotent(self):
        """ensure_agents_registered() is safe to call multiple times."""
        from agents import ensure_agents_registered
        from agents.registry import list_agents

        ensure_agents_registered()
        count_after_first = len(list_agents())

        ensure_agents_registered()
        count_after_second = len(list_agents())

        assert count_after_first == count_after_second, (
            "ensure_agents_registered() must not duplicate registrations"
        )

    def test_ensure_restores_cleared_registry(self):
        """After clear_registry(), ensure_agents_registered() re-populates."""
        from agents import ensure_agents_registered
        from agents.registry import clear_registry, list_agents

        # Simulate what TestAgentRegistry.teardown_method does
        clear_registry()
        assert list_agents() == [], "clear_registry() must empty the registry"

        # Now restore
        ensure_agents_registered()
        assert len(list_agents()) >= 4, (
            "ensure_agents_registered() must re-register all agents after clear"
        )


# ══════════════════════════════════════════════════════════════════════════════
# BASE AGENT CONTRACT — Concept 75: Worker Agents
# ══════════════════════════════════════════════════════════════════════════════

class TestBaseAgentContract:
    """All specialist agents must satisfy the BaseAgent ABC contract."""

    def test_billing_agent_satisfies_abc(self):
        """BillingAgent implements all abstract methods without error."""
        from agents.base import BaseAgent
        from agents.billing_agent import BillingAgent

        agent = BillingAgent()
        assert isinstance(agent, BaseAgent)
        assert agent.agent_name == "billing"
        assert isinstance(agent.get_tools(), list)

    def test_technical_agent_satisfies_abc(self):
        from agents.base import BaseAgent
        from agents.technical_agent import TechnicalAgent

        agent = TechnicalAgent()
        assert isinstance(agent, BaseAgent)
        assert agent.agent_name == "technical"
        assert isinstance(agent.get_tools(), list)

    def test_general_agent_satisfies_abc(self):
        from agents.base import BaseAgent
        from agents.general_agent import GeneralAgent

        agent = GeneralAgent()
        assert isinstance(agent, BaseAgent)
        assert agent.agent_name == "general"
        assert isinstance(agent.get_tools(), list)

    def test_research_agent_satisfies_abc(self):
        from agents.base import BaseAgent
        from agents.research_agent import ResearchAgent

        agent = ResearchAgent()
        assert isinstance(agent, BaseAgent)
        assert agent.agent_name == "research"
        tools = agent.get_tools()
        assert isinstance(tools, list)
        # ResearchAgent exposes SearchTool
        tool_names = [t.name for t in tools]
        assert "web_search" in tool_names

    def test_all_agents_have_unique_names(self):
        """Each agent name must be unique — no two agents share a name."""
        from agents import ensure_agents_registered
        from agents.registry import get_agent, list_agents

        ensure_agents_registered()
        names = [get_agent(n).agent_name for n in list_agents()]
        assert len(names) == len(set(names)), "Duplicate agent names detected"


# ══════════════════════════════════════════════════════════════════════════════
# BILLING AGENT — delegates to billing_node (HITL preserved)
# ══════════════════════════════════════════════════════════════════════════════

class TestBillingAgent:
    """BillingAgent correctly delegates to billing_node."""

    @pytest.mark.asyncio
    async def test_normal_billing_sets_active_agent(self):
        """Standard billing ticket: active_agent = 'billing', audit event written."""
        from agents.billing_agent import BillingAgent

        state = _make_state(
            messages=[HumanMessage(content="I was charged twice for my subscription")],
            ticket_type="billing",
        )
        result = await BillingAgent().process(state, _make_config())

        assert result["active_agent"] == "billing"
        assert any(
            e["node"] == "billing_node" for e in result.get("audit_events", [])
        )

    @pytest.mark.asyncio
    async def test_large_refund_sets_pending_approval(self):
        """Large refund ticket: pending_approval set (HITL trigger)."""
        from agents.billing_agent import BillingAgent

        state = _make_state(
            messages=[HumanMessage(content="Please refund $750 to my account")],
            ticket_type="billing",
        )
        result = await BillingAgent().process(state, _make_config())

        assert result.get("pending_approval") is not None, (
            "Large refund should set pending_approval for HITL"
        )
        assert result["active_agent"] == "billing"

    @pytest.mark.asyncio
    async def test_returns_partial_dict(self):
        """BillingAgent.process() must NOT return full state."""
        from agents.billing_agent import BillingAgent

        state = _make_state(ticket_type="billing")
        result = await BillingAgent().process(state, _make_config())

        assert "messages" not in result
        assert "classification_confidence" not in result


# ══════════════════════════════════════════════════════════════════════════════
# TECHNICAL AGENT
# ══════════════════════════════════════════════════════════════════════════════

class TestTechnicalAgent:
    """TechnicalAgent sets active_agent and writes audit events."""

    @pytest.mark.asyncio
    async def test_sets_active_agent(self):
        from agents.technical_agent import TechnicalAgent

        state = _make_state(ticket_type="technical")
        result = await TechnicalAgent().process(state, _make_config())

        assert result["active_agent"] == "technical"

    @pytest.mark.asyncio
    async def test_appends_agent_results(self):
        from agents.technical_agent import TechnicalAgent

        state = _make_state(ticket_type="technical")
        result = await TechnicalAgent().process(state, _make_config())

        assert len(result.get("agent_results", [])) >= 1
        assert result["agent_results"][0]["agent"] == "technical"

    @pytest.mark.asyncio
    async def test_appends_audit_event(self):
        from agents.technical_agent import TechnicalAgent

        state = _make_state(ticket_type="technical")
        result = await TechnicalAgent().process(state, _make_config())

        assert len(result.get("audit_events", [])) >= 1
        assert any(e["node"] == "technical_agent" for e in result["audit_events"])

    @pytest.mark.asyncio
    async def test_returns_partial_dict(self):
        from agents.technical_agent import TechnicalAgent

        state = _make_state(ticket_type="technical")
        result = await TechnicalAgent().process(state, _make_config())

        assert "messages" not in result
        assert "ticket_type" not in result


# ══════════════════════════════════════════════════════════════════════════════
# GENERAL AGENT
# ══════════════════════════════════════════════════════════════════════════════

class TestGeneralAgent:
    """GeneralAgent is the catch-all — sets active_agent = 'general'."""

    @pytest.mark.asyncio
    async def test_sets_active_agent(self):
        from agents.general_agent import GeneralAgent

        state = _make_state(ticket_type="general")
        result = await GeneralAgent().process(state, _make_config())

        assert result["active_agent"] == "general"

    @pytest.mark.asyncio
    async def test_appends_audit_event(self):
        from agents.general_agent import GeneralAgent

        state = _make_state(ticket_type="general")
        result = await GeneralAgent().process(state, _make_config())

        assert any(e["node"] == "general_agent" for e in result.get("audit_events", []))


# ══════════════════════════════════════════════════════════════════════════════
# RESEARCH AGENT — query extraction + fan-out signal
# ══════════════════════════════════════════════════════════════════════════════

class TestResearchAgent:
    """ResearchAgent extracts query and signals fan-out routing."""

    @pytest.mark.asyncio
    async def test_sets_active_agent_to_research(self):
        from agents.research_agent import ResearchAgent

        state = _make_state(
            ticket_type="research",
            messages=[HumanMessage(content="What is the refund policy for SaaS products?")],
        )
        result = await ResearchAgent().process(state, _make_config())

        assert result["active_agent"] == "research"

    @pytest.mark.asyncio
    async def test_extracts_query_from_last_message(self):
        """query is extracted from the last message content."""
        from agents.research_agent import ResearchAgent

        question = "What is the latency SLA for the API?"
        state = _make_state(
            ticket_type="research",
            messages=[
                HumanMessage(content="Hello"),
                HumanMessage(content=question),
            ],
        )
        result = await ResearchAgent().process(state, _make_config())

        assert result["query"] == question

    @pytest.mark.asyncio
    async def test_sets_query_for_empty_messages(self):
        """Empty messages → query is empty string, no crash."""
        from agents.research_agent import ResearchAgent

        state = _make_state(ticket_type="research", messages=[])
        result = await ResearchAgent().process(state, _make_config())

        assert result["query"] == ""
        assert result["active_agent"] == "research"


# ══════════════════════════════════════════════════════════════════════════════
# SUPERVISOR NODE — Concept 74: Supervisor Pattern
# ══════════════════════════════════════════════════════════════════════════════

class TestSupervisorNode:
    """supervisor_node dispatches to the correct registry agent."""

    @pytest.mark.asyncio
    async def test_routes_technical_to_technical_agent(self):
        from agents import ensure_agents_registered
        from agents.supervisor import supervisor_node

        ensure_agents_registered()
        state = _make_state(ticket_type="technical")
        result = await supervisor_node(state, _make_config())

        assert result["active_agent"] == "technical"

    @pytest.mark.asyncio
    async def test_routes_general_to_general_agent(self):
        from agents import ensure_agents_registered
        from agents.supervisor import supervisor_node

        ensure_agents_registered()
        state = _make_state(ticket_type="general")
        result = await supervisor_node(state, _make_config())

        assert result["active_agent"] == "general"

    @pytest.mark.asyncio
    async def test_routes_research_to_research_agent(self):
        from agents import ensure_agents_registered
        from agents.supervisor import supervisor_node

        ensure_agents_registered()
        state = _make_state(
            ticket_type="research",
            messages=[HumanMessage(content="What are the API rate limits?")],
        )
        result = await supervisor_node(state, _make_config())

        assert result["active_agent"] == "research"
        assert result.get("query") is not None

    @pytest.mark.asyncio
    async def test_falls_back_to_general_for_unknown_type(self):
        """
        Unknown ticket_type falls back to general agent, not crash.
        This handles edge cases where classify_ticket returns an unexpected value.
        """
        from agents import ensure_agents_registered
        from agents.supervisor import supervisor_node

        ensure_agents_registered()
        state = _make_state(ticket_type="unknown_type_xyz")
        result = await supervisor_node(state, _make_config())

        assert result["active_agent"] == "general"

    @pytest.mark.asyncio
    async def test_dispatches_to_billing_agent_when_registered(self):
        from agents import ensure_agents_registered
        from agents.supervisor import supervisor_node

        ensure_agents_registered()
        state = _make_state(ticket_type="billing")
        result = await supervisor_node(state, _make_config())

        assert result["active_agent"] == "billing"


# ══════════════════════════════════════════════════════════════════════════════
# ROUTE AFTER SUPERVISOR — Concept 84: Send API
# ══════════════════════════════════════════════════════════════════════════════

class TestRouteAfterSupervisor:
    """route_after_supervisor returns 'respond' for tech/general, Send list for research."""

    def test_technical_routes_to_respond(self):
        from agents.supervisor import route_after_supervisor

        state = _make_state(active_agent="technical")
        result = route_after_supervisor(state)

        assert result == "respond"

    def test_general_routes_to_respond(self):
        from agents.supervisor import route_after_supervisor

        state = _make_state(active_agent="general")
        result = route_after_supervisor(state)

        assert result == "respond"

    def test_research_returns_send_list(self):
        """
        For research, route_after_supervisor returns a list of Send objects.
        This triggers LangGraph's parallel fan-out to kb_worker and web_worker.
        """
        from langgraph.types import Send

        from agents.supervisor import route_after_supervisor

        state = _make_state(
            active_agent="research",
            query="What is the API rate limit?",
        )
        result = route_after_supervisor(state)

        assert isinstance(result, list), "Research must return list[Send] for fan-out"
        assert len(result) == 2, "Research fan-out must dispatch exactly 2 workers"
        assert all(isinstance(s, Send) for s in result)

        node_names = {s.node for s in result}
        assert "kb_worker" in node_names, "kb_worker must be in research fan-out"
        assert "web_worker" in node_names, "web_worker must be in research fan-out"

    def test_research_send_includes_query(self):
        """Each Send must carry the query so workers know what to search."""
        from agents.supervisor import route_after_supervisor

        query = "What is the refund policy?"
        state = _make_state(active_agent="research", query=query)
        sends = route_after_supervisor(state)

        for send in sends:
            assert send.arg.get("query") == query, (
                f"Send to {send.node} must include query={query!r}"
            )

    def test_none_active_agent_routes_to_respond(self):
        """None active_agent (unexpected state) routes to respond safely."""
        from agents.supervisor import route_after_supervisor

        state = _make_state(active_agent=None)
        result = route_after_supervisor(state)

        assert result == "respond"

    def test_is_synchronous(self):
        """route_after_supervisor must NOT be async — LangGraph requires sync."""
        import inspect

        from agents.supervisor import route_after_supervisor

        assert not inspect.iscoroutinefunction(route_after_supervisor), (
            "route_after_supervisor must be sync — async router returns a coroutine "
            "object (truthy) which causes silent misrouting in LangGraph"
        )


# ══════════════════════════════════════════════════════════════════════════════
# KB WORKER NODE — Concept 85: Fan-Out
# ══════════════════════════════════════════════════════════════════════════════

class TestKbWorkerNode:
    """kb_worker_node returns agent_results + audit_events."""

    @pytest.mark.asyncio
    async def test_returns_agent_results(self):
        from agents.supervisor import kb_worker_node

        state = _make_state(query="printer not working", active_agent="research")
        result = await kb_worker_node(state, _make_config())

        assert len(result.get("agent_results", [])) >= 1
        assert result["agent_results"][0]["agent"] == "kb_worker"
        assert result["agent_results"][0]["source"] == "knowledge_base"

    @pytest.mark.asyncio
    async def test_returns_audit_event(self):
        from agents.supervisor import kb_worker_node

        state = _make_state(query="api error 429", active_agent="research")
        result = await kb_worker_node(state, _make_config())

        assert any(
            e["node"] == "kb_worker_node" for e in result.get("audit_events", [])
        )

    @pytest.mark.asyncio
    async def test_handles_empty_query(self):
        """Empty query should not raise — returns empty content gracefully."""
        from agents.supervisor import kb_worker_node

        state = _make_state(query="", active_agent="research")
        result = await kb_worker_node(state, _make_config())

        assert "agent_results" in result
        assert result["agent_results"][0]["query"] == ""


# ══════════════════════════════════════════════════════════════════════════════
# WEB WORKER NODE — Tavily search (graceful degradation without API key)
# ══════════════════════════════════════════════════════════════════════════════

class TestWebWorkerNode:
    """web_worker_node calls SearchTool and returns results."""

    @pytest.mark.asyncio
    async def test_returns_agent_results(self):
        from agents.supervisor import web_worker_node

        state = _make_state(query="LangGraph interrupt tutorial", active_agent="research")
        result = await web_worker_node(state, _make_config())

        assert len(result.get("agent_results", [])) >= 1
        assert result["agent_results"][0]["agent"] == "web_worker"
        assert result["agent_results"][0]["source"] == "web_search"

    @pytest.mark.asyncio
    async def test_returns_audit_event(self):
        from agents.supervisor import web_worker_node

        state = _make_state(query="test query", active_agent="research")
        result = await web_worker_node(state, _make_config())

        assert any(
            e["node"] == "web_worker_node" for e in result.get("audit_events", [])
        )

    @pytest.mark.asyncio
    async def test_handles_empty_query(self):
        """Empty query returns empty content without crash."""
        from agents.supervisor import web_worker_node

        state = _make_state(query="", active_agent="research")
        result = await web_worker_node(state, _make_config())

        assert "agent_results" in result
        assert result["agent_results"][0]["content"] == ""

    @pytest.mark.asyncio
    async def test_graceful_degradation_without_tavily_key(self):
        """Without TAVILY_API_KEY, web_worker returns placeholder, not crash."""
        from agents.supervisor import web_worker_node

        state = _make_state(query="What is the API rate limit?", active_agent="research")
        # SearchTool._run() returns a placeholder when TAVILY_API_KEY not set
        result = await web_worker_node(state, _make_config())

        # Must not raise — content may be placeholder or empty
        assert "agent_results" in result
        assert result["agent_results"][0]["agent"] == "web_worker"


# ══════════════════════════════════════════════════════════════════════════════
# RESEARCH COLLECT NODE — Concept 86: Fan-In
# ══════════════════════════════════════════════════════════════════════════════

class TestResearchCollectNode:
    """research_collect_node fans in parallel worker results."""

    @pytest.mark.asyncio
    async def test_writes_audit_event(self):
        from agents.supervisor import research_collect_node

        state = _make_state(
            active_agent="research",
            agent_results=[
                {"agent": "kb_worker", "source": "knowledge_base", "content": ""},
                {"agent": "web_worker", "source": "web_search", "content": "some results"},
            ],
        )
        result = await research_collect_node(state, _make_config())

        assert any(
            e["node"] == "research_collect_node"
            for e in result.get("audit_events", [])
        )

    @pytest.mark.asyncio
    async def test_counts_research_sources_correctly(self):
        """Audit event source_count reflects number of research worker results."""
        from agents.supervisor import research_collect_node

        state = _make_state(
            agent_results=[
                {"agent": "kb_worker", "content": ""},
                {"agent": "web_worker", "content": "results"},
                {"agent": "billing", "content": "unrelated"},  # should not count
            ],
        )
        result = await research_collect_node(state, _make_config())

        audit = next(
            e for e in result["audit_events"]
            if e["node"] == "research_collect_node"
        )
        assert audit["source_count"] == 2, (
            "source_count must count only kb_worker + web_worker results"
        )

    @pytest.mark.asyncio
    async def test_handles_empty_agent_results(self):
        """Empty agent_results — no crash, audit event still written."""
        from agents.supervisor import research_collect_node

        state = _make_state(agent_results=[])
        result = await research_collect_node(state, _make_config())

        assert "audit_events" in result


# ══════════════════════════════════════════════════════════════════════════════
# ROUTE_KEYS — confirm non-billing routes to supervisor_node
# ══════════════════════════════════════════════════════════════════════════════

class TestRouteKeys:
    """ROUTE_KEYS maps non-billing tickets to supervisor_node (Module 8 change)."""

    def test_billing_routes_to_billing_node(self):
        from agents.nodes import ROUTE_KEYS
        assert ROUTE_KEYS["billing"] == "billing_node"

    def test_technical_routes_to_supervisor(self):
        from agents.nodes import ROUTE_KEYS
        assert ROUTE_KEYS["technical"] == "supervisor_node", (
            "Module 8: technical tickets go through supervisor_node, not direct node"
        )

    def test_general_routes_to_supervisor(self):
        from agents.nodes import ROUTE_KEYS
        assert ROUTE_KEYS["general"] == "supervisor_node"

    def test_research_routes_to_supervisor(self):
        from agents.nodes import ROUTE_KEYS
        assert ROUTE_KEYS["research"] == "supervisor_node"

    def test_all_four_ticket_types_have_routes(self):
        from agents.nodes import ROUTE_KEYS
        for ticket_type in ("billing", "technical", "general", "research"):
            assert ticket_type in ROUTE_KEYS, f"Missing route for ticket_type={ticket_type!r}"
