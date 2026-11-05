"""
Module: agents
Layer:  Graph
Owner:  agents/

Purpose:
    All LangGraph node functions, router functions, agent base class,
    agent registry, specialist agents, and supervisor.

    Module 8 additions:
      - BillingAgent, TechnicalAgent, GeneralAgent, ResearchAgent classes
      - supervisor_node: dispatches to registry agents by ticket_type
      - kb_worker_node, web_worker_node: parallel research workers (Send API)
      - research_collect_node: fan-in after parallel workers
      - route_after_supervisor: router fn for Send API fan-out

    Import note: the specialist agent modules are imported here for their
    @register_agent side effects. This ensures the registry is populated
    before supervisor_node calls get_agent() at runtime.

Public API:
    classify_ticket_node    — HF zero-shot ticket classification
    route_ticket            — pure router fn for conditional edges (Concept 6)
    billing_node            — billing ticket handler (HITL, Module 7)
    respond_node            — Groq LLM response generation
    ROUTE_KEYS              — mapping dict used by graph.py conditional edge
    BaseAgent               — ABC all specialist agents inherit from
    register_agent          — decorator: registers agent factory
    get_agent               — resolves agent by name from registry
    list_agents             — returns all registered agent names
    BillingAgent            — billing specialist agent (Module 8)
    TechnicalAgent          — technical specialist agent (Module 8)
    GeneralAgent            — general/fallback specialist agent (Module 8)
    ResearchAgent           — research specialist agent with fan-out (Module 8)
    supervisor_node         — dispatches to registry agents (Module 8)
    route_after_supervisor  — Send API router for research fan-out (Module 8)
    kb_worker_node          — parallel KB search worker (Module 8)
    web_worker_node         — parallel web search worker (Module 8)
    research_collect_node   — fan-in after parallel workers (Module 8)
"""
from agents.base import BaseAgent

# ── Specialist agents — imported for @register_agent side effects ─────────────
# These imports register each agent factory in the registry so supervisor_node
# can resolve them via get_agent("billing"), get_agent("technical"), etc.
# Order does not matter — registry is a plain dict keyed by agent name.
from agents.billing_agent import BillingAgent
from agents.general_agent import GeneralAgent
from agents.nodes import (
    ROUTE_KEYS,
    billing_node,
    classify_ticket_node,
    respond_node,
    route_ticket,
)
from agents.registry import get_agent, list_agents, register_agent
from agents.research_agent import ResearchAgent

# ── Supervisor and fan-out workers ────────────────────────────────────────────
from agents.supervisor import (
    kb_worker_node,
    research_collect_node,
    route_after_supervisor,
    supervisor_node,
    web_worker_node,
)
from agents.technical_agent import TechnicalAgent


def ensure_agents_registered() -> None:
    """
    Ensures all specialist agents are registered in the registry.

    Why this function exists:
      The @register_agent decorators run as module-level side effects when
      agent modules are first imported. Python caches module imports, so
      decorators do NOT re-run on subsequent imports. This means that if
      a test calls clear_registry() and later tests try to build the graph,
      the registry will be empty and supervisor_node will raise KeyError.

      This function handles that case by explicitly re-inserting factory
      functions if any are missing. It's idempotent — safe to call multiple
      times. Production: called by build_graph(); Tests: called by the same
      build_graph(), which fixes test ordering issues from clear_registry().

    This function directly accesses _REGISTRY to avoid triggering the
    @register_agent decorator twice (which logs a warning on overwrite).
    """
    from agents.registry import _REGISTRY

    if "billing" not in _REGISTRY:
        from agents.billing_agent import _create_billing_agent
        _REGISTRY["billing"] = _create_billing_agent

    if "technical" not in _REGISTRY:
        from agents.technical_agent import _create_technical_agent
        _REGISTRY["technical"] = _create_technical_agent

    if "general" not in _REGISTRY:
        from agents.general_agent import _create_general_agent
        _REGISTRY["general"] = _create_general_agent

    if "research" not in _REGISTRY:
        from agents.research_agent import _create_research_agent
        _REGISTRY["research"] = _create_research_agent


__all__ = [
    # Core nodes
    "classify_ticket_node",
    "route_ticket",
    "billing_node",
    "respond_node",
    "ROUTE_KEYS",
    # Agent system
    "BaseAgent",
    "register_agent",
    "get_agent",
    "list_agents",
    # Specialist agents (Module 8)
    "BillingAgent",
    "TechnicalAgent",
    "GeneralAgent",
    "ResearchAgent",
    # Supervisor + workers (Module 8)
    "supervisor_node",
    "route_after_supervisor",
    "kb_worker_node",
    "web_worker_node",
    "research_collect_node",
]
