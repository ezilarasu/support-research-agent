"""
Module: agents.registry
Layer:  Graph
Owner:  agents/

Purpose:
    Central registry that maps agent names to factory functions.
    The supervisor resolves agents by name from this registry — it never
    imports concrete agent classes directly (Dependency Inversion Principle).

    Pattern: Registry + Decorator
      @register_agent("billing") decorates a factory function.
      The supervisor calls get_agent("billing") to get a BillingAgent instance.
      Adding a new agent = adding a new @register_agent decorated factory.
      The supervisor code never changes.

Public API:
    register_agent(name)  — decorator: registers a factory under a name
    get_agent(name)       — resolves and instantiates a registered agent
    list_agents()         — returns all registered agent names
    _REGISTRY             — internal dict (exposed for testing only)

LangGraph Concepts demonstrated:
    - Dynamic Routing (Concept 18) — supervisor uses list_agents() to know routes
    - Registry Pattern             — maps names → factories, enables OCP
    - Dependency Inversion         — supervisor depends on registry, not concrete agents
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.base import BaseAgent

logger = logging.getLogger(__name__)

# ── Internal registry ─────────────────────────────────────────────────────────
# Maps agent name → factory function that creates the agent instance.
# Factory pattern: we store factories, not instances.
# This means each get_agent() call creates a fresh instance — no shared state.

AgentFactory = Callable[[], "BaseAgent"]
_REGISTRY: dict[str, AgentFactory] = {}


# ── Public API ────────────────────────────────────────────────────────────────

def register_agent(name: str) -> Callable[[AgentFactory], AgentFactory]:
    """
    Decorator: registers an agent factory function under a name.

    The supervisor discovers available agents via list_agents().
    It routes to them via get_agent(name).process(state, config).

    Args:
        name: The routing key for this agent. Must match a TicketType value.
              ("billing", "technical", "general", "research")

    Returns:
        The original factory function, unmodified. The decorator only
        registers it as a side effect — it doesn't wrap it.

    Example:
        @register_agent("billing")
        def create_billing_agent() -> BillingAgent:
            return BillingAgent(llm=get_router_llm(), tools=get_billing_tools())

        # Later, in the supervisor:
        agent = get_agent("billing")   # calls create_billing_agent()
        result = await agent.process(state, config)
    """
    def decorator(factory: AgentFactory) -> AgentFactory:
        if name in _REGISTRY:
            logger.warning(
                "register_agent: overwriting existing registration for name=%r",
                name,
            )
        _REGISTRY[name] = factory
        logger.debug("register_agent: registered name=%r factory=%s", name, factory.__name__)
        return factory   # return unchanged — decorator is for side effect only

    return decorator


def get_agent(name: str) -> BaseAgent:
    """
    Resolves and instantiates a registered agent by name.

    Each call creates a FRESH instance — agents are stateless.
    State lives in AgentState, not in the agent objects.

    Args:
        name: Agent name, e.g. "billing", "technical".

    Returns:
        A new instance of the registered agent.

    Raises:
        KeyError: If no agent is registered under this name.
                  Includes the list of available agents in the error message.
    """
    if name not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise KeyError(
            f"No agent registered under '{name}'. "
            f"Available agents: {available}. "
            f"Did you forget to call @register_agent('{name}')?"
        )
    return _REGISTRY[name]()


def list_agents() -> list[str]:
    """
    Returns all registered agent names.

    Used by the supervisor to build its routing table dynamically.
    The supervisor never hardcodes agent names — it reads from the registry.
    This means adding a new agent only requires @register_agent, not touching
    the supervisor.

    Returns:
        List of registered agent name strings.
    """
    return list(_REGISTRY.keys())


def clear_registry() -> None:
    """
    Clears all registered agents.
    Used in tests only — never call in production code.
    """
    _REGISTRY.clear()
    logger.debug("registry cleared")
