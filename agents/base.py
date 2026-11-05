"""
Module: agents.base
Layer:  Graph
Owner:  agents/

Purpose:
    Abstract base class for all specialist agents.
    Enforces the contract every agent must implement:
      - process(state, config) → dict  (the node function)
      - get_tools() → list             (tools the agent has access to)
      - agent_name → str               (identity for routing and logging)

    Why an ABC?
      Open/Closed Principle: adding a new agent (LegalAgent, HRAgent) means
      creating a new subclass — never modifying the supervisor or registry.
      The supervisor calls agent.process() for any registered agent without
      knowing its concrete type.

Public API:
    BaseAgent   — abstract base class all agents inherit from

LangGraph Concepts demonstrated:
    - Dynamic Routing (Concept 18) — supervisor resolves agents via this contract
    - Open/Closed Principle        — extend by adding subclasses, not modifying base
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from langchain_core.runnables import RunnableConfig  # noqa: TC002

from state import AgentState  # noqa: TC001

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all specialist support agents.

    Every agent in the registry must inherit from this class.
    The supervisor node calls `agent.process(state, config)` without
    knowing which concrete agent it is — pure polymorphism.

    Subclass example:
        class BillingAgent(BaseAgent):
            @property
            def agent_name(self) -> str:
                return "billing"

            def get_tools(self) -> list:
                return get_billing_tools()

            async def process(self, state: AgentState, config: RunnableConfig) -> dict:
                llm = get_router_llm().bind_tools(self.get_tools())
                ...
                return {"active_agent": self.agent_name, "audit_events": [...]}
    """

    @property
    @abstractmethod
    def agent_name(self) -> str:
        """
        Unique identifier for this agent.
        Must match the key used in agents/registry.py: @register_agent("billing")
        Used in: state["active_agent"], audit_events, log lines.
        """
        ...

    @abstractmethod
    def get_tools(self) -> list:
        """
        Returns the list of LangChain tools this agent has access to.
        Called by the supervisor to bind tools to the LLM.

        Returns:
            List of BaseTool instances.
            Return [] if this agent uses no tools.
        """
        ...

    @abstractmethod
    async def process(
        self,
        state: AgentState,
        config: RunnableConfig,
    ) -> dict:
        """
        The agent's node function — called by the supervisor.

        Must follow the standard node contract:
          - Extract inputs at the top
          - Return a PARTIAL dict (only fields this agent changes)
          - Never mutate state directly
          - @with_retry applied at the call site (supervisor wraps it)

        Args:
            state:  Current AgentState. Read-only.
            config: RunnableConfig carrying thread_id, tenant_id, etc.

        Returns:
            Partial state dict with this agent's updates.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.agent_name!r})"
