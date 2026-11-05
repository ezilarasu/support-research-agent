"""
Module: agents.general_agent
Layer:  Graph
Owner:  agents/

Purpose:
    GeneralAgent: specialist agent for general/uncategorised support tickets.
    Acts as the catch-all route — any ticket that doesn't match billing,
    technical, or research falls here.

    Also receives tickets where the classifier confidence was below threshold
    (< 0.4) — the safe fallback that prevents misrouting.

    Module 8: sets active_agent + audit_events, respond_node generates response.
    Module 9: add FAQ retrieval, policy lookups from KB.

Public API:
    GeneralAgent  — BaseAgent subclass for general/fallback tickets

LangGraph Concepts demonstrated:
    - Specialist Agents (Concept 76) — OCP: adding agents doesn't modify this
    - Worker Agents (Concept 75)     — BaseAgent ABC implementation
    - Registry Pattern (Concept 74)  — @register_agent("general") decorator
"""
from __future__ import annotations

import logging

from langchain_core.runnables import RunnableConfig  # noqa: TC002

from agents.base import BaseAgent
from agents.registry import register_agent

logger = logging.getLogger(__name__)


def _get_thread_id(config: RunnableConfig) -> str:
    """Safely extracts thread_id from RunnableConfig."""
    return config.get("configurable", {}).get("thread_id", "unknown")


class GeneralAgent(BaseAgent):
    """
    Specialist agent for general and uncategorised support tickets.

    This is the safe fallback route. Any ticket with:
      - ticket_type == "general" (explicit classification)
      - classification_confidence < 0.4 (low-confidence → general)
      - unknown ticket_type (supervisor fallback)
    will be handled here.

    Why a dedicated agent for "general" instead of a default branch?
      - Consistent BaseAgent contract — tests, hooks, and metrics treat it
        identically to billing/technical/research agents
      - Makes the fallback behaviour explicit and auditable
      - Easy to add FAQ retrieval in Module 9 without touching the supervisor
    """

    @property
    def agent_name(self) -> str:
        return "general"

    def get_tools(self) -> list:
        """
        Module 8: no tools.
        Module 9: add KnowledgeBaseTool for FAQ and policy retrieval.
        """
        return []

    async def process(
        self,
        state,
        config: RunnableConfig,
    ) -> dict:
        """
        Processes a general support ticket.

        Sets active_agent = "general" and writes an audit event.
        respond_node generates the final response using ticket context.

        Reads:  state["messages"]    — ticket text
                state["ticket_type"] — "general" (may be low-confidence fallback)

        Writes: state["active_agent"]  → "general"
                state["agent_results"] → appends result entry
                state["audit_events"]  → appends audit event

        Args:
            state:  Current AgentState.
            config: RunnableConfig with thread_id.

        Returns:
            Partial state dict.
        """
        thread_id = _get_thread_id(config)
        logger.info("GeneralAgent.process: thread=%s", thread_id)

        return {
            "active_agent": self.agent_name,
            "agent_results": [{
                "agent": self.agent_name,
                "action": "ticket_received",
                "thread_id": thread_id,
            }],
            "audit_events": [{
                "node": "general_agent",
                "action": "ticket_received",
                "thread_id": thread_id,
            }],
        }


@register_agent("general")
def _create_general_agent() -> GeneralAgent:
    """Factory registered under "general" in the agent registry."""
    return GeneralAgent()
