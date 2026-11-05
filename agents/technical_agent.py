"""
Module: agents.technical_agent
Layer:  Graph
Owner:  agents/

Purpose:
    TechnicalAgent: specialist agent for technical support tickets.
    Sets active_agent and records an audit event so respond_node can
    generate a contextual response.

    Module 8 (this module):
      - Establishes the BaseAgent pattern for the technical domain
      - Registers with the supervisor via @register_agent("technical")
      - Prepares the state fields respond_node needs

    Module 9 extension (RAG):
      - Add KB retrieval: similarity_search() over the FAISS index
      - Inject retrieved_docs into state → respond_node uses them for grounded answers
      - Add escalation logic for unresolved issues

Public API:
    TechnicalAgent  — BaseAgent subclass for technical tickets

LangGraph Concepts demonstrated:
    - Worker Agents (Concept 75)   — implements BaseAgent ABC
    - Specialist Agents (Concept 76) — domain-specific implementation
    - Open/Closed Principle        — adding a new agent doesn't modify this one
"""
from __future__ import annotations

import logging

from langchain_core.runnables import RunnableConfig  # noqa: TC002

from agents.base import BaseAgent
from agents.registry import register_agent

logger = logging.getLogger(__name__)

# ── Helper ────────────────────────────────────────────────────────────────────

def _get_thread_id(config: RunnableConfig) -> str:
    """Safely extracts thread_id from RunnableConfig."""
    return config.get("configurable", {}).get("thread_id", "unknown")


class TechnicalAgent(BaseAgent):
    """
    Specialist agent for technical support tickets.

    Responsibilities (Module 8):
      - Set active_agent = "technical" so respond_node knows the ticket domain
      - Write an audit event recording that the technical agent handled this ticket
      - Add an agent_results entry for fan-in aggregation (Concept 86)

    Responsibilities (Module 9, added):
      - Retrieve from KB: known errors, troubleshooting guides, API docs
      - Set retrieved_docs in state so respond_node produces a grounded answer
      - Detect unresolved issues → set escalation_level if needed

    Why no LLM call in Module 8?
      The respond_node handles LLM response generation for all ticket types.
      Agents set context (active_agent, retrieved_docs) that respond_node uses.
      This separation keeps each node's responsibility clear and testable.
    """

    @property
    def agent_name(self) -> str:
        return "technical"

    def get_tools(self) -> list:
        """
        Returns tools available to TechnicalAgent.

        Module 8: no tools (pure rule-based + LLM via respond_node).
        Module 9: add KnowledgeBaseTool for KB retrieval.
        """
        return []

    async def process(
        self,
        state,
        config: RunnableConfig,
    ) -> dict:
        """
        Processes a technical support ticket.

        Sets active_agent and records an audit event.
        In Module 9: also performs KB retrieval, setting retrieved_docs.

        Reads:  state["messages"]    — ticket text (Module 9: used as RAG query)
                state["ticket_type"] — already "technical" (from classify node)

        Writes: state["active_agent"]  → "technical"
                state["agent_results"] → appends result entry (operator.add)
                state["audit_events"]  → appends audit event (operator.add)

        Args:
            state:  Current AgentState.
            config: RunnableConfig with thread_id.

        Returns:
            Partial state dict with active_agent, agent_results, audit_events.
        """
        thread_id = _get_thread_id(config)
        logger.info("TechnicalAgent.process: thread=%s", thread_id)

        # ── Module 9 stub ───────────────────────────────────────────────────
        # In Module 9, replace this comment with:
        #   from rag import get_rag_graph
        #   retrieval_result = await _retrieve_technical_docs(state["query"])
        #   retrieved_docs = retrieval_result["documents"]
        # Then add retrieved_docs to the return dict so respond_node uses them.

        return {
            "active_agent": self.agent_name,
            "agent_results": [{
                "agent": self.agent_name,
                "action": "ticket_received",
                "thread_id": thread_id,
            }],
            "audit_events": [{
                "node": "technical_agent",
                "action": "ticket_received",
                "thread_id": thread_id,
            }],
        }


@register_agent("technical")
def _create_technical_agent() -> TechnicalAgent:
    """
    Factory registered under "technical" in the agent registry.

    Called by: supervisor_node → get_agent("technical") → TechnicalAgent()
    Fresh instance per call — no shared mutable state between requests.
    """
    return TechnicalAgent()
