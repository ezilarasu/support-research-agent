"""
Module: agents.research_agent
Layer:  Graph
Owner:  agents/

Purpose:
    ResearchAgent: specialist agent for research-type tickets requiring
    knowledge-base lookup or live web search.

    This agent extracts the search query from the ticket and signals the
    supervisor that research fan-out is needed. The actual parallel search
    is performed by kb_worker_node and web_worker_node in agents/supervisor.py.

    Fan-out model (Concept 84 — Send API):
      1. ResearchAgent.process() extracts query, sets active_agent = "research"
      2. route_after_supervisor() sees active_agent == "research" and returns:
           [Send("kb_worker", {...}), Send("web_worker", {...})]
      3. kb_worker and web_worker run IN PARALLEL (LangGraph handles this)
      4. Both append to agent_results (operator.add reducer — Concept 86)
      5. research_collect_node fans in and passes context to respond_node

    Module 8: query extraction + fan-out via Send API.
    Module 9: kb_worker_node gets full CRAG implementation (FAISS + reranking).

Public API:
    ResearchAgent  — BaseAgent subclass for research tickets

LangGraph Concepts demonstrated:
    - Specialist Agents (Concept 76) — research domain implementation
    - Send API (Concept 84)          — signals fan-out via active_agent
    - Fan-Out / Fan-In (Concepts 85–86) — parallel workers + operator.add
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


class ResearchAgent(BaseAgent):
    """
    Specialist agent for research tickets.

    Responsibilities:
      - Extract the search query from the ticket text
      - Set active_agent = "research" to signal fan-out routing
      - Set query in state so kb_worker and web_worker know what to search

    The agent does NOT perform the search itself — that is the job of
    kb_worker_node and web_worker_node (parallel fan-out workers in supervisor.py).
    This separation is intentional: ResearchAgent only knows what to search for,
    not how to search. The workers know how but not what.

    Why separate query extraction from execution?
      Single Responsibility Principle: this agent's job is domain analysis
      (understanding the research question), not I/O (fetching data).
      The fan-out workers handle I/O independently and in parallel.
    """

    @property
    def agent_name(self) -> str:
        return "research"

    def get_tools(self) -> list:
        """
        Returns tools available to the research worker nodes.

        These tools are used by web_worker_node, not by process() directly.
        Exposed here so supervisor can inspect research capabilities.
        """
        from tools.search_tool import SearchTool
        return [SearchTool()]

    async def process(
        self,
        state,
        config: RunnableConfig,
    ) -> dict:
        """
        Extracts the research query and signals fan-out routing.

        Sets active_agent = "research" and query — these two fields drive
        the route_after_supervisor function to emit Send objects for
        kb_worker and web_worker (parallel execution).

        Reads:  state["messages"]  — last message used as the search query

        Writes: state["active_agent"]  → "research"
                state["query"]         → extracted search query
                state["audit_events"]  → query_extracted event

        Args:
            state:  Current AgentState.
            config: RunnableConfig with thread_id.

        Returns:
            Partial state dict with active_agent, query, audit_events.
        """
        thread_id = _get_thread_id(config)
        messages = state.get("messages", [])
        query = messages[-1].content if messages else ""

        logger.info(
            "ResearchAgent.process: thread=%s query_len=%d",
            thread_id,
            len(query),
        )

        return {
            "active_agent": self.agent_name,
            "query": query,
            "audit_events": [{
                "node": "research_agent",
                "action": "query_extracted",
                "thread_id": thread_id,
                "query_len": len(query),
            }],
        }


@register_agent("research")
def _create_research_agent() -> ResearchAgent:
    """Factory registered under "research" in the agent registry."""
    return ResearchAgent()
