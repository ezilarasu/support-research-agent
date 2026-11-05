"""
Module: agents.billing_agent
Layer:  Graph
Owner:  agents/

Purpose:
    BillingAgent: specialist agent for billing support tickets.
    Handles refund detection and routes large refunds for HITL approval.

    This agent is intentionally SYNC-inside-async — the detection logic
    (regex matching) makes no LLM calls, so it delegates to billing_node
    (a sync function) rather than calling an LLM directly.

    The HITL flow is owned by the graph topology (interrupt_before=["approval_check"]),
    not by this agent class. BillingAgent signals HITL need by setting
    pending_approval in state; the graph pauses before approval_check runs.

Public API:
    BillingAgent  — BaseAgent subclass for billing tickets

LangGraph Concepts demonstrated:
    - Worker Agents (Concept 75)  — implements BaseAgent ABC contract
    - Registry Pattern (Concept 74) — @register_agent("billing") registers factory
    - Single Responsibility — this agent owns billing detection only
"""
from __future__ import annotations

import logging

from langchain_core.runnables import RunnableConfig  # noqa: TC002

from agents.base import BaseAgent
from agents.registry import register_agent

logger = logging.getLogger(__name__)


class BillingAgent(BaseAgent):
    """
    Specialist agent for billing support tickets.

    Responsibilities:
      - Extract refund amount from the ticket text (regex, zero tokens)
      - Detect large refunds (≥ $500) that require human approval
      - Set pending_approval in state to trigger the HITL interrupt_before flow
      - Write audit events for standard (no-approval) billing tickets

    No LLM calls in Module 8 — uses rule-based extraction from workflows.approval.
    Full response generation is handled downstream by respond_node.

    Module 9 extension:
      - Add KB retrieval for billing policies (FAQs, refund terms)
      - Inject retrieved_docs into state so respond_node uses them
    """

    @property
    def agent_name(self) -> str:
        return "billing"

    def get_tools(self) -> list:
        """Billing detection is rule-based — no tools needed."""
        return []

    async def process(
        self,
        state,
        config: RunnableConfig,
    ) -> dict:
        """
        Processes a billing ticket.

        Delegates to billing_node (synchronous function) which owns all
        billing detection logic. Wrapping sync logic in an async method
        follows the BaseAgent contract without adding unnecessary complexity.

        Reads:  state["messages"]   — scanned for refund amounts
        Writes: state["active_agent"]
                state["pending_approval"]  (large refund path)
                state["audit_events"]      (normal path)

        Args:
            state:  Current AgentState.
            config: RunnableConfig with thread_id.

        Returns:
            Partial state dict from billing_node.
        """
        # Late import avoids circular dependency during module initialization
        from agents.nodes import billing_node

        logger.debug("BillingAgent.process: delegating to billing_node")
        return billing_node(state, config)


@register_agent("billing")
def _create_billing_agent() -> BillingAgent:
    """
    Factory function registered under "billing" in the agent registry.

    Called by the supervisor: get_agent("billing")() → BillingAgent()
    Stateless: fresh instance per call — state lives in AgentState.
    """
    return BillingAgent()
