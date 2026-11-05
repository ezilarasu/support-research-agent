"""
Module: graph
Layer:  Graph
Owner:  root — the main application graph

Purpose:
    Assembles and compiles the top-level support-research-agent graph.
    This file owns the TOPOLOGY only — node implementations live in agents/.

    Current topology (Module 8 — Multi-Agent with Supervisor + Fan-Out):

        [START]
           │
           ▼
        [memory_retrieval]           ── fetches long-term memories
           │
           ▼
        [classify_ticket]            ── sets ticket_type via HF zero-shot
           │
           │  conditional_edge(route_ticket)
           ├──► "billing"   ──► [billing_node]
           │                        │
           │         conditional_edge(route_after_billing)
           │                    ├──► [approval_check] ──► [respond]
           │                    └──────────────────────── [respond]
           │
           └──► "technical" ──► [supervisor_node]  ──► [respond]
           └──► "general"   ──► [supervisor_node]  ──► [respond]
           └──► "research"  ──► [supervisor_node]
                                    │
                     Send API fan-out via route_after_supervisor
                                    │
                             ┌──────┴──────┐
                             ▼             ▼
                         [kb_worker]  [web_worker]   (parallel)
                             │             │
                             └──────┬──────┘
                                    ▼
                           [research_collect]  ──► [respond]

        All paths converge:
        [respond] ──► [memory_write] ──► [END]

    Key Module 8 changes vs Module 7:
      - technical/general/research now route to supervisor_node (not direct nodes)
      - supervisor_node dispatches to registry agents via BaseAgent ABC
      - research uses Send API fan-out: kb_worker + web_worker run in parallel
      - research_collect fans in parallel results before respond_node

    Compile kwargs (injected via build_graph() parameters):
      checkpointer      — None (test), SqliteSaver (production, Module 4)
      interrupt_before  — ["approval_check"] (HITL gate, Module 7)
      debug             — gated behind settings.graph_debug

Public API:
    build_graph()       — returns a compiled graph ready for ainvoke/astream
    get_graph_info()    — returns topology summary dict for debugging

LangGraph Concepts demonstrated:
    - StateGraph (Concept 1)
    - Nodes (Concept 4)
    - Edges (Concept 5)
    - Conditional Edges (Concept 6)   — route_ticket + ROUTE_KEYS
    - Compile (Concept 7)             — compile kwargs, checkpointer
    - Supervisor Pattern (Concept 74) — supervisor_node + registry dispatch
    - Send API (Concept 84)           — route_after_supervisor fan-out
    - Fan-Out/Fan-In (Concepts 85–86) — kb_worker + web_worker + research_collect
"""
from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END

# Import specialist agents to populate the registry before build_graph() runs.
# These imports trigger @register_agent decorators as side effects.
# Without them, supervisor_node.get_agent("technical") would raise KeyError.
import agents.billing_agent  # noqa: F401
import agents.general_agent  # noqa: F401
import agents.research_agent  # noqa: F401
import agents.technical_agent  # noqa: F401
from agents import ensure_agents_registered
from agents.nodes import (
    ROUTE_KEYS,
    approval_check_node,
    billing_node,
    classify_ticket_node,
    memory_retrieval_node,
    memory_write_node,
    respond_node,
    route_after_billing,
    route_ticket,
)
from agents.supervisor import (
    kb_worker_node,
    research_collect_node,
    route_after_supervisor,
    supervisor_node,
    web_worker_node,
)
from core.graph_builder import GraphBuilder
from state import AgentState, RouterFn

logger = logging.getLogger(__name__)

# Type-annotate the router so mypy checks the (AgentState) -> str contract.
# If route_ticket accidentally becomes async, mypy catches it here.
_route_ticket: RouterFn = route_ticket

# Route keys for the billing conditional edge.
# Kept local to graph.py — only the graph topology needs these names.
_BILLING_ROUTE_KEYS: dict[str, str] = {
    "approval_check": "approval_check",
    "respond": "respond",
}


def build_graph(
    checkpointer: Any = None,
    interrupt_before: list[str] | None = None,
    debug: bool = False,
):
    """
    Assembles and compiles the support-research-agent graph.

    Module 8 topology:
      - billing tickets: billing_node → (approval_check | respond) → memory_write
      - technical/general: supervisor_node (registry dispatch) → respond → memory_write
      - research: supervisor_node → [kb_worker + web_worker] → research_collect → respond

    Args:
        checkpointer:     BaseCheckpointSaver or None.
                          None = in-memory only (tests / no persistence).
                          SqliteSaver = production (Module 4+).
        interrupt_before: Node names to pause before for HITL.
                          Defaults to ["approval_check"] — billing HITL gate.
        debug:            Enable verbose node logging.
                          Always gate behind settings.graph_debug.

    Returns:
        CompiledStateGraph
    """
    # Re-register specialist agents in case the registry was cleared by a test.
    ensure_agents_registered()

    # Default: always include approval_check in interrupt_before.
    # approval_check is only reachable when billing_node detects a large refund,
    # so this has zero impact on non-billing ticket flows.
    if interrupt_before is None:
        interrupt_before = ["approval_check"]

    compile_kwargs: dict[str, Any] = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer
    if interrupt_before:
        compile_kwargs["interrupt_before"] = interrupt_before
    if debug:
        compile_kwargs["debug"] = debug

    graph = (
        GraphBuilder(AgentState)

        # ── Nodes ─────────────────────────────────────────────────────────────
        .node("memory_retrieval",   memory_retrieval_node)
        .node("classify_ticket",    classify_ticket_node)
        # Billing path (HITL preserved from Module 7)
        .node("billing_node",       billing_node)
        .node("approval_check",     approval_check_node)
        # Supervisor path (Module 8 — handles technical/general/research)
        .node("supervisor_node",    supervisor_node)
        # Research fan-out workers (Module 8 — Send API parallel execution)
        .node("kb_worker",          kb_worker_node)
        .node("web_worker",         web_worker_node)
        .node("research_collect",   research_collect_node)
        # Common output nodes
        .node("respond",            respond_node)
        .node("memory_write",       memory_write_node)

        # ── Unconditional edges ───────────────────────────────────────────────
        .edge("memory_retrieval",  "classify_ticket")
        # Billing HITL path: approval_check always continues to respond
        .edge("approval_check",    "respond")
        # Research fan-in: both workers flow to collect, then respond
        .edge("kb_worker",         "research_collect")
        .edge("web_worker",        "research_collect")
        .edge("research_collect",  "respond")
        # Common output path
        .edge("respond",           "memory_write")
        .edge("memory_write",      END)

        # ── Conditional edges ─────────────────────────────────────────────────
        # Main ticket routing: classify → billing_node OR supervisor_node
        .conditional_edge("classify_ticket", _route_ticket, ROUTE_KEYS)
        # Billing routing: large refund → approval_check (HITL), normal → respond
        .conditional_edge("billing_node", route_after_billing, _BILLING_ROUTE_KEYS)
        # Supervisor routing:
        #   technical/general → "respond" (string return)
        #   research          → [Send(kb_worker), Send(web_worker)] (fan-out)
        # path_map only needs the string returns; Send destinations are automatic.
        .conditional_edge("supervisor_node", route_after_supervisor, {"respond": "respond"})

        # ── Entry ─────────────────────────────────────────────────────────────
        .entry("memory_retrieval")

        # ── Compile ───────────────────────────────────────────────────────────
        .build(**compile_kwargs)
    )

    logger.info(
        "graph compiled: checkpointer=%s interrupts=%s debug=%s",
        type(checkpointer).__name__ if checkpointer else "none",
        interrupt_before or "none",
        debug,
    )
    return graph


def get_graph_info() -> dict[str, Any]:
    """
    Returns a summary of the compiled graph for debugging and development.

    Compiles a fresh graph (no checkpointer) and extracts topology metadata.
    Use during development to verify nodes and edges are wired correctly.
    Never call this on the hot path — it compiles a full graph each time.

    Returns:
        dict with keys:
          nodes       — list of registered node names
          node_count  — number of nodes
          mermaid     — Mermaid diagram string (paste into mermaid.live)

    Example:
        info = get_graph_info()
        print(info["mermaid"])   # paste into https://mermaid.live
    """
    graph = build_graph()
    drawable = graph.get_graph()
    return {
        "nodes": list(graph.nodes),
        "node_count": len(list(graph.nodes)),
        "mermaid": drawable.draw_mermaid(),
    }
