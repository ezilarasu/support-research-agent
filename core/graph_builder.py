"""
Module: core.graph_builder
Layer:  Infrastructure
Owner:  core/

Purpose:
    Fluent builder for LangGraph StateGraphs.
    Centralises all add_node / add_edge / add_conditional_edges calls so
    graph topology is declared in one place, not scattered across files.

    Why a builder over raw StateGraph API:
      - All topology in one place → easy to read the full graph structure
      - Method chaining → self-documenting construction
      - Compile kwargs isolated → checkpointer, interrupt_before etc. stay clean
      - Prevents the common mistake of calling graph.compile() before set_entry_point()

Public API:
    GraphBuilder  — fluent builder, returns a compiled graph from .build()

LangGraph Concepts demonstrated:
    - Compile (Concept 7) — .build() calls graph.compile()
    - Edges (Concept 5) — .edge() wraps add_edge()
    - Conditional Edges (Concept 6) — .conditional_edge() wraps add_conditional_edges()
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from langgraph.graph import StateGraph  # END is the terminal sentinel node

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    Fluent builder for LangGraph StateGraphs.

    Usage:
        graph = (
            GraphBuilder(AgentState)
            .node("classify", classify_ticket_node)
            .node("respond", respond_node)
            .edge("classify", "respond")
            .edge("respond", END)
            .entry("classify")
            .build()
        )

    With checkpointer (Module 4):
        graph = (
            GraphBuilder(AgentState)
            .node(...)
            .build(checkpointer=get_checkpointer())
        )

    With interrupt_before (Module 7 — HITL):
        graph = (
            GraphBuilder(AgentState)
            .node(...)
            .build(
                checkpointer=get_checkpointer(),
                interrupt_before=["approval_node"],
            )
        )
    """

    def __init__(self, state_schema: type) -> None:
        """
        Args:
            state_schema: The TypedDict class that defines graph state.
                          For us: AgentState from state.py
        """
        # StateGraph takes the state schema and creates the graph skeleton.
        # The schema defines what fields nodes can read and write.
        self._graph = StateGraph(state_schema)
        self._entry: str | None = None

        logger.debug("GraphBuilder initialised with schema=%s", state_schema.__name__)

    def node(self, name: str, fn: Callable) -> GraphBuilder:
        """
        Registers a node function under a name.

        Args:
            name: The node identifier used in edges and routing functions.
            fn:   The async (or sync) function that processes state.
                  Signature: async def fn(state: AgentState, config: RunnableConfig) -> dict

        Returns:
            self — for method chaining

        Example:
            .node("classify", classify_ticket_node)
        """
        self._graph.add_node(name, fn)
        logger.debug("node registered: %s → %s", name, fn.__name__)
        return self

    def edge(self, src: str, dst: str) -> GraphBuilder:
        """
        Adds an unconditional edge: src always transitions to dst.

        Use for linear steps where no branching is needed.

        Args:
            src: Source node name (or START sentinel).
            dst: Destination node name (or END sentinel).

        Returns:
            self — for method chaining
        """
        self._graph.add_edge(src, dst)
        logger.debug("edge added: %s → %s", src, dst)
        return self

    def conditional_edge(
        self,
        src: str,
        router_fn: Callable,
        mapping: dict[str, str] | None = None,
    ) -> GraphBuilder:
        """
        Adds a conditional edge: router_fn decides which node to go to next.

        Args:
            src:       Source node name.
            router_fn: Pure function (state) -> str | list[Send] that returns
                       a key from mapping OR a list of Send objects (fan-out).
            mapping:   Dict mapping router_fn string returns to destination
                       node names. Optional when router_fn always returns
                       Send objects (pure fan-out). When router_fn returns
                       a mix of strings and Send objects, include only the
                       string returns in mapping — Send destinations are
                       resolved automatically by LangGraph.

        Returns:
            self — for method chaining

        Example:
            # String routing (most common)
            .conditional_edge(
                "classify",
                route_ticket,
                {"billing": "billing_node", "technical": "supervisor_node"},
            )

            # Send API fan-out (Concept 84)
            .conditional_edge(
                "supervisor_node",
                route_after_supervisor,
                {"respond": "respond"},   # maps the non-fan-out path only
            )
        """
        if mapping is not None:
            self._graph.add_conditional_edges(src, router_fn, mapping)
            logger.debug(
                "conditional_edge added: %s → router=%s → %s",
                src,
                router_fn.__name__,
                list(mapping.keys()),
            )
        else:
            self._graph.add_conditional_edges(src, router_fn)
            logger.debug(
                "conditional_edge added (no mapping): %s → router=%s",
                src,
                router_fn.__name__,
            )
        return self

    def entry(self, node_name: str) -> GraphBuilder:
        """
        Sets the entry point — the first node LangGraph calls on invoke().

        Must be called before build(). Calling build() without entry() raises ValueError.

        Args:
            node_name: Name of the first node to execute.

        Returns:
            self — for method chaining
        """
        self._graph.set_entry_point(node_name)
        self._entry = node_name
        logger.debug("entry point set: %s", node_name)
        return self

    def build(self, **compile_kwargs: Any):
        """
        Compiles the StateGraph into an immutable, runnable CompiledStateGraph.

        This is the point of no return:
          - Topology is locked — no add_node/add_edge after this
          - All node name references are validated — typos raise immediately
          - Runtime machinery is attached: Pregel engine, state merging,
            checkpointer, interrupt support, streaming

        IMPORTANT — compile once, reuse forever:
          Compilation takes ~10–50ms and allocates the Pregel engine.
          Call build_graph() once at app startup; reuse for all requests.
          Never call build() inside a per-request handler.

        Args:
            **compile_kwargs: Passed directly to graph.compile().

            checkpointer (BaseCheckpointSaver | None):
                Enables state persistence. State is saved to the checkpointer
                before each node runs — graph is resumable after crash.
                None (default) = in-memory only, no persistence.
                Module 4 wires in SqliteSaver. Module 4+ uses PostgresSaver.

            interrupt_before (list[str] | None):
                Node names to pause before. Graph suspends at these nodes
                and waits for Command(resume=...) from the human operator.
                Module 7 wires in ["billing_node"] for refund approval.

            interrupt_after (list[str] | None):
                Node names to pause after (post-execution review).
                Less common than interrupt_before.

            debug (bool):
                Logs full state before and after every node.
                ALWAYS gate behind settings.graph_debug — never hardcode True.
                At scale: 1 request × 10 nodes × 10KB state = 100KB of debug logs.

        Returns:
            CompiledStateGraph: The runnable graph.
                Methods available after compile:
                  .invoke(state, config)   — sync, blocks until END
                  .ainvoke(state, config)  — async, awaits until END
                  .stream(state, config)   — sync generator, yields per node
                  .astream(state, config)  — async generator, yields per node
                  .get_graph()             — returns DrawableGraph for visualisation
                  .get_state(config)       — returns current checkpoint state
                  .get_state_history(config) — returns all checkpoints for thread

        Raises:
            ValueError: If .entry() was not called before .build().
        """
        if self._entry is None:
            raise ValueError(
                "GraphBuilder.build() called without setting an entry point. "
                "Call .entry('node_name') before .build()."
            )

        # Log which compile kwargs are active — helps diagnose missing checkpointer
        active_kwargs = {k: type(v).__name__ for k, v in compile_kwargs.items()}
        logger.debug("graph compiling: entry=%s kwargs=%s", self._entry, active_kwargs)

        compiled = self._graph.compile(**compile_kwargs)

        logger.info(
            "graph compiled: entry=%s nodes=%d kwargs=%s",
            self._entry,
            len(list(self._graph.nodes)),
            list(active_kwargs.keys()) or "none",
        )
        return compiled
