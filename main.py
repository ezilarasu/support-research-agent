"""
Module: main
Layer:  Delivery
Owner:  root

Purpose:
    CLI entry point for support-research-agent.
    This is the OUTERMOST layer — it owns all I/O and calls the graph.

    Clean architecture rules enforced here:
      - Graph logic, LLM calls, and business rules NEVER live here
      - All print/Rich formatting lives here — never in node files
      - compile-once pattern: build_graph() called once, reused for session

    Session model:
      - One session = one CLI run (process lifetime)
      - One session = one thread_id (UUID generated at startup)
      - Multiple messages within a session share the same thread_id

    Current state (Module 6 — Streaming):
      - astream() with stream_mode="messages": tokens appear word-by-word
      - Updates mode progress: node names printed as they complete
      - streaming_enabled=True set in initial state to activate astream path
      - SqliteSaver checkpointer: sessions persist across restarts

LangGraph Concepts demonstrated:
    - Invoke (Concept 8)            — ainvoke, config, thread_id management
    - graph.astream() (Concept 43)  — async generator, yields chunks
    - Stream Modes (Concept 44)     — "messages" for tokens, "updates" for nodes
    - messages mode (Concept 47)    — AIMessageChunk word-by-word display
    - Token Streaming (Concept 49)  — collecting AIMessageChunk content
    - Streaming UIs (Concept 51)    — all Rich/print in delivery layer only
"""
from __future__ import annotations

import asyncio
import uuid

from langchain_core.messages import HumanMessage

# configure_logging MUST be called before any other import that touches logging
from core.logging import configure_logging

configure_logging()

import logging  # noqa: E402

from core.settings import get_settings  # noqa: E402
from graph import build_graph  # noqa: E402
from observability.metrics import configure_metrics  # noqa: E402
from observability.tracing import configure_tracing  # noqa: E402

logger = logging.getLogger(__name__)


def _build_config(thread_id: str, settings) -> dict:
    """
    Builds the RunnableConfig dict for a graph invocation.

    WHY "configurable" namespace?
      LangGraph reads thread_id, user_id, and checkpoint metadata from
      config["configurable"]. Anything outside that key is invisible to
      the checkpointer and thread management machinery.

    Args:
        thread_id: Session-scoped UUID. Same for all messages in one session.
        settings:  Settings singleton — provides recursion_limit.

    Returns:
        dict: Valid RunnableConfig ready to pass to graph.astream().
    """
    return {
        "configurable": {
            "thread_id": thread_id,
        },
        "recursion_limit": settings.graph_recursion_limit,
    }


def _build_initial_state(ticket: str) -> dict:
    """
    Builds the initial state dict for a graph invocation.

    Module 6 change: streaming_enabled=True activates the astream() branch
    in respond_node, which uses llm.astream() instead of llm.ainvoke().

    Args:
        ticket: Raw ticket text from the user.

    Returns:
        dict: Minimal initial state with the HumanMessage and streaming flag set.
    """
    return {
        "messages": [HumanMessage(content=ticket)],
        "ticket_type": None,
        "ticket_metadata": None,
        "classification_confidence": 0.0,
        "thread_id": None,
        "retrieved_memories": [],
        "streaming_enabled": True,          # ← Module 6: activates astream path
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


async def run() -> None:
    """
    Main async session loop.

    Session lifecycle:
      1. Generate a session-scoped thread_id (one per process run)
      2. Compile the graph once
      3. Loop: read ticket → astream → display tokens in real time
      4. Ctrl+C or EOF terminates gracefully

    Module 6 upgrade: astream() with stream_mode="messages" for token streaming.
    """
    settings = get_settings()

    # Module 10: configure tracing + metrics at startup
    configure_tracing(settings)
    configure_metrics()

    session_id = str(uuid.uuid4())
    logger.info(
        "session started: thread_id=%s provider=%s tracing=%s",
        session_id,
        settings.llm_provider,
        settings.langchain_tracing_v2,
    )

    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    async with AsyncSqliteSaver.from_conn_string(settings.sqlite_path) as checkpointer:
        graph = build_graph(checkpointer=checkpointer, debug=settings.graph_debug)

        _print_banner(session_id, settings)

        while True:
            try:
                ticket = input("You: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye.")
                logger.info("session ended: thread_id=%s", session_id)
                break

            if not ticket:
                continue

            config = _build_config(session_id, settings)
            initial_state = _build_initial_state(ticket)

            logger.info(
                "streaming graph: thread_id=%s ticket_len=%d",
                session_id,
                len(ticket),
            )

            try:
                await _stream_response(graph, initial_state, config)
            except Exception as e:
                logger.error(
                    "graph stream failed: thread_id=%s error=%s",
                    session_id,
                    e,
                )
                print(f"\nAgent: Sorry, something went wrong: {e}\n")


async def _stream_response(graph, initial_state: dict, config: dict) -> None:
    """
    Runs the graph and prints the response to the terminal.

    Single pass with stream_mode="updates":
      - Yields {node_name: {partial_state}} after each node completes.
      - Captures ticket_type, active_agent, final_response, and error from updates.
      - Prints node progress indicators while the graph runs.
      - Prints the final_response after the graph completes.

    WHY not stream_mode="messages" (token streaming)?
      stream_mode="messages" requires a single ainvoke/astream call and does
      not yield node-level updates — we would lose progress indicators and
      metadata (ticket_type, agent) unless we read aget_state() afterward.
      The updates approach is simpler and reliable. Token-by-token streaming
      can be added later by switching to stream_mode="messages" and calling
      aget_state() post-run for metadata.

    Clean architecture rule:
      All print() calls live in this function only. Nodes never print.

    Args:
        graph:         The compiled LangGraph graph.
        initial_state: The starting state dict.
        config:        RunnableConfig with thread_id.
    """
    ticket_type = "unknown"
    active_agent = "general"
    confidence = 0.0
    final_response = None
    final_error = None
    agent_results: list = []

    print()  # blank line before agent output

    async for event in graph.astream(initial_state, config=config, stream_mode="updates"):
        for node_name, node_output in event.items():
            logger.debug("node_completed: %s", node_name)
            _print_node_progress(node_name)

            if isinstance(node_output, dict):
                if node_output.get("ticket_type"):
                    ticket_type = node_output["ticket_type"]
                if node_output.get("active_agent"):
                    active_agent = node_output["active_agent"]
                if "classification_confidence" in node_output:
                    confidence = node_output["classification_confidence"]
                if node_output.get("final_response"):
                    final_response = node_output["final_response"]
                if node_output.get("error"):
                    final_error = node_output["error"]
                if node_output.get("agent_results"):
                    agent_results.extend(node_output["agent_results"])

    if final_error:
        print(f"\nAgent [error]: {final_error}\n")
        return

    # ── Interrupt detection ───────────────────────────────────────────────────
    state_snapshot = await graph.aget_state(config)
    if state_snapshot.next:
        interrupt_handled = await _handle_interrupt(graph, state_snapshot, config)
        if not interrupt_handled:
            return

        # After resume, read final_response from the completed checkpoint
        final_snapshot = await graph.aget_state(config)
        final_vals = final_snapshot.values
        if final_vals.get("ticket_type"):
            ticket_type = final_vals["ticket_type"]
        if final_vals.get("active_agent"):
            active_agent = final_vals["active_agent"]
        if final_vals.get("error"):
            print(f"\nAgent [error]: {final_vals['error']}\n")
            return
        if final_vals.get("final_response"):
            final_response = final_vals["final_response"]

    # ── Print response ────────────────────────────────────────────────────────
    print(f"\r\033[KAgent [{ticket_type} • {active_agent} • conf={confidence:.0%}]:")
    print(final_response or "(no response generated)")

    # ── Print sources for research tickets ───────────────────────────────────
    _print_sources(agent_results)

    print()
    logger.info(
        "response_complete: ticket_type=%s agent=%s response_len=%d",
        ticket_type,
        active_agent,
        len(final_response) if final_response else 0,
    )


async def _handle_interrupt(graph, state_snapshot, config: dict) -> bool:
    """
    Handles a graph interrupt_before pause — shows the approval request and collects
    the operator's decision, then resumes the graph with the decision as a state update.

    Concept 53 — interrupt_before: graph is paused BEFORE approval_check runs.
    Concept 55 — Resume: graph.ainvoke({"approval_decision": decision}, config)
                 merges the decision into state before approval_check executes.

    Called when Pass 1 detects state_snapshot.next is non-empty (interrupted).

    WHY state update instead of Command(resume=...)?
      With interrupt_before, there is no interrupt() call inside the node —
      the node hasn't run yet. Command(resume=...) is only for nodes that call
      interrupt() internally. For interrupt_before, we resume by passing a
      partial state dict that merges into the graph state before the paused
      node runs. The paused node (approval_check_node) then reads
      state["approval_decision"] to record the operator's decision.

    Flow:
      1. Read pending_approval from state_snapshot.values — contains amount, message
      2. Display the approval request to the operator
      3. Read decision from stdin (approved / rejected / escalated)
      4. Resume: graph.astream({"approval_decision": decision}, stream_mode="updates")
      5. Return True — Pass 2 can now stream tokens from completed checkpoint

    Args:
        graph:          Compiled LangGraph graph.
        state_snapshot: Result of graph.get_state(config) after Pass 1.
        config:         RunnableConfig with thread_id.

    Returns:
        True if interrupt was handled (decision received and graph resumed).
        False if user aborted (empty input) — caller should exit early.
    """
    # ── Extract payload from state ─────────────────────────────────────────────
    # With interrupt_before, the payload is in state.values["pending_approval"],
    # set by billing_node. (interrupt() would put it in task.interrupts instead.)
    payload = state_snapshot.values.get("pending_approval")

    if not payload:
        # state.next is non-empty but no pending_approval — not a billing interrupt.
        # Could be interrupt_before=["respond"] or another node. Let Pass 2 handle.
        return True

    # ── Display the approval request ───────────────────────────────────────────
    print()
    print("  " + "─" * 50)
    print("  ⚠  APPROVAL REQUIRED")
    if isinstance(payload, dict):
        message = payload.get("message", "An action requires your approval.")
        print(f"  {message}")
        for key, val in payload.items():
            if key not in {"message", "action", "ticket_type"}:
                print(f"    {key}: {val}")
    else:
        print(f"  {payload}")
    print()
    print("  Options: approved / rejected / escalated")
    print("  " + "─" * 50)

    # ── Collect decision ───────────────────────────────────────────────────────
    _VALID_DECISIONS = {"approved", "rejected", "escalated"}
    try:
        raw = input("  Decision: ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        print("\n  Aborted.")
        return False

    if not raw:
        print("  No decision entered — aborting.")
        return False

    # Normalise: accept "approve" → "approved", "reject" → "rejected", etc.
    decision = raw
    if raw == "approve":
        decision = "approved"
    elif raw == "reject":
        decision = "rejected"
    elif raw == "escalate":
        decision = "escalated"

    if decision not in _VALID_DECISIONS:
        print(f"  Unknown decision {decision!r} — defaulting to 'rejected'.")
        decision = "rejected"

    logger.info(
        "interrupt_decision: decision=%s thread=%s",
        decision,
        config.get("configurable", {}).get("thread_id"),
    )

    # ── Resume the graph ───────────────────────────────────────────────────────
    # Step 1: inject the decision into the checkpoint via update_state().
    #   This is DIFFERENT from passing a state dict to ainvoke/astream.
    #   With interrupt_before, ainvoke(state_dict) starts a NEW graph run
    #   from the entry node. update_state() writes directly to the checkpoint
    #   without triggering a run.
    await graph.aupdate_state(config, {"approval_decision": decision})

    # Step 2: resume from the paused position (before approval_check) with None.
    #   ainvoke(None) / astream(None) resumes the existing checkpoint run.
    #   approval_check_node then reads approval_decision from state.
    print()
    async for event in graph.astream(
        None,
        config=config,
        stream_mode="updates",
    ):
        for node_name, _node_output in event.items():
            _print_node_progress(node_name)

    return True


def _print_node_progress(node_name: str) -> None:
    """
    Prints a brief progress indicator when a node completes.

    Only shown for "interesting" nodes — internal LangGraph nodes
    like __start__ are filtered out. Keeps the UI clean.

    Args:
        node_name: The name of the node that just completed.
    """
    # Internal LangGraph nodes — not meaningful to the user
    _INTERNAL_NODES = {"__start__"}
    # Nodes that run silently — no progress indicator needed
    _SILENT_NODES = {"memory_retrieval", "memory_write"}

    if node_name in _INTERNAL_NODES or node_name in _SILENT_NODES:
        return

    _NODE_LABELS = {
        "classify_ticket":   "Classifying",
        "billing_node":      "Billing agent",
        "approval_check":    "Recording approval",
        "supervisor_node":   "Routing to agent",
        "kb_worker":         "Searching knowledge base",
        "web_worker":        "Searching web",
        "research_collect":  "Collecting research",
        "respond":           "Generating response",
    }
    label = _NODE_LABELS.get(node_name, node_name)
    print(f"  [{label}]", end="\r", flush=True)


def _print_sources(agent_results: list) -> None:
    """
    Prints KB and web sources used to answer a research ticket.
    Only shown when agent_results contains kb_worker or web_worker entries.
    """
    kb = next((r for r in agent_results if r.get("agent") == "kb_worker"), None)
    web = next((r for r in agent_results if r.get("agent") == "web_worker"), None)

    if not kb and not web:
        return

    print("\n  ── Sources ─────────────────────────────────────")

    if kb:
        doc_count = kb.get("doc_count", 0)
        print(f"  📄 Knowledge Base: {doc_count} internal doc(s) retrieved")

    if web:
        content = web.get("content", "")
        # Extract URLs from the formatted Tavily output (lines starting with "    URL: ")
        urls = [
            line.strip().replace("URL: ", "")
            for line in content.splitlines()
            if line.strip().startswith("URL:")
        ]
        if urls:
            print(f"  🌐 Web sources ({len(urls)} sites):")
            for url in urls[:5]:  # cap at 5
                print(f"      {url}")
        elif "[Search unavailable" in content:
            print("  🌐 Web search: unavailable (TAVILY_API_KEY not set)")
        elif "[Search failed" in content:
            print("  🌐 Web search: failed")
        else:
            print("  🌐 Web search: completed (no URLs extracted)")

    print("  " + "─" * 48)


def _print_banner(session_id: str, settings) -> None:
    """Prints the startup banner with session info."""
    print("\n" + "═" * 60)
    print("  Support Research Agent")
    print(f"  Provider : {settings.llm_provider}")
    print(f"  Session  : {session_id[:8]}...")
    print("  Streaming: enabled")
    print("  Type your support ticket. Ctrl+C to exit.")
    print("═" * 60 + "\n")


def main() -> None:
    """
    Synchronous entry point — wraps async run() for CLI use.

    Why asyncio.run() here and not at module level?
      asyncio.run() creates a new event loop and blocks until run() returns.
      Calling it at module level would block on import — breaks test discovery.
      Calling it in main() means it only runs when explicitly invoked.
    """
    asyncio.run(run())


if __name__ == "__main__":
    main()
