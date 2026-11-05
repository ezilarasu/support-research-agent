"""
Module: agents.supervisor
Layer:  Graph
Owner:  agents/

Purpose:
    Supervisor pattern (Concept 74): one node that dispatches to all
    non-billing specialist agents via the registry. Callers add new agents
    by registering them — the supervisor code never changes (OCP).

    Also owns the research fan-out workers:
      - kb_worker_node   — parallel KB search (stub; full CRAG in Module 9)
      - web_worker_node  — parallel web search via Tavily SearchTool
      - research_collect_node — fan-in: counts and logs parallel results

    Execution flow for non-billing tickets:
        classify_ticket → supervisor_node (dispatches to registry agent)
          ├── technical → TechnicalAgent.process() → respond
          ├── general   → GeneralAgent.process()   → respond
          └── research  → ResearchAgent.process()
                          route_after_supervisor → [Send(kb_worker), Send(web_worker)]
                          kb_worker ─┐
                          web_worker ─┴─→ research_collect → respond

    Concept 84 — Send API:
        route_after_supervisor returns a list[Send] for research tickets.
        Each Send(node_name, state_update) schedules an independent parallel
        execution. LangGraph dispatches them concurrently; both append to
        agent_results via the operator.add reducer.

    Concept 86 — Fan-In:
        Both kb_worker and web_worker write to agent_results.
        Because agent_results is Annotated[list[dict], operator.add], their
        results are ACCUMULATED, not replaced. research_collect_node
        reads the combined list.

Public API:
    supervisor_node       — dispatches to registry agent by ticket_type
    route_after_supervisor — routes research to Send fan-out; others to respond
    kb_worker_node        — parallel KB search worker
    web_worker_node       — parallel web search worker
    research_collect_node — fan-in node after parallel workers

LangGraph Concepts demonstrated:
    - Supervisor Pattern (Concept 74) — registry-driven dispatch, OCP
    - Worker Graphs (Concept 77)      — kb/web workers as graph nodes
    - Agent Coordination (Concept 82) — agent_results reducer aggregation
    - Send API (Concept 84)           — fan-out from route_after_supervisor
    - Fan-Out (Concept 85)            — kb_worker + web_worker parallel
    - Fan-In (Concept 86)             — operator.add on agent_results
    - Parallel Execution (Concept 87) — Groq rate limit implications noted
"""
from __future__ import annotations

import logging

from langchain_core.runnables import RunnableConfig  # noqa: TC002
from langgraph.types import Send

from agents.registry import get_agent
from llm_provider import with_retry
from state import AgentState  # noqa: TC001

logger = logging.getLogger(__name__)


def _get_thread_id(config: RunnableConfig) -> str:
    """Safely extracts thread_id from RunnableConfig."""
    return config.get("configurable", {}).get("thread_id", "unknown")


# ══════════════════════════════════════════════════════════════════════════════
# SUPERVISOR NODE — Concept 74: Supervisor Pattern
# ══════════════════════════════════════════════════════════════════════════════

@with_retry
async def supervisor_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    """
    Dispatcher: routes to the appropriate specialist agent via the registry.

    WHY a supervisor instead of direct routing from classify_ticket?
      The supervisor decouples routing from agent construction. classify_ticket
      uses a conditional edge to route by ticket_type — that works for a fixed
      set of routes. The supervisor adds a layer that delegates to registry
      agents, which means adding a new agent type requires only:
        1. Create a new BaseAgent subclass
        2. Add @register_agent("new_type") — that's it.
      The supervisor, the conditional edges, and all other nodes stay unchanged.
      This is the Open/Closed Principle applied to multi-agent systems.

    Reads:  state["ticket_type"]  — determines which agent to dispatch to

    Writes: whatever the dispatched agent.process() returns
            (active_agent, agent_results, audit_events, query, ...)

    Args:
        state:  Current AgentState. Read-only — passed to agent.process().
        config: RunnableConfig with thread_id, user_id, tenant_id.

    Returns:
        Partial state dict from the dispatched agent's process() method.

    Raises:
        RetryError: If the agent.process() call fails after all retry attempts.
    """
    ticket_type = state.get("ticket_type") or "general"
    thread_id = _get_thread_id(config)

    logger.info(
        "supervisor_node: thread=%s ticket_type=%s",
        thread_id,
        ticket_type,
    )

    try:
        agent = get_agent(ticket_type)
        logger.debug(
            "supervisor_node: dispatching to agent=%r",
            agent.agent_name,
        )
        result = await agent.process(state, config)
        logger.debug(
            "supervisor_node: agent=%s completed keys=%s",
            agent.agent_name,
            list(result.keys()),
        )
        return result

    except KeyError:
        # No agent registered for this ticket_type — safe fallback to general.
        # This should not happen in production if all TicketType values are
        # registered. Log a warning so it's visible in traces.
        logger.warning(
            "supervisor_node: no agent registered for ticket_type=%r "
            "— falling back to general",
            ticket_type,
        )
        fallback_agent = get_agent("general")
        return await fallback_agent.process(state, config)


# ══════════════════════════════════════════════════════════════════════════════
# ROUTER — Concept 84: Send API for research fan-out
# ══════════════════════════════════════════════════════════════════════════════

def route_after_supervisor(
    state: AgentState,
):
    """
    Routes after supervisor_node:
      - research tickets → [Send(kb_worker), Send(web_worker)] for parallel execution
      - all other tickets → "respond" (LLM response generation)

    CRITICAL: This function must be SYNCHRONOUS.
      LangGraph calls conditional edge functions synchronously.
      An async router returns a coroutine object (truthy non-string) and
      silently misroutes everything.

    The Send API (Concept 84):
      Returning a list of Send objects schedules parallel node executions.
      Each Send(node_name, state_override) creates an independent graph branch.
      Both branches run concurrently; their results are merged by the reducer
      (operator.add on agent_results) before research_collect_node runs.

    Rate limit warning (Concept 87 — Parallel Execution):
      If kb_worker and web_worker both make Groq LLM calls, they consume
      tokens concurrently. On Groq free tier this can trigger rate limits.
      Module 8: kb_worker is a stub (no LLM calls), so only web_worker
      may call external APIs. This is safe on free tier.
      Module 9: add staggered retry with jitter if both workers use LLMs.

    Reads:  state["active_agent"]  — "research" triggers fan-out
            state["query"]         — passed to both workers via Send update

    Returns:
        list[Send]: Parallel worker dispatch (research tickets)
        str "respond": Direct route to response generation (others)
    """
    active_agent = state.get("active_agent")

    if active_agent == "research":
        query = state.get("query") or ""
        logger.debug(
            "route_after_supervisor: research fan-out query_len=%d",
            len(query),
        )
        # Each Send schedules an independent node execution.
        # The state_override {"query": query} ensures workers have the query
        # even if state["query"] wasn't set before supervisor_node ran.
        return [
            Send("kb_worker", {"query": query}),
            Send("web_worker", {"query": query}),
        ]

    logger.debug(
        "route_after_supervisor: active_agent=%r → respond",
        active_agent,
    )
    return "respond"


# ══════════════════════════════════════════════════════════════════════════════
# PARALLEL WORKERS — Concepts 85–86: Fan-Out + Fan-In
# ══════════════════════════════════════════════════════════════════════════════

@with_retry
async def kb_worker_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    """
    Parallel worker: searches the internal knowledge base.

    Fan-out worker (Concept 85): runs concurrently with web_worker_node.
    Writes to agent_results via operator.add reducer — safe for fan-in.

    Module 9 implementation:
      - FAISS similarity_search(query, k=20) via make_retriever_node()
      - Cross-encoder reranking to top rag_top_k_rerank docs
      - Populates retrieved_docs in state for respond_node / research_collect

    Why @with_retry here (not only on supervisor)?
      Workers are dispatched directly by Send API — they bypass supervisor_node.
      Each worker needs its own retry wrapper for resilience.

    Reads:  state["query"]  — passed by route_after_supervisor via Send

    Writes: state["agent_results"]  — appends KB search result (operator.add)
            state["audit_events"]   — appends kb_searched event (operator.add)

    Args:
        state:  Current AgentState (includes Send state override).
        config: RunnableConfig with thread_id.

    Returns:
        Partial state dict with agent_results and audit_events.
    """
    query = state.get("query") or ""
    thread_id = _get_thread_id(config)

    logger.info(
        "kb_worker_node: thread=%s query_len=%d",
        thread_id,
        len(query),
    )

    # ── Module 9: FAISS retrieval + cross-encoder reranking ───────────────────
    retrieved_docs: list = []
    content_summary = ""
    doc_count = 0

    if query:
        try:
            from rag.retriever import make_retriever_node

            # make_retriever_node() returns an async node function.
            # We call it directly with a minimal state to get the retrieval result.
            retriever_fn = make_retriever_node()
            result = await retriever_fn(state, config)
            retrieved_docs = result.get("retrieved_docs", [])
            doc_count = len(retrieved_docs)

            # Build a context summary from top-3 docs for agent_results
            if retrieved_docs:
                snippets = [
                    f"[{doc.get('source', 'kb')}] {doc['content'][:200]}"
                    for doc in retrieved_docs[:3]
                ]
                content_summary = "\n\n".join(snippets)
                logger.info(
                    "kb_worker_node: thread=%s retrieved doc_count=%d",
                    thread_id,
                    doc_count,
                )
        except Exception as e:
            logger.warning(
                "kb_worker_node: retrieval failed thread=%s error=%s",
                thread_id,
                e,
            )

    return {
        "retrieved_docs": retrieved_docs,
        "agent_results": [{
            "agent": "kb_worker",
            "source": "knowledge_base",
            "query": query,
            "content": content_summary,
            "doc_count": doc_count,
        }],
        "audit_events": [{
            "node": "kb_worker_node",
            "action": "kb_searched",
            "thread_id": thread_id,
            "query_len": len(query),
            "doc_count": doc_count,
        }],
    }


@with_retry
async def web_worker_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    """
    Parallel worker: searches the web via Tavily.

    Fan-out worker (Concept 85): runs concurrently with kb_worker_node.
    Falls back gracefully when TAVILY_API_KEY is not set.

    Why Tavily here (not in ResearchAgent.process)?
      Separation of concerns: ResearchAgent decides WHAT to search,
      web_worker decides HOW to search the web. This means we can swap
      Tavily for another provider by only changing this worker, not the agent.

    Reads:  state["query"]  — passed by route_after_supervisor via Send

    Writes: state["agent_results"]  — appends web search result (operator.add)
            state["audit_events"]   — appends web_searched event (operator.add)

    Args:
        state:  Current AgentState (includes Send state override).
        config: RunnableConfig with thread_id.

    Returns:
        Partial state dict with agent_results and audit_events.
    """
    query = state.get("query") or ""
    thread_id = _get_thread_id(config)

    logger.info(
        "web_worker_node: thread=%s query_len=%d",
        thread_id,
        len(query),
    )

    # ── Web search ────────────────────────────────────────────────────────────
    # SearchTool._run() returns a placeholder if TAVILY_API_KEY is not set.
    # This means web_worker always returns a result — never raises.
    # Tests don't need a Tavily key: they get the placeholder, which is fine.
    search_content = ""
    if query:
        try:
            from tools.search_tool import SearchTool
            search_content = SearchTool()._run(query=query, max_results=5)
        except Exception as e:
            logger.warning(
                "web_worker_node: search failed query_len=%d error=%s",
                len(query),
                e,
            )
            search_content = f"[Search failed: {e}]"

    return {
        "agent_results": [{
            "agent": "web_worker",
            "source": "web_search",
            "query": query,
            "content": search_content,
        }],
        "audit_events": [{
            "node": "web_worker_node",
            "action": "web_searched",
            "thread_id": thread_id,
            "query_len": len(query),
        }],
    }


async def research_collect_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    """
    Fan-in: collects results from parallel research workers.

    Why a dedicated collect node?
      After Send API fan-out, both kb_worker and web_worker write to
      agent_results. LangGraph merges their state updates via operator.add
      before this node runs. research_collect_node provides a single
      synchronisation point: all parallel work is done by the time this
      node executes.

    In Module 9 this node will synthesize KB + web results into a
    structured context that respond_node uses for grounded answering.
    For Module 8: just counts and logs results.

    Why no @with_retry?
      This node makes no external calls — it only reads state.
      Retry semantics don't apply to pure state reads.

    Reads:  state["agent_results"]  — populated by kb_worker + web_worker

    Writes: state["audit_events"]   — records collection complete

    Args:
        state:  Current AgentState (merged fan-in results available).
        config: RunnableConfig with thread_id.

    Returns:
        Partial state dict with audit_events.
    """
    thread_id = _get_thread_id(config)
    agent_results = state.get("agent_results", [])

    # Count results from the research workers in this run
    research_sources = [
        r for r in agent_results
        if r.get("agent") in ("kb_worker", "web_worker")
    ]

    logger.info(
        "research_collect_node: thread=%s sources=%d",
        thread_id,
        len(research_sources),
    )

    # ── Module 9 extension ────────────────────────────────────────────────────
    # In Module 9, this node will:
    #   1. Combine KB docs + web results into a unified context string
    #   2. Run hallucination grader on the combined context
    #   3. Store context in state["retrieved_docs"] for respond_node
    # For now: record that collection is complete.

    return {
        "audit_events": [{
            "node": "research_collect_node",
            "action": "results_collected",
            "thread_id": thread_id,
            "source_count": len(research_sources),
        }],
    }
