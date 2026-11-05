"""
Module: agents.nodes
Layer:  Graph
Owner:  agents/

Purpose:
    Production-grade LangGraph node functions for support-research-agent.
    Each node does exactly one thing (Single Responsibility Principle).
    All nodes follow the mandatory template from CLAUDE.md:
      - @with_retry decorator on every LLM-calling node
      - Router functions are PURE SYNC functions (never async)
      - Full Google-style docstring with Reads/Writes sections
      - Inputs extracted at top, never mid-function
      - logger.debug on entry, logger.info on result
      - Partial dict return — never full state
      - _prefixed pure helpers for all logic

    Provider note:
      - classify_ticket_node uses HuggingFace zero-shot (LOCAL, zero tokens)
      - respond_node uses Groq via get_research_llm() (LLM tokens consumed)
      Both imported exclusively from llm_provider — never instantiated directly.

Public API:
    classify_ticket_node  — HF zero-shot ticket pre-classification
    route_ticket          — pure router fn: reads ticket_type, returns route key
    memory_retrieval_node — long-term memory lookup before agent responds
    memory_write_node     — writes memory summary after agent responds
    billing_node          — stub: handles billing tickets (full impl in Module 8)
    technical_node        — stub: handles technical tickets
    general_node          — stub: handles general tickets
    research_node         — stub: handles research tickets (CRAG in Module 9)
    respond_node          — Groq LLM response generation

LangGraph Concepts demonstrated:
    - Nodes (Concept 4)
    - Conditional Edges (Concept 6) — route_ticket is the router function
    - RunnableConfig (Concept 20) — config accessed safely with .get()
"""
from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig  # noqa: TC002

from core.settings import get_settings
from llm_provider import get_research_llm, with_retry
from memory.store import get_memory_store
from state import AgentState, TicketType  # noqa: TC001

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
# Defined at module level — never hardcoded inside functions.
# This makes them easy to spot, change, and test.

_TICKET_LABELS: list[str] = ["billing", "technical", "general", "research"]
"""Candidate labels for zero-shot classification. Order matters — most common first."""

_CLASSIFICATION_THRESHOLD: float = 0.4
"""Minimum confidence to accept a classification. Below this → fall back to 'general'."""

_SYSTEM_PROMPT = (
    "You are a helpful customer support agent. "
    "Answer the user's question clearly and concisely. "
    "If you cannot help, say so directly and suggest escalation."
)
"""System prompt for the LLM respond node. Single source of truth."""


# ══════════════════════════════════════════════════════════════════════════════
# NODE 1: classify_ticket_node
# ══════════════════════════════════════════════════════════════════════════════

@with_retry
async def classify_ticket_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    """
    Classifies an incoming support ticket using HuggingFace zero-shot classification.

    Why HuggingFace zero-shot instead of an LLM?
      - Runs fully locally — zero API tokens consumed, zero cost
      - ~50ms latency vs ~500ms for an LLM call
      - Accurate enough for routing (BART-large-mnli, NLI-based)
      - Saves the powerful LLM budget for actual reasoning

    Reads:
        state["messages"]  — uses the last message as the ticket text

    Writes:
        state["ticket_type"]               — classification result
        state["classification_confidence"] — confidence score (0.0–1.0)

    Args:
        state:  Current AgentState. Read-only — never mutate directly.
        config: RunnableConfig carrying thread_id, user_id, and metadata.

    Returns:
        Partial state dict with ticket_type and classification_confidence.

    Raises:
        ValueError: If messages list is empty.
    """
    # ── 1. Extract inputs ─────────────────────────────────────────────────────
    messages = state.get("messages", [])
    if not messages:
        raise ValueError("classify_ticket_node: received empty messages list")

    ticket_text = messages[-1].content
    thread_id = _get_thread_id(config)

    logger.debug(
        "classify_ticket_node: thread=%s text_len=%d",
        thread_id,
        len(ticket_text),
    )

    # ── 2. Business logic ─────────────────────────────────────────────────────
    ticket_type, confidence = _classify_with_hf(ticket_text)

    logger.info(
        "classify_ticket_node: thread=%s ticket_type=%s confidence=%.2f",
        thread_id,
        ticket_type,
        confidence,
    )

    # ── 3. Return partial dict — only fields this node changes ────────────────
    return {
        "ticket_type": ticket_type,
        "classification_confidence": confidence,
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 2: respond_node
# ══════════════════════════════════════════════════════════════════════════════

@with_retry
async def respond_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    """
    Generates a response to the support ticket using the Groq LLM.

    Uses get_research_llm() — the most capable Groq model (llama-3.3-70b-versatile).
    This will be replaced in Module 8 with specialist sub-agents per ticket_type.

    Reads:
        state["messages"]    — full conversation history sent to LLM
        state["ticket_type"] — included in system prompt for context

    Writes:
        state["messages"]       — appends the AIMessage response (via add_messages)
        state["final_response"] — plain-text response for CLI display

    Streaming:
        When stream_mode="messages" is used, LangGraph intercepts every
        AIMessageChunk emitted during llm.astream() and forwards it to the
        caller before this node returns. main.py receives tokens word-by-word.
        This node is identical whether streaming is on or off — the caller
        controls stream_mode, not the node.

    Args:
        state:  Current AgentState. Read-only — never mutate directly.
        config: RunnableConfig carrying thread_id and metadata.

    Returns:
        Partial state dict with messages (AIMessage) and final_response.

    Raises:
        ValueError: If messages list is empty.
    """
    # ── 1. Extract inputs ─────────────────────────────────────────────────────
    messages = state.get("messages", [])
    if not messages:
        raise ValueError("respond_node: received empty messages list")

    ticket_type = state.get("ticket_type") or "general"
    thread_id = _get_thread_id(config)
    streaming_enabled = state.get("streaming_enabled", False)

    logger.debug(
        "respond_node: thread=%s ticket_type=%s msg_count=%d streaming=%s",
        thread_id,
        ticket_type,
        len(messages),
        streaming_enabled,
    )

    # ── 2. Business logic ─────────────────────────────────────────────────────
    # Include retrieved memories in the prompt if any exist.
    # Memories are plain strings — injected as context in the system message.
    retrieved_memories = state.get("retrieved_memories", [])
    llm = get_research_llm()
    prompt = _build_response_prompt(messages, ticket_type, retrieved_memories)

    if streaming_enabled:
        # astream() yields AIMessageChunks. LangGraph's "messages" stream mode
        # intercepts these chunks and forwards them to the caller token-by-token.
        # We collect all chunks to build the final complete response.
        chunks = []
        async for chunk in llm.astream(prompt, config=config):
            chunks.append(chunk.content)
        response_text = "".join(chunks)
    else:
        # ainvoke() returns a complete AIMessage in one shot.
        # Used when the caller uses stream_mode="updates" or ainvoke().
        response = await llm.ainvoke(prompt, config=config)
        response_text = response.content

    logger.info(
        "respond_node: thread=%s response_len=%d",
        thread_id,
        len(response_text),
    )

    # ── 3. Return partial dict ────────────────────────────────────────────────
    return {
        "messages": [AIMessage(content=response_text)],
        "final_response": response_text,
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 3: memory_retrieval_node
# ══════════════════════════════════════════════════════════════════════════════

async def memory_retrieval_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    """
    Retrieves relevant long-term memories for the current user before responding.

    Runs BEFORE the specialist agent nodes so they have full user context.
    Uses semantic search — finds memories by meaning, not exact key match.

    Why no @with_retry here?
      This node calls our own memory store, not an external LLM API.
      Transient rate-limit retries (what @with_retry handles) don't apply.
      If the store fails, we return an empty list — the agent degrades
      gracefully rather than blocking on a memory error.

    Reads:
        state["messages"]  — last message used as the semantic search query
        config["configurable"]["user_id"] — scopes search to this user only

    Writes:
        state["retrieved_memories"] — list of relevant memory texts (operator.add)

    Args:
        state:  Current AgentState.
        config: RunnableConfig — must contain user_id under "configurable".

    Returns:
        Partial state dict with retrieved_memories.
    """
    # ── 1. Extract inputs ─────────────────────────────────────────────────────
    messages = state.get("messages", [])
    user_id = _get_user_id(config)
    thread_id = _get_thread_id(config)

    if not messages:
        logger.debug("memory_retrieval_node: no messages — skipping search")
        return {"retrieved_memories": []}

    # Use the latest message as the search query.
    # This finds memories most relevant to what the user just said.
    query = messages[-1].content

    logger.debug(
        "memory_retrieval_node: thread=%s user=%s query_len=%d",
        thread_id,
        user_id,
        len(query),
    )

    # ── 2. Retrieve memories ──────────────────────────────────────────────────
    try:
        store = get_memory_store()
        memories = await store.asearch(user_id=user_id, query=query)
        memory_texts = [m["text"] for m in memories if m.get("text")]
    except Exception as e:
        # Memory failure must NEVER block the main ticket flow.
        # Return empty list — agent proceeds without memory context.
        logger.warning(
            "memory_retrieval_node: store search failed user=%s error=%s",
            user_id,
            e,
        )
        memory_texts = []

    logger.info(
        "memory_retrieval_node: thread=%s user=%s memories_found=%d",
        thread_id,
        user_id,
        len(memory_texts),
    )

    # ── 3. Return partial dict ────────────────────────────────────────────────
    # retrieved_memories uses operator.add reducer — these are APPENDED,
    # not replaced. Multiple retrieval passes accumulate memories.
    return {"retrieved_memories": memory_texts}


# ══════════════════════════════════════════════════════════════════════════════
# NODE 4: memory_write_node
# ══════════════════════════════════════════════════════════════════════════════

@with_retry
async def memory_write_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    """
    Writes a compressed memory of this conversation turn to long-term storage.

    Runs AFTER respond_node. Summarises the ticket + response into a single
    sentence and stores it under the user's namespace. Future conversations
    will retrieve this when the user discusses a related topic.

    Why @with_retry here (but not on memory_retrieval_node)?
      This node calls the LLM for summarisation — an external API call
      subject to rate limits. @with_retry handles transient failures.

    Reads:
        state["messages"]         — full conversation for summarisation
        state["final_response"]   — the agent's response to summarise
        state["ticket_type"]      — used as the memory key prefix

    Writes:
        state["audit_events"] — records that a memory was written

    Args:
        state:  Current AgentState.
        config: RunnableConfig — must contain user_id under "configurable".

    Returns:
        Partial state dict with audit_events.
    """
    # ── 1. Extract inputs ─────────────────────────────────────────────────────
    messages = state.get("messages", [])
    final_response = state.get("final_response", "")
    ticket_type = state.get("ticket_type") or "general"
    user_id = _get_user_id(config)
    thread_id = _get_thread_id(config)

    get_settings()

    # Only write a memory if we have enough conversation to summarise
    if not messages or not final_response:
        logger.debug(
            "memory_write_node: nothing to summarise thread=%s",
            thread_id,
        )
        return {}

    # Only write every N turns (config-driven — settings.memory_summarize_after_turns)
    # For now: always write after every turn (Module 5 keeps it simple).
    # Module 6 will gate this on turn count.
    logger.debug(
        "memory_write_node: thread=%s user=%s ticket_type=%s",
        thread_id,
        user_id,
        ticket_type,
    )

    # ── 2. Summarise the conversation into a memory ───────────────────────────
    try:
        summary = await _summarise_for_memory(messages, final_response)
    except Exception as e:
        # Summarisation failure must NEVER block the response being delivered.
        logger.warning(
            "memory_write_node: summarisation failed user=%s error=%s",
            user_id,
            e,
        )
        return {}

    # ── 3. Write to long-term memory ──────────────────────────────────────────
    # Key format: "{ticket_type}_{thread_id_prefix}"
    # Using thread_id prefix ensures multiple turns don't collide.
    memory_key = f"{ticket_type}_{thread_id[:8]}"

    try:
        store = get_memory_store()
        await store.astore(
            user_id=user_id,
            key=memory_key,
            text=summary,
            metadata={"ticket_type": ticket_type, "thread_id": thread_id},
        )
        logger.info(
            "memory_write_node: stored thread=%s user=%s key=%s",
            thread_id,
            user_id,
            memory_key,
        )
    except Exception as e:
        logger.warning(
            "memory_write_node: store write failed user=%s error=%s",
            user_id,
            e,
        )
        return {}

    # ── 4. Return partial dict ────────────────────────────────────────────────
    return {
        "audit_events": [{
            "node": "memory_write_node",
            "action": "memory_written",
            "user_id": user_id,
            "memory_key": memory_key,
            "thread_id": thread_id,
        }],
    }


# ══════════════════════════════════════════════════════════════════════════════
# PURE HELPER FUNCTIONS
# All prefixed with _ — not exported, not called by graph directly.
# Pure = no LLM calls, no state access. Easy to unit test.
# ══════════════════════════════════════════════════════════════════════════════

def _classify_with_hf(ticket_text: str) -> tuple[TicketType, float]:
    """
    Runs HuggingFace zero-shot classification on the ticket text.

    Uses the facebook/bart-large-mnli model via get_classifier().
    Falls back to "general" if the top score is below threshold.

    Args:
        ticket_text: Raw ticket text to classify.

    Returns:
        Tuple of (ticket_type, confidence_score).
        ticket_type is always a valid TicketType literal.
    """
    from llm_provider import get_classifier

    classifier = get_classifier()
    result = classifier(
        ticket_text,
        _TICKET_LABELS,
        hypothesis_template=(
            "This support request is about {}. "
            "Billing means invoices, charges, refunds, or payments. "
            "Technical means errors, bugs, API issues, or troubleshooting. "
            "General means account help, passwords, or basic how-to questions. "
            "Research means comparing options, understanding policies, or looking up information."
        ),
        multi_label=False,
    )

    top_label: str = result["labels"][0]
    top_score: float = result["scores"][0]

    # Validate label is a known TicketType before trusting it
    if top_label not in _TICKET_LABELS:
        logger.warning(
            "_classify_with_hf: unexpected label=%r score=%.2f, defaulting to 'general'",
            top_label,
            top_score,
        )
        return "general", 0.0

    # Below threshold → not confident enough, fall back to general
    if top_score < _CLASSIFICATION_THRESHOLD:
        logger.info(
            "_classify_with_hf: low confidence label=%r score=%.2f → 'general'",
            top_label,
            top_score,
        )
        return "general", top_score

    return top_label, top_score  # type: ignore[return-value]


def _build_response_prompt(
    messages: list,
    ticket_type: str,
    retrieved_memories: list[str] | None = None,
) -> list:
    """
    Builds the prompt list for the LLM respond node.

    Prepends a system message with context about ticket_type and any
    long-term memories retrieved for this user. Appends the full
    conversation history so the LLM has complete context.

    Pure function — no LLM calls, no I/O. Trivially unit-testable.

    Args:
        messages:           Full conversation history (list of BaseMessage).
        ticket_type:        Classification result — included in system prompt.
        retrieved_memories: Long-term memories from Module 5. Injected as
                            context into the system message. None = no memories.

    Returns:
        List of messages ready for llm.ainvoke() or llm.astream().
    """
    memory_context = ""
    if retrieved_memories:
        formatted = "\n".join(f"- {m}" for m in retrieved_memories)
        memory_context = f"\n\nRelevant context from past interactions:\n{formatted}"

    system = SystemMessage(
        content=(
            f"{_SYSTEM_PROMPT}\n\n"
            f"This is a {ticket_type} support ticket. "
            f"Tailor your response accordingly."
            f"{memory_context}"
        )
    )
    # System message first, then full conversation history
    return [system, *messages]


def _get_user_id(config: RunnableConfig) -> str:
    """
    Safely extracts user_id from RunnableConfig.

    Falls back to "anonymous" — memories are still stored under this namespace
    so the store never errors. "anonymous" memories accumulate across all
    unauthenticated users, which is acceptable for dev/test environments.

    In Module 10 (Security), JWT auth populates user_id before this node runs.

    Args:
        config: RunnableConfig passed to every node by LangGraph.

    Returns:
        user_id string, or "anonymous" if not set.
    """
    return config.get("configurable", {}).get("user_id", "anonymous")


async def _summarise_for_memory(
    messages: list,
    final_response: str,
) -> str:
    """
    Summarises a conversation turn into a single memory sentence.

    Uses get_fast_llm() — the cheapest/fastest model.
    Memory summaries don't need reasoning power, just compression.
    Keeping this in a separate pure-ish helper makes it easy to:
      - Unit test with a mock LLM
      - Swap compression strategies (Module 5, Concept 39)
      - Change the summary prompt without touching node logic

    Args:
        messages:       Full conversation history.
        final_response: The agent's final response text.

    Returns:
        A single sentence summarising what happened and what was resolved.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    from llm_provider import get_fast_llm

    llm = get_fast_llm()

    # Extract just the human messages for context (not the full LLM back-and-forth)
    user_messages = [m.content for m in messages if hasattr(m, "type") and m.type == "human"]
    ticket_summary = " ".join(user_messages[-3:])  # last 3 user messages max

    prompt = [
        SystemMessage(content=(
            "Summarise the following support interaction in ONE sentence. "
            "Include: what the customer asked, and what was resolved. "
            "Be specific — include any relevant details like amounts, "
            "account numbers, or error codes. Maximum 50 words."
        )),
        HumanMessage(content=(
            f"Customer: {ticket_summary}\n"
            f"Agent resolved: {final_response[:300]}"
        )),
    ]

    response = await llm.ainvoke(prompt)
    return response.content.strip()


def _get_thread_id(config: RunnableConfig) -> str:
    """
    Safely extracts thread_id from RunnableConfig.

    Uses .get() with a fallback — never KeyError.
    The "configurable" namespace is LangGraph's standard location for
    thread_id, user_id, and other per-invocation metadata.

    Args:
        config: RunnableConfig passed to every node by LangGraph.

    Returns:
        thread_id string, or "unknown" if not set.
    """
    return config.get("configurable", {}).get("thread_id", "unknown")


# ══════════════════════════════════════════════════════════════════════════════
# ROUTER FUNCTION — Concept 6: Conditional Edges
# ══════════════════════════════════════════════════════════════════════════════

# Valid route keys — must match the mapping dict in graph.py exactly.
# Defined here as a constant so graph.py and tests can import it.
ROUTE_KEYS: dict[str, str] = {
    "billing": "billing_node",
    "technical": "supervisor_node",    # Module 8: supervisor dispatches via registry
    "general": "supervisor_node",
    "research": "supervisor_node",
}
"""
Mapping from TicketType values to destination node names.

Module 8 change: technical, general, and research all route to supervisor_node.
The supervisor dispatches to the appropriate BaseAgent subclass via the registry.
Billing stays on its own path for HITL (billing_node → approval_check).
"""


def route_ticket(state: AgentState) -> str:
    """
    Router function for the conditional edge after classify_ticket_node.

    CRITICAL RULES (enforced here, tested in test_nodes.py):
      1. MUST be synchronous — never async def. LangGraph calls this
         synchronously. An async router returns a coroutine object
         (a truthy non-empty string) routing everything silently wrong.
      2. MUST return a key present in the mapping dict in graph.py.
         An unmatched key raises InvalidUpdateError at runtime.
      3. MUST be a pure function — reads state only, no I/O, no LLM.
         Any routing decision requiring an LLM belongs in a preceding node.

    Reads:
        state["ticket_type"] — set by classify_ticket_node

    Returns:
        str: One of "billing", "technical", "general", "research".
             Always returns "general" as the safe fallback.

    Type annotation:
        Declared as RouterFn in graph.py: route_ticket: RouterFn
        This makes mypy check the (AgentState) -> str contract.
    """
    ticket_type = state.get("ticket_type")

    if ticket_type not in ROUTE_KEYS:
        # Safe fallback — never let an unknown ticket_type crash the graph.
        # This handles: None (classify_ticket_node didn't run), or an unexpected
        # value that slipped past the HF classifier.
        logger.warning(
            "route_ticket: unexpected ticket_type=%r, routing to general",
            ticket_type,
        )
        return "general"

    logger.debug("route_ticket: ticket_type=%r → %r", ticket_type, ticket_type)
    return ticket_type  # type: ignore[return-value]


# ══════════════════════════════════════════════════════════════════════════════
# STUB HANDLER NODES — Concepts 6, expanded in Module 8
#
# These are intentionally minimal — each sets active_agent and adds an audit
# event, but delegates final response generation to respond_node.
# Full specialist implementations (BillingAgent, TechnicalAgent, etc.)
# are built in Module 8 using the BaseAgent ABC and Registry pattern.
# ══════════════════════════════════════════════════════════════════════════════

def billing_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    """
    Handles billing support tickets. Detects large refunds requiring human approval.

    WHY synchronous (not async)?
      This node makes NO LLM calls — only regex matching via pure functions.
      Rule: nodes without LLM calls should be sync.

    HITL design — WHY NOT interrupt() directly?
      LangGraph 1.2.5's interrupt() reads the `var_child_runnable_config`
      context variable. On Python 3.10, loop.run_in_executor() does NOT copy
      context variables to the thread pool (Python 3.11 added this). So
      var_child_runnable_config is None in the executor thread → RuntimeError.
      Even in async nodes, the context propagation is only set up on Python 3.11
      (see ASYNCIO_ACCEPTS_CONTEXT in langgraph/_internal/_runnable.py).

      Solution: state-based HITL using interrupt_before.
        1. This node sets pending_approval in state (signals HITL needed).
        2. route_after_billing() sees pending_approval → routes to approval_check.
        3. The graph is compiled with interrupt_before=["approval_check"].
        4. The graph pauses BEFORE approval_check runs.
        5. The caller reads state.values["pending_approval"] for the payload.
        6. On resume, the caller passes {"approval_decision": decision} as a
           state update — approval_check_node then reads and records it.

    Concept 53 — interrupt_before:
      Compile-time static interrupt. Pauses the graph before the named node
      runs. No Python version restrictions — no context var issues.

    Reads:
        state["messages"]  — scanned for refund amounts via regex

    Writes (normal path — amount < $500 or no amount):
        state["active_agent"]   → "billing"
        state["audit_events"]   → ticket_received event

    Writes (large refund path — amount >= $500):
        state["active_agent"]    → "billing"
        state["pending_approval"] → ApprovalRequest payload dict
        (no audit event here — approval_check_node records it after decision)

    Args:
        state:  Current AgentState.
        config: RunnableConfig with thread_id.

    Returns:
        Partial state dict.
    """
    from workflows.approval import (
        _extract_refund_amount,
        build_approval_payload,
        check_approval_required,
    )

    thread_id = _get_thread_id(config)
    messages = state.get("messages", [])
    ticket_text = messages[-1].content if messages else ""

    logger.info("billing_node: thread=%s text_len=%d", thread_id, len(ticket_text))

    # ── Detect refund amount ───────────────────────────────────────────────────
    refund_amount = _extract_refund_amount(ticket_text)

    if check_approval_required(refund_amount):
        payload = build_approval_payload(refund_amount, thread_id)  # type: ignore[arg-type]

        logger.warning(
            "billing_node: large_refund_detected amount=%.2f thread=%s — pending approval",
            refund_amount,
            thread_id,
        )

        # Signal HITL: route_after_billing() will route to approval_check.
        # The graph pauses BEFORE approval_check via interrupt_before.
        return {
            "active_agent": "billing",
            "pending_approval": dict(payload),
        }

    # ── Standard billing ticket — no approval needed ───────────────────────────
    return {
        "active_agent": "billing",
        "audit_events": [{
            "node": "billing_node",
            "action": "ticket_received",
            "thread_id": thread_id,
        }],
    }


def approval_check_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    """
    Records the operator's approval decision for a large refund request.

    This node runs AFTER the graph is resumed from the interrupt_before pause.
    At resume time, the caller passes {"approval_decision": "approved"|...} as
    a state update, so this node reads the operator's decision from state.

    Concept 53 — interrupt_before resume:
      The graph pauses before this node. The caller resumes by calling:
        graph.ainvoke({"approval_decision": decision}, config=config)
      The state update merges into the graph state before this node runs.
      This node then reads approval_decision and records the audit event.

    Reads:
        state["approval_decision"]  — "approved" | "rejected" | "escalated"
        state["pending_approval"]   — the original request payload (has amount)

    Writes:
        state["pending_approval"]  → None (cleared — decision received)
        state["audit_events"]      → refund_decision event with amount + decision

    Args:
        state:  Current AgentState (has approval_decision set by resume).
        config: RunnableConfig with thread_id.

    Returns:
        Partial state dict clearing pending_approval and recording audit event.
    """
    thread_id = _get_thread_id(config)
    approval_decision = state.get("approval_decision")
    pending = state.get("pending_approval") or {}
    amount = pending.get("amount", 0.0)

    logger.info(
        "approval_check_node: thread=%s amount=%.2f decision=%s",
        thread_id,
        amount,
        approval_decision,
    )

    return {
        "pending_approval": None,           # cleared — decision recorded
        "audit_events": [{
            "node": "approval_check_node",
            "action": "refund_decision",
            "thread_id": thread_id,
            "amount": amount,
            "decision": approval_decision,
        }],
    }


def route_after_billing(state: AgentState) -> str:
    """
    Router function after billing_node.

    If billing_node set pending_approval (large refund detected), route to
    approval_check so the graph can pause for human review via interrupt_before.
    Otherwise route directly to respond.

    Pure synchronous function — no I/O, no LLM calls.

    Reads:
        state["pending_approval"] — set by billing_node for large refunds

    Returns:
        "approval_check" if pending_approval is set, else "respond".
    """
    if state.get("pending_approval"):
        return "approval_check"
    return "respond"


@with_retry
async def technical_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    """
    Handles technical support tickets.

    Stub for Concept 6. Full implementation in Module 8:
      - KB retrieval for known issues (Module 9 RAG)
      - Escalation to L2 on unresolved issues (Module 7)

    Reads:  state["messages"], state["ticket_type"]
    Writes: state["active_agent"], state["audit_events"]
    """
    thread_id = _get_thread_id(config)
    logger.info("technical_node: thread=%s", thread_id)

    return {
        "active_agent": "technical",
        "audit_events": [{
            "node": "technical_node",
            "action": "ticket_received",
            "thread_id": thread_id,
        }],
    }


@with_retry
async def general_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    """
    Handles general/uncategorised support tickets.

    Stub for Concept 6. Acts as the catch-all route.
    Full implementation in Module 8.

    Reads:  state["messages"], state["ticket_type"]
    Writes: state["active_agent"], state["audit_events"]
    """
    thread_id = _get_thread_id(config)
    logger.info("general_node: thread=%s", thread_id)

    return {
        "active_agent": "general",
        "audit_events": [{
            "node": "general_node",
            "action": "ticket_received",
            "thread_id": thread_id,
        }],
    }


@with_retry
async def research_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    """
    Handles research tickets requiring web or knowledge-base lookup.

    Stub for Concept 6. Full implementation in Module 9:
      - CRAG over internal KB (FAISS + HF embeddings)
      - Tavily web search fallback when KB retrieval is insufficient

    Reads:  state["messages"], state["ticket_type"]
    Writes: state["active_agent"], state["query"], state["audit_events"]
    """
    thread_id = _get_thread_id(config)
    ticket_text = state.get("messages", [])
    query = ticket_text[-1].content if ticket_text else ""

    logger.info("research_node: thread=%s query_len=%d", thread_id, len(query))

    return {
        "active_agent": "research",
        "query": query,
        "audit_events": [{
            "node": "research_node",
            "action": "ticket_received",
            "thread_id": thread_id,
        }],
    }
