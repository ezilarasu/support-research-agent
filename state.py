"""
Module: state
Layer:  Infrastructure (shared contract)
Owner:  root — imported by every layer

Purpose:
    Defines AgentState — the single shared memory schema for the entire graph.
    Every node reads from and writes to this TypedDict.
    This is the contract that makes the graph type-safe end-to-end.

    Fields are added incrementally as modules are taught.
    Each field is commented with the Module that introduces it so you can
    track exactly when each piece of state becomes active.

    Design rules enforced here:
      - List fields that are APPENDED use Annotated[list[T], reducer]
      - Scalar fields that are REPLACED use plain type annotations
      - Never use raw dict — define a TypedDict for structured sub-objects
      - Nodes return PARTIAL dicts — they only write their own fields

Public API:
    AgentState         — main state TypedDict for StateGraph(AgentState)
    TicketMetadata     — structured ticket metadata sub-object
    RetrievedDocument  — structured RAG result sub-object

LangGraph Concepts demonstrated:
    - State (Concept 2)
    - TypedDict (Concept 3)
    - Reducers + Annotated (Concepts 9–12, Module 2)
    - add_messages reducer (Concept 12)
    - operator.add reducer (Concept 11)
"""
from __future__ import annotations

import operator
from collections.abc import Callable
from enum import Enum
from typing import Annotated, Literal, TypeAlias, TypedDict

from langchain_core.messages import (
    BaseMessage,  # noqa: TC002 — needed at runtime for get_type_hints
)
from langgraph.graph.message import add_messages

# ── TypeAliases ───────────────────────────────────────────────────────────────
# Define once here. Import from state everywhere else.
# Avoids repeating Literal["billing", "technical", ...] across 10+ files.

TicketType: TypeAlias = Literal["billing", "technical", "general", "research"]
"""Valid ticket classification values. Used by classifier and router nodes."""


# ── Enums ─────────────────────────────────────────────────────────────────────

class TicketStatus(str, Enum):
    """
    Lifecycle state of a support ticket — Concept 65: Workflow State Machines.

    Using str-Enum means values serialise to JSON as plain strings.
    This makes them checkpoint-safe (SQLite stores "pending", not TicketStatus.PENDING).

    State machine transitions:
      PENDING → IN_PROGRESS (ticket assigned to agent)
      IN_PROGRESS → AWAITING_APPROVAL (large refund detected)
      IN_PROGRESS → ESCALATED (threshold/error triggered)
      AWAITING_APPROVAL → APPROVED | REJECTED | ESCALATED (human decides)
      APPROVED | REJECTED → RESOLVED (response sent)
      ESCALATED → IN_PROGRESS (at higher level) | RESOLVED

    Not added to AgentState directly — used by workflow nodes that need
    to track ticket lifecycle explicitly. Import from state.py wherever needed.
    """
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    ESCALATED = "escalated"
    RESOLVED = "resolved"

AgentName: TypeAlias = Literal["billing", "technical", "general", "research", "supervisor"]
"""Valid agent names for routing and the active_agent field."""

GradeLabel: TypeAlias = Literal["relevant", "irrelevant", "partial"]
"""Retrieval grader verdict. Used in retrieval_grade field."""

ApprovalDecision: TypeAlias = Literal["approved", "rejected", "escalated"]
"""HITL operator decision. Used in approval_decision field."""

Priority: TypeAlias = Literal["low", "medium", "high", "critical"]
"""Ticket priority levels. Used in TicketMetadata."""

RouterFn: TypeAlias = Callable[["AgentState"], str]
"""
Type signature for all LangGraph router functions.

A router is a PURE SYNCHRONOUS function — no async, no LLM calls, no I/O.
It reads state and returns a string key from the conditional_edge mapping.

IMPORTANT: Never make a router async. LangGraph calls it synchronously.
An async router returns a coroutine object (truthy) instead of a string,
silently routing everything to the wrong branch with no error raised.

Usage:
    def route_ticket(state: AgentState) -> str:   # ✅ sync
        return state["ticket_type"] or "general"

    async def route_ticket(state: AgentState) -> str:  # ❌ never async
        ...
"""


# ── Public API ────────────────────────────────────────────────────────────────
__all__ = [
    "AgentState",
    "TicketMetadata",
    "RetrievedDocument",
    # TypeAliases — exported so other modules import from state, not redefine
    "TicketType",
    "AgentName",
    "GradeLabel",
    "ApprovalDecision",
    "Priority",
    "RouterFn",
]


# ── Sub-object TypedDicts ─────────────────────────────────────────────────────
# These are used as field types inside AgentState.
# Using TypedDict instead of raw dict gives us type safety on nested data.

class TicketMetadata(TypedDict):
    """
    Structured metadata attached to every incoming support ticket.

    Populated by the auth/extraction node in Module 10.
    Until then, ticket_metadata in AgentState is None.

    Why a TypedDict instead of dict[str, str]?
      - Autocomplete: state["ticket_metadata"]["tenant_id"] works in IDE
      - mypy: catches "tenenat_id" typos at analysis time
      - Documentation: the fields self-document what metadata exists
    """
    ticket_id: str
    tenant_id: str
    user_id: str
    priority: Priority
    created_at: str     # ISO 8601 datetime string
    sla_deadline: str   # ISO 8601 datetime string


class RetrievedDocument(TypedDict):
    """
    A single document retrieved from the RAG knowledge base.

    Populated by the retriever node in Module 9.
    The retrieved_docs field in AgentState holds a list of these.
    """
    content: str
    source: str     # filename or URL the chunk came from
    score: float    # similarity or reranker score (higher = more relevant)
    metadata: dict  # raw metadata from the vector store (page, section, etc.)


# ── Main state schema ─────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """
    Central state schema for support-research-agent.

    This TypedDict is the shared whiteboard for the entire graph.
    Every node receives the current state and returns a PARTIAL dict
    containing only the fields it modified.

    CRITICAL RULE — Annotated fields with reducers:
      fields typed as  Annotated[list[T], some_reducer]
      are MERGED on update, not replaced.
      fields typed as  plain types (str, bool, etc.)
      are REPLACED on update.

      This is explained in depth in Module 2 (Concepts 9–12).
      For now: if a field is a list that multiple nodes append to,
      it uses Annotated. If a scalar that only one node sets, plain type.

    Fields are grouped by the Module that introduces them.
    Fields from future modules are included now so state.py is
    the single complete source of truth from day one.
    """

    # ── Module 1: Core ────────────────────────────────────────────────────────

    messages: Annotated[list[BaseMessage], add_messages]
    """
    Full conversation history as LangChain message objects.

    add_messages reducer: when two nodes both write to 'messages',
    LangGraph APPENDS and deduplicates by message ID — never overwrites.
    This is safe for HITL resume: replaying a checkpoint won't duplicate messages.

    HumanMessage, AIMessage, SystemMessage, ToolMessage all go here.
    Taught in depth in Concept 12.
    """

    # ── Module 2 / 3: Routing ─────────────────────────────────────────────────

    ticket_type: TicketType | None
    """
    Classification result: "billing" | "technical" | "general" | "research".
    None until classify_ticket_node runs.
    Scalar field — replaced on each update (only one node writes it).
    Using TicketType alias: mypy catches "billin" typos in routing functions.
    """

    ticket_metadata: TicketMetadata | None
    """
    Structured ticket metadata. Populated by auth/extraction nodes in Module 10.
    None in dev mode (default_tenant_id from settings is used instead).
    """

    classification_confidence: float
    """
    Confidence score from the HF zero-shot classifier (0.0–1.0).
    Low confidence (<0.5) triggers a fallback to "general" routing.
    """

    # ── Module 4: Persistence ─────────────────────────────────────────────────

    thread_id: str | None
    """
    LangGraph thread identifier. Set by the caller via config, not by nodes.
    Used as the primary key for checkpoint storage (SQLite / Postgres).
    Carries through to every log line as a correlation ID.
    """

    # ── Module 5: Long-Term Memory ────────────────────────────────────────────

    retrieved_memories: Annotated[list[str], operator.add]
    """
    Long-term memories retrieved for this turn (semantic similarity search).
    operator.add reducer: each memory retrieval node APPENDS to this list.
    Multiple retrieval passes in one turn are safe — results accumulate.
    """

    # ── Module 6: Streaming ───────────────────────────────────────────────────

    streaming_enabled: bool
    """
    True = stream tokens to stdout as they are generated.
    False = buffer complete response (default for tests).
    Set by the delivery layer (main.py), not by nodes.
    """

    # ── Module 7: HITL ────────────────────────────────────────────────────────

    pending_approval: dict | None
    """
    Payload for a HITL approval request. Structure:
      {"action": "refund", "amount": 49.99, "reason": "...", "ticket_id": "..."}
    Set by the billing node when a refund requires human sign-off.
    Cleared (set to None) once the approval_decision is recorded.
    """

    approval_decision: ApprovalDecision | None
    """
    Human operator's decision: "approved" | "rejected" | "escalated".
    Set by the HITL resume step after the operator responds.
    Using ApprovalDecision alias: approval routing logic is mypy-checked.
    None while pending_approval is waiting.
    """

    escalation_level: int
    """
    Current escalation level. 0 = L1 (frontline), 1 = L2, 2 = L3 (specialist).
    Incremented by the escalation node when SLA is breached or L1 cannot resolve.
    """

    # ── Module 8: Multi-Agent ─────────────────────────────────────────────────

    active_agent: AgentName | None
    """
    Name of the sub-agent currently handling the ticket.
    Set by the supervisor node when it routes to a worker.
    Using AgentName alias: routing functions get mypy-checked against valid names.
    """

    agent_results: Annotated[list[dict], operator.add]
    """
    Results collected from worker agents during fan-out execution.
    operator.add reducer: each worker APPENDS its result — safe for parallel runs.
    The supervisor aggregates these in the fan-in step.
    """

    # ── Module 9: RAG ─────────────────────────────────────────────────────────

    query: str | None
    """
    The current search query used for RAG retrieval.
    May differ from the original ticket text (query rewriting node rewrites it
    to be more retrieval-friendly in Concept 91).
    """

    retrieved_docs: Annotated[list[RetrievedDocument], operator.add]
    """
    Documents retrieved from the knowledge base.
    operator.add reducer: multi-query retrieval (Concept 92) runs multiple
    retrieval passes and appends results from each — no data loss.
    """

    retrieval_grade: GradeLabel | None
    """
    Verdict from the retrieval grader node: "relevant" | "irrelevant" | "partial".
    "irrelevant" triggers the Tavily web search fallback in CRAG.
    Using GradeLabel alias: conditional edge router functions get mypy-checked.
    """

    hallucination_check: bool | None
    """
    Verdict from the hallucination grader: True = grounded, False = hallucinated.
    False triggers a regeneration loop in Self-RAG (Concept 96).
    """

    # ── Module 10: Security ───────────────────────────────────────────────────

    authenticated: bool
    """
    True once the JWT auth node has verified the caller's token.
    The auth guard node sets this at graph entry — all subsequent nodes
    trust it. Never set to True by any node other than auth.
    """

    rbac_roles: Annotated[list[str], operator.add]
    """
    Roles granted to the authenticated user for this session.
    e.g. ["agent", "billing_read"] or ["admin", "billing_write"]
    operator.add reducer: role assignment nodes append — safe for
    multi-step role resolution (base roles + tenant-specific roles).
    """

    audit_events: Annotated[list[dict], operator.add]
    """
    Immutable audit log entries. APPEND ONLY — never overwrite or delete.
    operator.add reducer enforces this: every sensitive node appends one entry.
    Persisted in the checkpoint — full audit trail survives process restarts.

    Entry structure:
      {"node": "billing_agent", "action": "refund_requested",
       "user_id": "u-123", "tenant_id": "acme", "ts": "2026-06-17T10:00:00Z"}
    """

    # ── Output ────────────────────────────────────────────────────────────────

    final_response: str | None
    """
    The formatted response ready to stream to the user.
    Set by the final formatting node. None until that node runs.
    """

    error: str | None
    """
    Error description if a node failed. None during normal execution.
    Set by error-handler nodes — the graph routes to a graceful error
    response instead of propagating the exception to the user.
    """
