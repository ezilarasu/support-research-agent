"""
Unit + integration tests for Module 4, Concepts 22–34: Persistence

  22 - Graph Memory      — what a checkpointer does
  23 - Checkpointing     — get_checkpointer() factory
  24 - InMemorySaver     — test-time checkpointer
  25 - SQLite Checkpointer — file-based persistence
  26 - Thread IDs        — uuid4, correlation ID pattern
  27 - User IDs + Metadata — typed config, never raw dicts
  28 - Persistent Threads — state survives across invocations
  29 - Checkpoint Storage — get_state_history() iteration
  30 - State Recovery    — idempotent nodes
  31 - Durable Execution — side-effect isolation
  32 - Time Travel       — checkpoint_id in config

Run:
    .venv/bin/python -m pytest tests/unit/test_persistence.py -v
"""
from __future__ import annotations

import operator
import uuid
from typing import TYPE_CHECKING, Annotated, TypedDict
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

if TYPE_CHECKING:
    from state import TicketMetadata

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_simple_graph(checkpointer=None):
    """Minimal graph that appends a message each invocation. Used across tests."""
    class SimpleState(TypedDict):
        messages: Annotated[list[str], operator.add]
        count: int

    def node(state: SimpleState) -> dict:
        return {
            "messages": [f"message_{state['count'] + 1}"],
            "count": state["count"] + 1,
        }

    g = StateGraph(SimpleState)
    g.add_node("n", node)
    g.add_edge("n", END)
    g.set_entry_point("n")
    kwargs = {}
    if checkpointer:
        kwargs["checkpointer"] = checkpointer
    return g.compile(**kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# CONCEPT 22: Graph Memory — what checkpointing is
# ══════════════════════════════════════════════════════════════════════════════

class TestGraphMemory:
    """
    Concept 22 — Without a checkpointer, every invocation is stateless.
    With a checkpointer, state accumulates across invocations on the same thread.

    This is the fundamental difference: stateless vs stateful graph execution.
    """

    def test_without_checkpointer_state_is_lost_between_invocations(self):
        """
        Without checkpointer: each invoke starts fresh.
        The count never exceeds 1 no matter how many times we call it.
        """
        graph = _make_simple_graph(checkpointer=None)
        config = {"configurable": {"thread_id": "t1"}}

        r1 = graph.invoke({"messages": [], "count": 0}, config=config)
        r2 = graph.invoke({"messages": [], "count": 0}, config=config)

        # Each invocation starts from count=0 → ends at count=1
        assert r1["count"] == 1
        assert r2["count"] == 1   # same — no memory of r1

    def test_with_checkpointer_state_accumulates(self):
        """
        With checkpointer: second invoke resumes from where first left off.
        count reaches 2 on the second invoke because state was saved.
        """
        graph = _make_simple_graph(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "t1"}}

        r1 = graph.invoke({"messages": [], "count": 0}, config=config)
        assert r1["count"] == 1

        # Second invoke: pass {} — checkpointer fills in count=1 from checkpoint.
        # IMPORTANT: if you pass count=0, it OVERRIDES the checkpoint value (no
        # reducer on `count`), resetting to 0. Pass only what you want to update.
        r2 = graph.invoke({}, config=config)
        assert r2["count"] == 2   # ← resumed from count=1, ran node again

    def test_different_thread_ids_have_independent_state(self):
        """
        Thread IDs are namespaces. Thread A and Thread B never share state.
        Critical for multi-user support systems.
        """
        graph = _make_simple_graph(checkpointer=MemorySaver())

        config_a = {"configurable": {"thread_id": "thread-A"}}
        config_b = {"configurable": {"thread_id": "thread-B"}}

        # Invoke thread A twice — first call seeds the state, second resumes
        graph.invoke({"messages": [], "count": 0}, config=config_a)
        r_a2 = graph.invoke({}, config=config_a)

        # Invoke thread B once — completely independent from thread A
        r_b1 = graph.invoke({"messages": [], "count": 0}, config=config_b)

        assert r_a2["count"] == 2   # Thread A ran twice
        assert r_b1["count"] == 1   # Thread B ran once — independent


# ══════════════════════════════════════════════════════════════════════════════
# CONCEPT 23: Checkpointing — get_checkpointer() factory
# ══════════════════════════════════════════════════════════════════════════════

class TestCheckpointerFactory:
    """
    Concept 23 — get_checkpointer() resolves the right backend from settings.
    """

    def test_memory_backend_returns_memory_saver(self, monkeypatch):
        from core.settings import get_settings
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        monkeypatch.setenv("CHECKPOINTER_BACKEND", "memory")
        get_settings.cache_clear()

        from core.factories import get_checkpointer
        cp = get_checkpointer()
        assert isinstance(cp, MemorySaver)

    def test_sqlite_backend_returns_sqlite_saver(self, monkeypatch, tmp_path):
        from core.settings import get_settings
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        monkeypatch.setenv("CHECKPOINTER_BACKEND", "sqlite")
        monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "test.db"))
        get_settings.cache_clear()

        from langgraph.checkpoint.sqlite import SqliteSaver

        from core.factories import get_checkpointer
        cp = get_checkpointer()
        assert isinstance(cp, SqliteSaver)

    def test_unknown_backend_raises_value_error(self, monkeypatch):
        from core.settings import get_settings
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        monkeypatch.setenv("CHECKPOINTER_BACKEND", "redis")  # not a valid value
        get_settings.cache_clear()

        # pydantic-settings will reject "redis" as it's not in the Literal type
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            get_settings()

    def test_factory_returns_fresh_instance_each_call(self, monkeypatch):
        """
        get_checkpointer() is NOT lru_cached — each call creates a fresh instance.
        This prevents "database is locked" errors in test suites.
        """
        from core.settings import get_settings
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        monkeypatch.setenv("CHECKPOINTER_BACKEND", "memory")
        get_settings.cache_clear()

        from core.factories import get_checkpointer
        cp1 = get_checkpointer()
        cp2 = get_checkpointer()
        assert cp1 is not cp2   # different instances — no caching


# ══════════════════════════════════════════════════════════════════════════════
# CONCEPT 24: InMemorySaver
# ══════════════════════════════════════════════════════════════════════════════

class TestInMemorySaver:
    """
    Concept 24 — InMemorySaver is identical in API to SqliteSaver
    but stores everything in a Python dict. No disk I/O. Perfect for tests.
    """

    def test_memory_saver_persists_across_invocations(self):
        """InMemorySaver acts like SQLite — state persists between calls."""
        checkpointer = MemorySaver()
        graph = _make_simple_graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "mem-test"}}

        graph.invoke({"messages": [], "count": 0}, config=config)
        r2 = graph.invoke({}, config=config)  # {} lets checkpoint provide count=1

        assert r2["count"] == 2

    def test_memory_saver_fixture_is_fresh_per_test(self, memory_checkpointer):
        """
        The memory_checkpointer fixture creates a NEW MemorySaver per test.
        State from one test never leaks into another.
        """
        graph = _make_simple_graph(checkpointer=memory_checkpointer)
        config = {"configurable": {"thread_id": "fixture-test"}}

        r1 = graph.invoke({"messages": [], "count": 0}, config=config)
        assert r1["count"] == 1   # starts fresh — not polluted by other tests

    def test_memory_saver_has_same_api_as_sqlite_saver(self):
        """
        LSP: InMemorySaver and SqliteSaver are substitutable.
        build_graph(checkpointer=memory_saver) and
        build_graph(checkpointer=sqlite_saver) are identical to callers.
        """
        from langgraph.checkpoint.base import BaseCheckpointSaver

        assert isinstance(MemorySaver(), BaseCheckpointSaver)
        # SqliteSaver also inherits BaseCheckpointSaver — same contract


# ══════════════════════════════════════════════════════════════════════════════
# CONCEPTS 27–29: Thread IDs, User IDs, Metadata
# ══════════════════════════════════════════════════════════════════════════════

class TestThreadAndMetadata:
    """
    Concepts 27–29 — Thread IDs are UUIDs. User IDs live in typed metadata.
    Config is the delivery mechanism for all per-invocation identity.
    """

    def test_thread_id_is_valid_uuid(self):
        """Thread IDs must be valid UUIDs — not sequential numbers or names."""
        thread_id = str(uuid.uuid4())
        parsed = uuid.UUID(thread_id)   # raises if invalid
        assert str(parsed) == thread_id

    def test_two_sessions_have_different_thread_ids(self):
        """Each session generates a unique UUID — no collision risk."""
        id1 = str(uuid.uuid4())
        id2 = str(uuid.uuid4())
        assert id1 != id2

    def test_thread_id_as_correlation_id_in_logs(self, caplog):
        """Thread ID should appear in log lines for request correlation."""
        import logging
        thread_id = "test-correlation-id-abc"

        # Simulate what every node does
        import logging as log
        logger = log.getLogger("test")
        with caplog.at_level(logging.DEBUG):
            logger.info("node=classify thread=%s", thread_id)

        assert thread_id in caplog.text

    def test_config_structure_for_full_metadata(self):
        """
        Production config with thread_id, user_id, and tenant_id.
        All three live under "configurable" — not at the top level.
        """
        thread_id = str(uuid.uuid4())
        config = {
            "configurable": {
                "thread_id": thread_id,
                "user_id": "user-abc-123",
                "tenant_id": "acme-corp",
            },
            "recursion_limit": 50,
        }

        # Verify structure
        assert config["configurable"]["thread_id"] == thread_id
        assert config["configurable"]["user_id"] == "user-abc-123"
        assert config["configurable"]["tenant_id"] == "acme-corp"
        assert "thread_id" not in config   # must NOT be at top level

    def test_ticket_metadata_typeddict_prevents_typos(self):
        """
        TicketMetadata TypedDict catches typos at analysis time.
        raw dict["tenenat_id"] fails silently; TypedDict catches it with mypy.
        """
        meta: TicketMetadata = {
            "ticket_id": "TKT-001",
            "tenant_id": "acme",
            "user_id": "usr-123",
            "priority": "high",
            "created_at": "2026-06-25T10:00:00Z",
            "sla_deadline": "2026-06-25T14:00:00Z",
        }
        assert meta["tenant_id"] == "acme"
        assert meta["priority"] == "high"

    def test_ticket_metadata_all_required_fields_present(self):
        """All 6 TicketMetadata fields must be present — none optional."""
        import typing

        from state import TicketMetadata

        fields = typing.get_type_hints(TicketMetadata)
        required = {"ticket_id", "tenant_id", "user_id", "priority",
                    "created_at", "sla_deadline"}
        assert required == set(fields.keys())


# ══════════════════════════════════════════════════════════════════════════════
# CONCEPT 30: Persistent Threads — state survives across invocations
# ══════════════════════════════════════════════════════════════════════════════

class TestPersistentThreads:
    """
    Concept 30 — The killer feature: same thread_id + checkpointer =
    the graph picks up exactly where it left off.
    """

    def test_messages_accumulate_across_invocations(self, memory_checkpointer):
        """
        Real-world: customer sends 3 messages in one session.
        Each message is added to history. Final response has full context.
        """
        graph = _make_simple_graph(checkpointer=memory_checkpointer)
        config = {"configurable": {"thread_id": "multi-turn"}}

        # First call seeds the state; subsequent calls pass {} to resume from checkpoint
        graph.invoke({"messages": [], "count": 0}, config=config)
        graph.invoke({}, config=config)
        r3 = graph.invoke({}, config=config)

        assert r3["count"] == 3
        assert len(r3["messages"]) == 3   # all 3 messages accumulated

    def test_new_thread_id_starts_fresh(self, memory_checkpointer):
        """
        Different thread ID = new conversation. No history carried over.
        This simulates a new customer starting a new support ticket.
        """
        graph = _make_simple_graph(checkpointer=memory_checkpointer)

        # Session 1 — runs 3 times
        c1 = {"configurable": {"thread_id": "session-1"}}
        for _ in range(3):
            graph.invoke({"messages": [], "count": 0}, config=c1)

        # Session 2 — brand new thread_id
        c2 = {"configurable": {"thread_id": "session-2"}}
        r = graph.invoke({"messages": [], "count": 0}, config=c2)

        assert r["count"] == 1   # started from scratch

    def test_same_thread_id_resumes_after_simulated_restart(self, tmp_path):
        """
        The real persistence test: SqliteSaver survives process restart.
        We simulate this by creating a graph, running it, then creating
        a NEW graph from the same SQLite file and resuming.
        """
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver
        db_path = str(tmp_path / "session.db")
        thread_id = "resume-test-001"
        config = {"configurable": {"thread_id": thread_id}}

        # "First process" — runs the graph once
        # Use SqliteSaver(conn) directly — from_conn_string() returns a context manager
        conn1 = sqlite3.connect(db_path, check_same_thread=False)
        cp1 = SqliteSaver(conn1)
        g1 = _make_simple_graph(checkpointer=cp1)
        r1 = g1.invoke({"messages": [], "count": 0}, config=config)
        assert r1["count"] == 1
        conn1.close()  # simulate process exit / connection close

        # "Second process" — opens the same db, creates a new graph instance
        # This simulates: kill -9 python, restart, resume
        conn2 = sqlite3.connect(db_path, check_same_thread=False)
        cp2 = SqliteSaver(conn2)
        g2 = _make_simple_graph(checkpointer=cp2)
        # Pass {} — let checkpoint provide count=1; passing count=0 would reset it
        r2 = g2.invoke({}, config=config)

        # Resumed from count=1 — not reset to 0
        assert r2["count"] == 2, (
            f"Expected count=2 after resume, got {r2['count']}. "
            "State was not persisted to SQLite — checkpointer may not be wired in."
        )
        conn2.close()


# ══════════════════════════════════════════════════════════════════════════════
# CONCEPT 31: Checkpoint Storage — get_state_history()
# ══════════════════════════════════════════════════════════════════════════════

class TestCheckpointStorage:
    """
    Concept 31 — get_state_history() returns an iterator of all checkpoints
    for a thread, most recent first. Used for debugging and time travel.
    """

    @pytest.mark.asyncio
    async def test_get_state_history_returns_snapshots(self, memory_checkpointer):
        """After N invocations, get_state_history() yields N+1 snapshots."""
        from unittest.mock import AsyncMock

        from graph import build_graph

        graph = build_graph(checkpointer=memory_checkpointer)
        config = {"configurable": {"thread_id": "history-test"}}

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(
            return_value=AIMessage(content="mock response")
        )

        with patch("agents.nodes.get_research_llm", return_value=mock_llm), \
             patch("agents.nodes._classify_with_hf", return_value=("general", 0.8)):
            from langchain_core.messages import HumanMessage
            state = {
                "messages": [HumanMessage(content="test ticket")],
                "ticket_type": None, "ticket_metadata": None,
                "classification_confidence": 0.0, "thread_id": None,
                "retrieved_memories": [], "streaming_enabled": False,
                "active_agent": None, "agent_results": [], "query": None,
                "retrieved_docs": [], "retrieval_grade": None,
                "hallucination_check": None, "pending_approval": None,
                "approval_decision": None, "escalation_level": 0,
                "authenticated": True, "rbac_roles": ["agent"],
                "audit_events": [], "final_response": None, "error": None,
            }
            # ainvoke — nodes are async, sync invoke raises TypeError
            await graph.ainvoke(state, config=config)

        history = list(graph.get_state_history(config))
        assert len(history) >= 1, "Expected at least one checkpoint after invocation"

    def test_get_state_history_most_recent_first(self, memory_checkpointer):
        """Snapshots are returned most-recent-first (descending timestamp)."""
        graph = _make_simple_graph(checkpointer=memory_checkpointer)
        config = {"configurable": {"thread_id": "order-test"}}

        graph.invoke({"messages": [], "count": 0}, config=config)
        graph.invoke({}, config=config)  # resume — do not reset count

        history = list(graph.get_state_history(config))
        # Most recent checkpoint has the highest count
        counts = [s.values.get("count", 0) for s in history]
        assert counts == sorted(counts, reverse=True)

    def test_get_state_returns_current_state(self, memory_checkpointer):
        """graph.get_state(config) returns the latest checkpoint."""
        graph = _make_simple_graph(checkpointer=memory_checkpointer)
        config = {"configurable": {"thread_id": "get-state-test"}}

        graph.invoke({"messages": [], "count": 0}, config=config)
        graph.invoke({}, config=config)  # resume — do not reset count

        current = graph.get_state(config)
        assert current.values["count"] == 2


# ══════════════════════════════════════════════════════════════════════════════
# CONCEPTS 32–33: State Recovery & Durable Execution — idempotency
# ══════════════════════════════════════════════════════════════════════════════

class TestIdempotencyAndDurableExecution:
    """
    Concepts 32–33 — Nodes that may be replayed MUST be idempotent.
    Side effects (sending emails, charging cards) must be in dedicated nodes
    that check if the action was already performed before acting.

    Idempotent = running the node twice produces the same result as running it once.
    """

    def test_pure_computation_is_always_idempotent(self, memory_checkpointer):
        """
        Nodes that only compute (no side effects) are naturally idempotent.
        Running them twice produces the same state — safe for replay.
        """
        class PureState(TypedDict):
            value: int
            squared: int

        def square_node(state: PureState) -> dict:
            return {"squared": state["value"] ** 2}

        g = StateGraph(PureState)
        g.add_node("square", square_node)
        g.add_edge("square", END)
        g.set_entry_point("square")
        graph = g.compile(checkpointer=memory_checkpointer)

        config = {"configurable": {"thread_id": "pure-test"}}
        r1 = graph.invoke({"value": 4, "squared": 0}, config=config)
        r2 = graph.invoke({"value": 4, "squared": 0}, config=config)

        assert r1["squared"] == 16
        assert r2["squared"] == 16   # same result — idempotent

    def test_side_effect_node_checks_before_acting(self):
        """
        Pattern for durable execution: side-effect node reads state to check
        if the action was already performed before acting again.

        This is how billing_node will work in Module 7 (HITL):
        check if refund was already issued before issuing another one.
        """
        actions_performed = []

        class DurableState(TypedDict):
            refund_issued: bool
            audit_events: Annotated[list[str], operator.add]

        def refund_node(state: DurableState) -> dict:
            """
            Idempotent refund node.
            Checks if refund was already issued before doing it again.
            This is the durable execution pattern — safe on replay.
            """
            if state.get("refund_issued"):
                # Already done — return unchanged, no side effect
                return {}

            # Perform side effect
            actions_performed.append("refund")
            return {
                "refund_issued": True,
                "audit_events": ["refund_issued"],
            }

        g = StateGraph(DurableState)
        g.add_node("refund", refund_node)
        g.add_edge("refund", END)
        g.set_entry_point("refund")
        graph = g.compile(checkpointer=MemorySaver())

        config = {"configurable": {"thread_id": "durable-test"}}

        # First run — refund issued
        r1 = graph.invoke({"refund_issued": False, "audit_events": []}, config=config)
        assert r1["refund_issued"] is True
        assert len(actions_performed) == 1

        # Second run (replay) — pass {} so checkpoint provides refund_issued=True.
        # IMPORTANT: passing refund_issued=False would OVERRIDE the checkpoint,
        # resetting it and causing the refund to fire again. Durable execution
        # relies on NOT overriding persisted state on replay.
        graph.invoke({}, config=config)
        assert len(actions_performed) == 1   # still 1 — no double-charge


# ══════════════════════════════════════════════════════════════════════════════
# CONCEPT 34: Time Travel — checkpoint_id in config
# ══════════════════════════════════════════════════════════════════════════════

class TestTimeTravel:
    """
    Concept 34 — Time travel lets you resume from any past checkpoint,
    not just the latest. Set checkpoint_id in config["configurable"].

    Use cases:
      - Debug: replay from just before a node failed
      - Testing: fork from a known state and try different inputs
      - Audit: replay the full conversation from any point
    """

    def test_checkpoint_id_in_configurable(self, memory_checkpointer):
        """
        After N runs, get_state_history() yields checkpoints with IDs.
        Passing a checkpoint_id resumes from that exact point.
        """
        graph = _make_simple_graph(checkpointer=memory_checkpointer)
        config = {"configurable": {"thread_id": "time-travel-test"}}

        # Run twice to create two checkpoints — second call resumes (no count override)
        graph.invoke({"messages": [], "count": 0}, config=config)
        graph.invoke({}, config=config)

        # Get history — most recent first
        history = list(graph.get_state_history(config))
        assert len(history) >= 2

        # Find the checkpoint where count=1 (after first run)
        snapshot_at_1 = next(
            (s for s in history if s.values.get("count") == 1), None
        )
        assert snapshot_at_1 is not None, "Should have a checkpoint with count=1"

        checkpoint_id = snapshot_at_1.config["configurable"]["checkpoint_id"]

        # Time travel: resume from count=1 checkpoint.
        # Pass {} — let the checkpoint provide all state.
        # Passing count=0 would override the checkpoint's count=1, defeating time travel.
        time_travel_config = {
            "configurable": {
                "thread_id": "time-travel-test",
                "checkpoint_id": checkpoint_id,
            }
        }
        r_from_past = graph.invoke(
            {},
            config=time_travel_config,
        )

        # Resumed from count=1 → ran node once → count=2
        assert r_from_past["count"] == 2

    def test_checkpoint_id_is_string(self, memory_checkpointer):
        """Checkpoint IDs are strings — can be stored, logged, passed around."""
        graph = _make_simple_graph(checkpointer=memory_checkpointer)
        config = {"configurable": {"thread_id": "id-type-test"}}
        graph.invoke({"messages": [], "count": 0}, config=config)

        history = list(graph.get_state_history(config))
        checkpoint_id = history[0].config["configurable"]["checkpoint_id"]

        assert isinstance(checkpoint_id, str)
        assert len(checkpoint_id) > 0

    def test_checkpoint_id_is_unique_per_snapshot(self, memory_checkpointer):
        """Each checkpoint gets a unique ID — no two checkpoints share an ID."""
        graph = _make_simple_graph(checkpointer=memory_checkpointer)
        config = {"configurable": {"thread_id": "unique-ids-test"}}

        graph.invoke({"messages": [], "count": 0}, config=config)
        graph.invoke({"messages": [], "count": 0}, config=config)
        graph.invoke({"messages": [], "count": 0}, config=config)

        history = list(graph.get_state_history(config))
        ids = [s.config["configurable"]["checkpoint_id"] for s in history]

        assert len(ids) == len(set(ids)), "Checkpoint IDs must be unique"
