"""
Unit tests for Module 2, Concepts 9–12: Reducers

This test file exists to answer three questions:

  1. WHAT is the exact data-loss bug that happens without a reducer?
  2. HOW does operator.add prevent it?
  3. HOW does add_messages differ from operator.add (deduplication)?

Tests are structured to teach — each class has a docstring explaining
the principle being demonstrated, not just "what the code does".

Tests use a minimal compiled graph (InMemorySaver) to demonstrate
reducer behaviour end-to-end, not just in isolation.

Run:
    .venv/bin/python -m pytest tests/unit/test_reducers.py -v
"""

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

# ── Helpers: minimal graphs for demonstrating reducer behaviour ────────────────

def _make_list_graph(use_reducer: bool):
    """
    Builds a two-node graph where both nodes write to the same list field.

    WITH use_reducer=True  → field uses operator.add → both writes accumulate
    WITH use_reducer=False → field has no reducer  → second write overwrites first

    This graph is the "proof" for Concept 9.
    """
    if use_reducer:
        class State(TypedDict):
            items: Annotated[list[str], operator.add]
    else:
        class State(TypedDict):  # type: ignore[no-redef]
            items: list[str]     # no reducer — last write wins

    def node_a(state: State) -> dict:
        return {"items": ["from_node_a"]}

    def node_b(state: State) -> dict:
        return {"items": ["from_node_b"]}

    g = StateGraph(State)
    g.add_node("a", node_a)
    g.add_node("b", node_b)
    g.add_edge("a", "b")
    g.add_edge("b", END)
    g.set_entry_point("a")
    return g.compile()


def _make_message_graph(use_add_messages: bool):
    """
    Builds a graph that demonstrates add_messages deduplication.

    WITH use_add_messages=True  → replaying same message ID → appears once
    WITH use_add_messages=False → operator.add → replaying same message → duplicated
    """
    if use_add_messages:
        class MsgState(TypedDict):
            messages: Annotated[list, add_messages]
    else:
        class MsgState(TypedDict):  # type: ignore[no-redef]
            messages: Annotated[list, operator.add]  # append but no dedup

    def echo_node(state: MsgState) -> dict:
        return {"messages": [AIMessage(content="response", id="fixed-ai-id")]}

    g = StateGraph(MsgState)
    g.add_node("echo", echo_node)
    g.add_edge("echo", END)
    g.set_entry_point("echo")
    return g.compile(checkpointer=MemorySaver())


# ══════════════════════════════════════════════════════════════════════════════
# CONCEPT 9: Reducers
# ══════════════════════════════════════════════════════════════════════════════

class TestTheDataLossBugWithoutReducer:
    """
    Concept 9 — The core problem reducers solve.

    When two nodes both write to the same list field and there is NO reducer,
    LangGraph uses last-write-wins: the second node's value replaces the first.
    This is silent — no error, no warning, just missing data.

    This class demonstrates the exact bug so you recognise it in production.
    """

    def test_without_reducer_second_node_overwrites_first(self):
        """
        THE BUG: Node A writes ["from_node_a"]. Node B writes ["from_node_b"].
        Without a reducer, Node B's write REPLACES Node A's write.
        Final state has only ["from_node_b"] — Node A's data is silently lost.
        """
        graph = _make_list_graph(use_reducer=False)
        result = graph.invoke({"items": []})

        # Node A's contribution is gone — overwritten by Node B
        assert "from_node_a" not in result["items"], (
            "Without a reducer, Node A's write should be overwritten by Node B. "
            "If this fails, LangGraph changed its default merge behaviour."
        )
        assert result["items"] == ["from_node_b"], (
            "Last-write-wins: only the final node's return value survives."
        )

    def test_without_reducer_audit_trail_would_be_incomplete(self):
        """
        Real consequence: imagine Node A is the auth node and Node B is the
        billing node. Both append to audit_events WITHOUT a reducer.
        The auth event is silently dropped — compliance audit trail is broken.
        """
        class BrokenAuditState(TypedDict):
            audit_events: list[dict]   # ← no reducer — WRONG

        def auth_node(state: BrokenAuditState) -> dict:
            return {"audit_events": [{"node": "auth", "action": "jwt_verified"}]}

        def billing_node(state: BrokenAuditState) -> dict:
            return {"audit_events": [{"node": "billing", "action": "refund_requested"}]}

        g = StateGraph(BrokenAuditState)
        g.add_node("auth", auth_node)
        g.add_node("billing", billing_node)
        g.add_edge("auth", "billing")
        g.add_edge("billing", END)
        g.set_entry_point("auth")
        graph = g.compile()

        result = graph.invoke({"audit_events": []})

        # The auth audit event is gone — the trail is broken
        actions = [e["action"] for e in result["audit_events"]]
        assert "jwt_verified" not in actions, (
            "Auth event overwritten — this is the data-loss bug reducers fix."
        )
        assert actions == ["refund_requested"]


class TestOperatorAddReducer:
    """
    Concept 9 continued — operator.add as a reducer.

    operator.add(existing, new) = existing + new
    This is standard Python list concatenation — safe, no mutation.
    """

    def test_with_reducer_both_nodes_accumulate(self):
        """
        THE FIX: With operator.add, both Node A and Node B contribute.
        Final state has ["from_node_a", "from_node_b"] — neither is lost.
        """
        graph = _make_list_graph(use_reducer=True)
        result = graph.invoke({"items": []})

        assert "from_node_a" in result["items"]
        assert "from_node_b" in result["items"]
        assert len(result["items"]) == 2

    def test_with_reducer_order_is_preserved(self):
        """
        operator.add preserves execution order: Node A before Node B.
        Important for audit trails where chronological order matters.
        """
        graph = _make_list_graph(use_reducer=True)
        result = graph.invoke({"items": []})

        assert result["items"] == ["from_node_a", "from_node_b"]

    def test_reducer_fixes_audit_trail(self):
        """
        With operator.add on audit_events, both nodes' entries are preserved.
        The compliance audit trail is now complete.
        """
        class FixedAuditState(TypedDict):
            audit_events: Annotated[list[dict], operator.add]   # ← reducer

        def auth_node(state: FixedAuditState) -> dict:
            return {"audit_events": [{"node": "auth", "action": "jwt_verified"}]}

        def billing_node(state: FixedAuditState) -> dict:
            return {"audit_events": [{"node": "billing", "action": "refund_requested"}]}

        g = StateGraph(FixedAuditState)
        g.add_node("auth", auth_node)
        g.add_node("billing", billing_node)
        g.add_edge("auth", "billing")
        g.add_edge("billing", END)
        g.set_entry_point("auth")
        graph = g.compile()

        result = graph.invoke({"audit_events": []})

        actions = [e["action"] for e in result["audit_events"]]
        assert "jwt_verified" in actions,     "auth event must be in audit trail"
        assert "refund_requested" in actions, "billing event must be in audit trail"
        assert len(result["audit_events"]) == 2

    def test_reducer_node_returns_only_new_items(self):
        """
        Nodes return ONLY new items — not the accumulated list.
        The reducer handles the merge. Nodes must never read-then-write.

        If a node reads state["audit_events"] and returns existing + [new],
        every checkpoint replay will DOUBLE every entry.
        """
        appended_values = []

        class AuditState(TypedDict):
            audit_events: Annotated[list[str], operator.add]

        def node_that_returns_only_new(state: AuditState) -> dict:
            # ✅ CORRECT: return only the new item
            new_event = "event_from_node"
            appended_values.append(new_event)
            return {"audit_events": [new_event]}   # NOT state["audit_events"] + [new_event]

        g = StateGraph(AuditState)
        g.add_node("n", node_that_returns_only_new)
        g.add_edge("n", END)
        g.set_entry_point("n")
        graph = g.compile()

        initial_state = {"audit_events": ["pre_existing_event"]}
        result = graph.invoke(initial_state)

        # pre_existing_event is preserved + new event added — no doubling
        assert result["audit_events"] == ["pre_existing_event", "event_from_node"]
        assert len(result["audit_events"]) == 2   # not 3 (would mean doubling)

    def test_operator_add_does_not_mutate_existing_list(self):
        """
        operator.add creates a NEW list — never mutates the existing one.
        This is LangGraph's copy-on-write contract. Mutation breaks replay.
        """
        existing = ["a", "b"]
        new = ["c"]
        result = operator.add(existing, new)

        assert result == ["a", "b", "c"]
        assert existing == ["a", "b"]  # original unchanged — no mutation
        assert result is not existing   # new object — not the same reference

    def test_multiple_parallel_nodes_all_accumulate(self):
        """
        fan-out: three nodes run 'in parallel' (sequential in test),
        all appending to the same reducer field.
        All three contributions must survive.
        """
        class FanOutState(TypedDict):
            results: Annotated[list[str], operator.add]

        def worker_a(state: FanOutState) -> dict:
            return {"results": ["worker_a_result"]}

        def worker_b(state: FanOutState) -> dict:
            return {"results": ["worker_b_result"]}

        def worker_c(state: FanOutState) -> dict:
            return {"results": ["worker_c_result"]}

        # Sequential simulation of parallel fan-out
        g = StateGraph(FanOutState)
        g.add_node("a", worker_a)
        g.add_node("b", worker_b)
        g.add_node("c", worker_c)
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        g.add_edge("c", END)
        g.set_entry_point("a")
        graph = g.compile()

        result = graph.invoke({"results": []})

        assert len(result["results"]) == 3
        assert "worker_a_result" in result["results"]
        assert "worker_b_result" in result["results"]
        assert "worker_c_result" in result["results"]


# ══════════════════════════════════════════════════════════════════════════════
# CONCEPT 10: Annotated State
# ══════════════════════════════════════════════════════════════════════════════

class TestAnnotatedStateDeclaration:
    """
    Concept 10 — Annotated[T, reducer] is how LangGraph knows which reducer
    to apply to each field.

    The Annotated type from Python's typing module carries two pieces of info:
      1. The actual type (list[dict]) — used by mypy for type checking
      2. The metadata (operator.add) — read by LangGraph at compile time

    LangGraph inspects the schema at compile() time and builds a merge map:
      {"audit_events": operator.add, "messages": add_messages, ...}

    This merge map is applied every time a node returns a partial dict.
    """

    def test_annotated_field_carries_reducer_as_metadata(self):
        """
        Annotated[list[str], operator.add].__metadata__ contains the reducer.
        This is how LangGraph discovers which reducer to use.
        """
        from typing import get_args

        field_type = Annotated[list[str], operator.add]

        # get_args returns (actual_type, *metadata)
        args = get_args(field_type)
        assert args[0] == list[str]         # the real type
        assert args[1] is operator.add      # the reducer (metadata[0])

    def test_agentstate_reducer_fields_have_annotated_type(self):
        """
        All list fields in AgentState that multiple nodes write to
        must be declared with Annotated, not plain list[T].

        We use get_type_hints(include_extras=True) instead of __annotations__
        because state.py uses `from __future__ import annotations`, which turns
        all annotations into lazy strings. get_type_hints() resolves them to
        real types, preserving Annotated metadata.
        """
        from typing import get_args, get_type_hints

        from state import AgentState

        reducer_fields = [
            "messages",
            "retrieved_memories",
            "agent_results",
            "retrieved_docs",
            "rbac_roles",
            "audit_events",
        ]

        # include_extras=True preserves Annotated[...] — without it, Annotated
        # is stripped and get_args returns an empty tuple
        hints = get_type_hints(AgentState, include_extras=True)
        for field in reducer_fields:
            assert field in hints, f"Field '{field}' missing from AgentState"
            field_type = hints[field]
            args = get_args(field_type)
            assert len(args) >= 2, (
                f"Field '{field}' is not Annotated — it has no reducer. "
                f"Multiple nodes write to this field; it WILL lose data. "
                f"Tip: use get_type_hints(include_extras=True), not __annotations__."
            )

    def test_scalar_fields_do_not_have_reducer(self):
        """
        Scalar fields (ticket_type, active_agent, etc.) that only one node
        writes to should NOT have Annotated — they should use plain types.
        Unnecessarily applying operator.add to a scalar field would cause errors.
        """
        from typing import get_type_hints

        from state import AgentState

        scalar_fields = [
            "ticket_type",
            "active_agent",
            "final_response",
            "error",
            "authenticated",
            "ticket_metadata",
            "retrieval_grade",
            "hallucination_check",
            "pending_approval",
            "approval_decision",
        ]

        import typing
        from typing import get_origin

        hints = get_type_hints(AgentState, include_extras=True)
        for field in scalar_fields:
            if field not in hints:
                continue
            field_type = hints[field]
            # A reducer field has Annotated as its OUTERMOST type constructor.
            # get_origin(Annotated[X, reducer]) is Annotated.
            # get_origin(str | None) is Union — NOT Annotated.
            # This is more precise than inspecting args[1] (which includes NoneType).
            assert get_origin(field_type) is not typing.Annotated, (
                f"Scalar field '{field}' is declared as Annotated[...] — "
                f"this means a reducer was applied to a field with a single writer. "
                f"Remove the Annotated wrapper; use a plain type annotation instead."
            )


# ══════════════════════════════════════════════════════════════════════════════
# CONCEPT 11: operator.add as a reducer
# ══════════════════════════════════════════════════════════════════════════════

class TestOperatorAddBehaviour:
    """
    Concept 11 — operator.add(existing, new) = existing + new

    This is standard Python list concatenation. Used as a LangGraph reducer,
    it means: "when a node writes to this field, append its items to the
    existing list, don't replace it."

    Key properties to understand:
      - Works on any sequence type that supports +
      - Creates a new list — does NOT mutate the original
      - Preserves order: existing items first, new items appended
      - No deduplication — same item can appear multiple times
    """

    def test_operator_add_is_list_concatenation(self):
        existing = [1, 2, 3]
        new = [4, 5]
        assert operator.add(existing, new) == [1, 2, 3, 4, 5]

    def test_operator_add_on_empty_existing(self):
        """First node write: existing is []. Result is just the new items."""
        assert operator.add([], ["first_item"]) == ["first_item"]

    def test_operator_add_on_empty_new(self):
        """Node returns empty list — nothing added, existing preserved."""
        assert operator.add(["existing"], []) == ["existing"]

    def test_operator_add_allows_duplicates(self):
        """
        operator.add does NOT deduplicate. The same event can appear twice
        if a node is retried or if two nodes return the same item.
        Use a custom reducer if deduplication is required.
        """
        existing = [{"id": "evt-1"}]
        new = [{"id": "evt-1"}]   # duplicate
        result = operator.add(existing, new)
        assert len(result) == 2   # both copies present

    def test_operator_add_vs_read_then_write_bug(self):
        """
        The most common reducer bug: a node reads state and returns
        state[field] + [new_item]. This doubles on replay.

        Correct pattern: return ONLY [new_item] and let the reducer append.
        """
        class TestState(TypedDict):
            items: Annotated[list[str], operator.add]

        replay_count = [0]

        def bad_node(state: TestState) -> dict:
            """❌ WRONG: reads existing + appends — doubles on replay."""
            replay_count[0] += 1
            # Simulate what happens on replay: operator.add called again
            # on the doubled list
            return {"items": state["items"] + ["new_item"]}

        def good_node(state: TestState) -> dict:
            """✅ CORRECT: returns only the new item."""
            return {"items": ["new_item"]}

        # Good node: operator.add([existing], [new_item]) → correct
        g_good = StateGraph(TestState)
        g_good.add_node("n", good_node)
        g_good.add_edge("n", END)
        g_good.set_entry_point("n")
        result_good = g_good.compile().invoke({"items": ["pre"]})
        assert result_good["items"] == ["pre", "new_item"]   # ✅ correct

        # Bad node: state["items"] already contains "pre", returns ["pre", "new_item"]
        # operator.add merges: ["pre"] + ["pre", "new_item"] = ["pre", "pre", "new_item"]
        g_bad = StateGraph(TestState)
        g_bad.add_node("n", bad_node)
        g_bad.add_edge("n", END)
        g_bad.set_entry_point("n")
        result_bad = g_bad.compile().invoke({"items": ["pre"]})
        assert result_bad["items"] == ["pre", "pre", "new_item"]  # ❌ "pre" doubled


# ══════════════════════════════════════════════════════════════════════════════
# CONCEPT 12: add_messages
# ══════════════════════════════════════════════════════════════════════════════

class TestAddMessagesReducer:
    """
    Concept 12 — add_messages vs operator.add for chat history.

    add_messages does two things operator.add cannot:
      1. APPENDS new messages (same as operator.add)
      2. DEDUPLICATES by message.id — replaying from a checkpoint won't
         inject duplicate messages into the conversation history

    Why this matters: when LangGraph resumes from a checkpoint after HITL,
    it replays the saved state. Without deduplication, every resume adds
    a copy of all prior messages. With add_messages, replay is idempotent.
    """

    def test_add_messages_appends_new_messages(self):
        """Basic case: add_messages works like operator.add for fresh messages."""
        from langgraph.graph.message import add_messages
        existing = [HumanMessage(content="hello", id="msg-1")]
        new = [AIMessage(content="hi there", id="msg-2")]
        result = add_messages(existing, new)
        assert len(result) == 2
        assert result[0].content == "hello"
        assert result[1].content == "hi there"

    def test_add_messages_deduplicates_by_id(self):
        """
        If a message with the same ID is added twice, add_messages keeps
        only the latest version (the new one replaces the existing one).

        operator.add would produce two copies — this is the key difference.
        """
        from langgraph.graph.message import add_messages
        msg_v1 = AIMessage(content="original response", id="ai-msg-1")
        msg_v2 = AIMessage(content="updated response",  id="ai-msg-1")  # same ID

        existing = [msg_v1]
        new = [msg_v2]
        result = add_messages(existing, new)

        assert len(result) == 1   # only ONE message — not two
        assert result[0].content == "updated response"   # latest version

    def test_operator_add_does_not_deduplicate(self):
        """
        operator.add on messages would duplicate on checkpoint replay.
        This is exactly why add_messages exists.
        """
        msg = HumanMessage(content="hello", id="msg-1")
        existing = [msg]
        replay = [msg]   # same object replayed
        result = operator.add(existing, replay)
        assert len(result) == 2   # ❌ duplicated — NOT safe for replay

    def test_add_messages_in_graph_deduplicates_on_replay(self):
        """
        End-to-end: a graph with add_messages reducer, invoked twice with
        the same initial message ID, should not duplicate the message.
        """
        graph = _make_message_graph(use_add_messages=True)
        config = {"configurable": {"thread_id": "dedup-test"}}
        initial = {"messages": [HumanMessage(content="first", id="h-1")]}

        # First invocation
        result1 = graph.invoke(initial, config=config)
        len(result1["messages"])

        # Second invocation — same human message ID
        # With add_messages: the HumanMessage is deduplicated (same id="h-1")
        # A new AI response is appended (different id each invocation via LangGraph)
        result2 = graph.invoke(initial, config=config)
        # The human message should appear once, not twice
        human_msgs = [m for m in result2["messages"] if isinstance(m, HumanMessage)]
        assert len(human_msgs) == 1, (
            f"Human message with id='h-1' appeared {len(human_msgs)} times. "
            "add_messages should deduplicate by id — replay must be idempotent."
        )

    def test_add_messages_preserves_message_types(self):
        """
        add_messages works with all LangChain message types:
        HumanMessage, AIMessage, SystemMessage, ToolMessage.
        """
        from langgraph.graph.message import add_messages
        existing = [
            SystemMessage(content="You are helpful", id="sys-1"),
            HumanMessage(content="hello", id="h-1"),
        ]
        new = [AIMessage(content="hi", id="ai-1")]
        result = add_messages(existing, new)

        assert len(result) == 3
        assert isinstance(result[0], SystemMessage)
        assert isinstance(result[1], HumanMessage)
        assert isinstance(result[2], AIMessage)


# ══════════════════════════════════════════════════════════════════════════════
# CONCEPT 13: State Merging — partial dict returns
# ══════════════════════════════════════════════════════════════════════════════

class TestStateMerging:
    """
    Concept 13 — nodes return PARTIAL dicts.

    When a node returns {"ticket_type": "billing"}, LangGraph merges ONLY
    that key into the state. All other fields are left untouched.

    Returning the FULL state from a node is a design smell:
      - It couples the node to every other field
      - Reducer fields get re-applied — potential doubling
      - Makes the node's responsibility unclear

    The partial dict return is the contract that makes nodes composable.
    """

    def test_node_partial_return_preserves_other_fields(self):
        """
        A node that returns only {"ticket_type": "billing"} leaves all
        other state fields unchanged.
        """
        class PartialState(TypedDict):
            ticket_type: str | None
            confidence: float
            tags: Annotated[list[str], operator.add]

        def classify_node(state: PartialState) -> dict:
            return {"ticket_type": "billing"}   # ← partial: only updates ticket_type

        g = StateGraph(PartialState)
        g.add_node("classify", classify_node)
        g.add_edge("classify", END)
        g.set_entry_point("classify")
        graph = g.compile()

        initial = {"ticket_type": None, "confidence": 0.92, "tags": ["urgent"]}
        result = graph.invoke(initial)

        assert result["ticket_type"] == "billing"   # updated
        assert result["confidence"] == 0.92         # unchanged — node didn't touch it
        assert result["tags"] == ["urgent"]         # unchanged

    def test_full_state_return_doubles_reducer_fields(self):
        """
        THE BUG: returning the full state from a node with a reducer field
        causes the reducer to be applied to the ENTIRE existing list plus the new list.

        audit_events starts as ["pre"]. Node reads it, returns the full state
        including {"audit_events": ["pre", "new"]}. LangGraph applies operator.add:
        ["pre"] + ["pre", "new"] = ["pre", "pre", "new"] — "pre" appears twice.
        """
        class AuditState(TypedDict):
            audit_events: Annotated[list[str], operator.add]
            ticket_type: str | None

        def buggy_node(state: AuditState) -> dict:
            # ❌ WRONG: reads existing events, returns ALL of them
            new_events = state["audit_events"] + ["new_event"]
            return {
                "audit_events": new_events,   # includes "pre_event" again!
                "ticket_type": "billing",
            }

        g = StateGraph(AuditState)
        g.add_node("n", buggy_node)
        g.add_edge("n", END)
        g.set_entry_point("n")
        graph = g.compile()

        result = graph.invoke({"audit_events": ["pre_event"], "ticket_type": None})
        # "pre_event" appears TWICE — the full-state-return bug
        assert result["audit_events"].count("pre_event") == 2, (
            "The full-state-return bug: operator.add([pre_event], [pre_event, new_event]) "
            "= [pre_event, pre_event, new_event]. 'pre_event' is doubled."
        )

    def test_correct_partial_return_no_doubling(self):
        """
        THE FIX: return only the new items from the node.
        operator.add([pre_event], [new_event]) = [pre_event, new_event]. Correct.
        """
        class AuditState(TypedDict):
            audit_events: Annotated[list[str], operator.add]
            ticket_type: str | None

        def correct_node(state: AuditState) -> dict:
            # ✅ CORRECT: return only what this node adds
            return {
                "audit_events": ["new_event"],  # NOT state["audit_events"] + [...]
                "ticket_type": "billing",
            }

        g = StateGraph(AuditState)
        g.add_node("n", correct_node)
        g.add_edge("n", END)
        g.set_entry_point("n")
        graph = g.compile()

        result = graph.invoke({"audit_events": ["pre_event"], "ticket_type": None})
        assert result["audit_events"] == ["pre_event", "new_event"]  # ✅ no doubling
        assert result["audit_events"].count("pre_event") == 1


# ── Module-level custom reducers ─────────────────────────────────────────────
# Must be at module level — TypedDict annotations with `from __future__ import
# annotations` are evaluated lazily; functions defined inside test methods
# are not visible during annotation resolution.

def _dedup_by_source(existing: list, new: list) -> list:
    """Merge docs, skipping duplicates by source URL."""
    seen = {d["source"] for d in existing}
    return existing + [d for d in new if d["source"] not in seen]


_BOUNDED_MAX = 3


def _bounded_append(existing: list, new: list) -> list:
    """Append and cap list at _BOUNDED_MAX items (keep newest)."""
    return (existing + new)[-_BOUNDED_MAX:]


# ══════════════════════════════════════════════════════════════════════════════
# CONCEPT 14: Reducer Design Patterns — custom reducers
# ══════════════════════════════════════════════════════════════════════════════

class TestCustomReducers:
    """
    Concept 14 — any callable with signature (existing: T, new: T) -> T
    can be used as a reducer.

    Custom reducers let you encode merge logic beyond simple append:
    - Deduplication (seen in add_messages)
    - Bounded lists (keep last N)
    - Priority-based merge (keep highest scored items)
    - Idempotent merge (skip if already present)
    """

    def test_custom_dedup_reducer(self):
        """
        Custom reducer that deduplicates by a key field.
        Used in Module 9 to prevent duplicate documents from multi-query retrieval.
        """
        class DocState(TypedDict):
            docs: Annotated[list[dict], _dedup_by_source]

        def query_a_node(state: DocState) -> dict:
            return {"docs": [
                {"source": "kb/billing.md", "content": "billing info"},
                {"source": "kb/faq.md",     "content": "faq info"},
            ]}

        def query_b_node(state: DocState) -> dict:
            # kb/billing.md is a duplicate — should be skipped
            return {"docs": [
                {"source": "kb/billing.md",   "content": "billing info again"},  # dup
                {"source": "kb/technical.md", "content": "tech info"},
            ]}

        g = StateGraph(DocState)
        g.add_node("a", query_a_node)
        g.add_node("b", query_b_node)
        g.add_edge("a", "b")
        g.add_edge("b", END)
        g.set_entry_point("a")
        graph = g.compile()

        result = graph.invoke({"docs": []})
        sources = [d["source"] for d in result["docs"]]

        assert len(result["docs"]) == 3  # billing.md, faq.md, technical.md
        assert sources.count("kb/billing.md") == 1  # deduplicated
        assert "kb/technical.md" in sources

    def test_bounded_list_reducer(self):
        """
        Custom reducer that caps the list at N items.
        Prevents unbounded memory growth for long-running sessions.
        """
        class BoundedState(TypedDict):
            events: Annotated[list[str], _bounded_append]

        def add_events_node(state: BoundedState) -> dict:
            return {"events": [f"event_{len(state['events']) + 1}"]}

        g = StateGraph(BoundedState)
        g.add_node("n", add_events_node)
        g.add_edge("n", END)
        g.set_entry_point("n")
        graph = g.compile()

        # Start with MAX events already in state
        initial = {"events": ["event_1", "event_2", "event_3"]}
        result = graph.invoke(initial)

        assert len(result["events"]) == _BOUNDED_MAX   # capped at _BOUNDED_MAX
        assert "event_4" in result["events"]  # newest added
        assert "event_1" not in result["events"]  # oldest evicted

    def test_custom_reducer_must_be_pure(self):
        """
        Custom reducers must be pure functions — no side effects, no I/O.
        They are called synchronously by LangGraph's merge engine.

        This test verifies that the reducer receives (existing, new) and
        returns a new value without modifying either input.
        """
        mutations = []

        def bad_mutating_reducer(existing: list, new: list) -> list:
            existing.append("mutated!")   # ❌ mutates the existing list
            mutations.append(True)
            return existing + new

        def good_pure_reducer(existing: list, new: list) -> list:
            return existing + new  # ✅ creates new list

        # Demonstrate that mutation is dangerous
        original = ["a"]
        bad_mutating_reducer(original, ["b"])
        assert "mutated!" in original  # original was mutated — dangerous

        # Demonstrate that pure reducer is safe
        original2 = ["a"]
        good_pure_reducer(original2, ["b"])
        assert original2 == ["a"]  # original unchanged
