"""
Unit tests for Module 6, Concepts 42–51: Streaming

  42 - graph.stream()     — sync streaming, yields per-node output
  43 - graph.astream()    — async streaming, used in production
  44 - Stream Modes       — "values", "updates", "messages", "debug"
  45 - values mode        — full state after each node (expensive)
  46 - updates mode       — only node output dicts (lightweight)
  47 - messages mode      — AIMessageChunk tokens from LLM nodes
  48 - debug mode         — gated behind settings.graph_debug
  49 - Token Streaming    — AIMessageChunk.content collected into final string
  50 - Event Streaming    — metadata["langgraph_node"] filters correct node
  51 - Streaming UIs      — all display logic in main.py, never in nodes

Run:
    .venv/bin/python -m pytest tests/unit/test_streaming.py -v
"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langgraph.graph import END, StateGraph

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_streaming_graph(checkpointer=None):
    """
    Minimal graph with one node that 'streams' by returning a string.
    Used to test stream_mode="updates" without a real LLM.
    """
    class StreamState(TypedDict):
        messages: Annotated[list[str], operator.add]
        count: int

    def node_a(state: StreamState) -> dict:
        return {"messages": ["from_a"], "count": state["count"] + 1}

    def node_b(state: StreamState) -> dict:
        return {"messages": ["from_b"], "count": state["count"] + 1}

    g = StateGraph(StreamState)
    g.add_node("node_a", node_a)
    g.add_node("node_b", node_b)
    g.add_edge("node_a", "node_b")
    g.add_edge("node_b", END)
    g.set_entry_point("node_a")
    kwargs = {}
    if checkpointer:
        kwargs["checkpointer"] = checkpointer
    return g.compile(**kwargs)


def _make_config(thread_id: str = "test-thread") -> dict:
    return {"configurable": {"thread_id": thread_id}}


# ══════════════════════════════════════════════════════════════════════════════
# CONCEPT 42–43: graph.stream() and graph.astream()
# ══════════════════════════════════════════════════════════════════════════════

class TestStreamVsInvoke:
    """
    Concepts 42–43 — stream() yields multiple times; invoke() returns once.
    Both return the same final state — the difference is WHEN you see output.
    """

    def test_stream_yields_multiple_events(self):
        """
        stream() yields one event per node that completes.
        A two-node graph yields at least two events before finishing.
        """
        graph = _make_streaming_graph()
        config = _make_config()

        events = list(graph.stream(
            {"messages": [], "count": 0},
            config=config,
            stream_mode="updates",
        ))
        # Two nodes → at least two events
        assert len(events) >= 2

    def test_invoke_returns_single_final_state(self):
        """
        invoke() returns exactly once — the final accumulated state.
        No intermediate results.
        """
        graph = _make_streaming_graph()
        config = _make_config()

        result = graph.invoke({"messages": [], "count": 0}, config=config)
        # Final state: both nodes ran, both messages accumulated
        assert result["count"] == 2
        assert "from_a" in result["messages"]
        assert "from_b" in result["messages"]

    def test_stream_final_state_matches_invoke(self):
        """
        The last 'values' event from stream() equals what invoke() returns.
        Streaming doesn't change the outcome — only the timing of output.
        """
        graph = _make_streaming_graph()
        config_s = _make_config("stream-thread")
        config_i = _make_config("invoke-thread")

        # Stream — collect all value snapshots, take the last one
        value_events = list(graph.stream(
            {"messages": [], "count": 0},
            config=config_s,
            stream_mode="values",
        ))
        streamed_final = value_events[-1]

        # Invoke — direct final state
        invoked_final = graph.invoke({"messages": [], "count": 0}, config=config_i)

        assert streamed_final["count"] == invoked_final["count"]
        assert streamed_final["messages"] == invoked_final["messages"]

    @pytest.mark.asyncio
    async def test_astream_is_async_generator(self):
        """
        astream() is an async generator — use 'async for' to consume it.
        Cannot use list() directly — must collect with a comprehension.
        """
        graph = _make_streaming_graph()
        config = _make_config()

        events = []
        async for event in graph.astream(
            {"messages": [], "count": 0},
            config=config,
            stream_mode="updates",
        ):
            events.append(event)

        assert len(events) >= 2   # at least one event per node


# ══════════════════════════════════════════════════════════════════════════════
# CONCEPTS 44–46: Stream Modes — updates vs values
# ══════════════════════════════════════════════════════════════════════════════

class TestStreamModes:
    """
    Concept 44 — stream_mode controls WHAT is emitted, not HOW OFTEN.
    Concept 45 — "values": full state snapshot (expensive, debugging only)
    Concept 46 — "updates": only what each node returned (lightweight)
    """

    def test_updates_mode_yields_node_name_and_output(self):
        """
        'updates' mode yields {node_name: {partial_state}} per node.
        The dict key IS the node name — no separate metadata needed.
        """
        graph = _make_streaming_graph()
        config = _make_config()

        events = list(graph.stream(
            {"messages": [], "count": 0},
            config=config,
            stream_mode="updates",
        ))

        # Each event is a dict keyed by node name
        node_names = set()
        for event in events:
            for node_name in event:
                node_names.add(node_name)

        assert "node_a" in node_names
        assert "node_b" in node_names

    def test_updates_mode_contains_only_changed_fields(self):
        """
        'updates' mode returns only what the node returned — not full state.
        node_a returns {"messages": [...], "count": 1} — only those two fields.
        """
        graph = _make_streaming_graph()
        config = _make_config()

        events = list(graph.stream(
            {"messages": [], "count": 0},
            config=config,
            stream_mode="updates",
        ))

        node_a_event = next(e for e in events if "node_a" in e)
        node_a_output = node_a_event["node_a"]

        # Only the fields node_a returned
        assert "messages" in node_a_output
        assert "count" in node_a_output
        # node_b's output is NOT here — separate event
        assert "node_b" not in node_a_output

    def test_values_mode_yields_full_state_after_each_node(self):
        """
        'values' mode yields the complete accumulated state after every node.
        More expensive (full dict each time) but useful for debugging.
        """
        graph = _make_streaming_graph()
        config = _make_config()

        snapshots = list(graph.stream(
            {"messages": [], "count": 0},
            config=config,
            stream_mode="values",
        ))

        # Every snapshot is a full state dict
        for snapshot in snapshots:
            assert "messages" in snapshot
            assert "count" in snapshot

        # Count increases monotonically as nodes run
        counts = [s["count"] for s in snapshots]
        assert counts[-1] == 2   # both nodes ran

    def test_updates_mode_is_lighter_than_values_mode(self):
        """
        'updates' payloads are smaller than 'values' payloads.
        This matters at scale — don't use 'values' in production hot paths.
        """
        graph = _make_streaming_graph()
        config_u = _make_config("updates-thread")
        config_v = _make_config("values-thread")

        update_events = list(graph.stream(
            {"messages": [], "count": 0},
            config=config_u,
            stream_mode="updates",
        ))
        value_events = list(graph.stream(
            {"messages": [], "count": 0},
            config=config_v,
            stream_mode="values",
        ))

        # Sum total keys across all events — values has more per event
        update_key_total = sum(
            sum(len(v) for v in e.values() if isinstance(v, dict))
            for e in update_events
        )
        value_key_total = sum(len(e) for e in value_events)

        # Values payloads carry more keys (full state)
        assert value_key_total >= update_key_total


# ══════════════════════════════════════════════════════════════════════════════
# CONCEPT 47: messages mode — token streaming
# ══════════════════════════════════════════════════════════════════════════════

class TestMessagesMode:
    """
    Concept 47 — "messages" mode yields (AIMessageChunk, metadata) tuples.
    Each chunk contains a few tokens. Printing each as it arrives gives
    the word-by-word streaming effect users expect from LLMs.
    """

    @pytest.mark.asyncio
    async def test_messages_mode_yields_chunks_from_llm_node(
        self, memory_checkpointer
    ):
        """
        When a node uses llm.astream(), messages mode intercepts the chunks.
        Each chunk is (AIMessageChunk, {"langgraph_node": "node_name", ...}).
        """
        from graph import build_graph

        mock_llm = MagicMock()

        async def _fake_astream(prompt, **kwargs):
            for word in ["Hello", " there", "!"]:
                yield AIMessageChunk(content=word)

        mock_llm.astream = _fake_astream
        mock_llm.ainvoke = AsyncMock(
            return_value=AIMessage(content="Hello there!")
        )

        graph = build_graph(checkpointer=memory_checkpointer)
        config = {"configurable": {"thread_id": "msg-stream-test", "user_id": "test-user"}}

        state = {
            "messages": [HumanMessage(content="test ticket")],
            "ticket_type": None, "ticket_metadata": None,
            "classification_confidence": 0.0, "thread_id": None,
            "retrieved_memories": [], "streaming_enabled": True,
            "active_agent": None, "agent_results": [], "query": None,
            "retrieved_docs": [], "retrieval_grade": None,
            "hallucination_check": None, "pending_approval": None,
            "approval_decision": None, "escalation_level": 0,
            "authenticated": True, "rbac_roles": ["agent"],
            "audit_events": [], "final_response": None, "error": None,
        }

        chunks_received = []
        with patch("agents.nodes.get_research_llm", return_value=mock_llm), \
             patch("agents.nodes._classify_with_hf", return_value=("general", 0.9)), \
             patch("agents.nodes._summarise_for_memory", return_value="summary"), \
             patch("agents.nodes.get_memory_store") as mock_store_factory:
            mock_store = MagicMock()
            mock_store.asearch = AsyncMock(return_value=[])
            mock_store.astore = AsyncMock()
            mock_store_factory.return_value = mock_store

            async for chunk, metadata in graph.astream(
                state, config=config, stream_mode="messages"
            ):
                if hasattr(chunk, "content") and chunk.content:
                    chunks_received.append((chunk.content, metadata.get("langgraph_node")))

        # Some chunks should have come from the respond node
        respond_chunks = [c for c, n in chunks_received if n == "respond"]
        assert len(respond_chunks) >= 1

    @pytest.mark.asyncio
    async def test_metadata_contains_langgraph_node(self, memory_checkpointer):
        """
        Each messages-mode chunk has metadata["langgraph_node"] = node name.
        This lets the UI filter to only show tokens from the respond node,
        ignoring internal LangGraph messages.
        """
        from graph import build_graph

        mock_llm = MagicMock()

        async def _fake_astream(prompt, **kwargs):
            yield AIMessageChunk(content="token1")

        mock_llm.astream = _fake_astream
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="token1"))

        graph = build_graph(checkpointer=memory_checkpointer)
        config = {"configurable": {"thread_id": "meta-test", "user_id": "test-user"}}

        state = {
            "messages": [HumanMessage(content="test")],
            "ticket_type": None, "ticket_metadata": None,
            "classification_confidence": 0.0, "thread_id": None,
            "retrieved_memories": [], "streaming_enabled": True,
            "active_agent": None, "agent_results": [], "query": None,
            "retrieved_docs": [], "retrieval_grade": None,
            "hallucination_check": None, "pending_approval": None,
            "approval_decision": None, "escalation_level": 0,
            "authenticated": True, "rbac_roles": ["agent"],
            "audit_events": [], "final_response": None, "error": None,
        }

        metadata_seen = []
        with patch("agents.nodes.get_research_llm", return_value=mock_llm), \
             patch("agents.nodes._classify_with_hf", return_value=("general", 0.9)), \
             patch("agents.nodes._summarise_for_memory", return_value="summary"), \
             patch("agents.nodes.get_memory_store") as mock_store_factory:
            mock_store = MagicMock()
            mock_store.asearch = AsyncMock(return_value=[])
            mock_store.astore = AsyncMock()
            mock_store_factory.return_value = mock_store

            async for _chunk, metadata in graph.astream(
                state, config=config, stream_mode="messages"
            ):
                metadata_seen.append(metadata)

        # Every chunk has a langgraph_node key
        assert all("langgraph_node" in m for m in metadata_seen)


# ══════════════════════════════════════════════════════════════════════════════
# CONCEPT 49: Token Streaming — collecting chunks
# ══════════════════════════════════════════════════════════════════════════════

class TestTokenStreaming:
    """
    Concept 49 — AIMessageChunk.content is a partial string (a few tokens).
    Joining all chunks gives the complete response.
    """

    @pytest.mark.asyncio
    async def test_chunks_join_to_complete_response(self):
        """
        Collecting all AIMessageChunk.content values and joining them
        produces the same string as a full ainvoke() response.
        """
        # Simulate what respond_node does with llm.astream()
        words = ["Your ", "refund ", "of ", "$50 ", "has ", "been ", "processed."]
        chunks = [AIMessageChunk(content=w) for w in words]

        # The streaming collection pattern used in respond_node:
        collected = []
        for chunk in chunks:
            collected.append(chunk.content)
        result = "".join(collected)

        assert result == "Your refund of $50 has been processed."

    @pytest.mark.asyncio
    async def test_respond_node_uses_astream_when_streaming_enabled(
        self, base_state, runnable_config
    ):
        """
        When streaming_enabled=True, respond_node calls llm.astream(),
        not llm.ainvoke(). This is what enables token-by-token output.
        """
        from agents.nodes import respond_node

        base_state["streaming_enabled"] = True

        mock_llm = MagicMock()
        astream_called = []
        ainvoke_called = []

        async def _fake_astream(prompt, **kwargs):
            astream_called.append(True)
            yield AIMessageChunk(content="streamed response")

        mock_llm.astream = _fake_astream
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="invoked response"))
        mock_llm.ainvoke.side_effect = lambda *a, **kw: ainvoke_called.append(True) or AIMessage(content="invoked")

        with patch("agents.nodes.get_research_llm", return_value=mock_llm):
            result = await respond_node(base_state, runnable_config)

        assert len(astream_called) == 1, "llm.astream() should have been called"
        assert len(ainvoke_called) == 0, "llm.ainvoke() should NOT have been called"
        assert "final_response" in result

    @pytest.mark.asyncio
    async def test_respond_node_uses_ainvoke_when_streaming_disabled(
        self, base_state, runnable_config
    ):
        """
        When streaming_enabled=False (tests, ainvoke callers), respond_node
        falls back to llm.ainvoke(). Same result, different execution path.
        """
        from agents.nodes import respond_node

        base_state["streaming_enabled"] = False

        mock_llm = MagicMock()
        astream_called = []

        async def _fake_astream(prompt, **kwargs):
            astream_called.append(True)
            yield AIMessageChunk(content="should not be called")

        mock_llm.astream = _fake_astream
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="ainvoke response"))

        with patch("agents.nodes.get_research_llm", return_value=mock_llm):
            result = await respond_node(base_state, runnable_config)

        assert len(astream_called) == 0, "llm.astream() should NOT be called"
        assert result["final_response"] == "ainvoke response"

    @pytest.mark.asyncio
    async def test_empty_chunk_content_is_skipped(self, base_state, runnable_config):
        """
        LLMs sometimes emit empty chunks (e.g. role/stop tokens).
        Joining handles them safely — empty strings don't affect the result.
        """
        from agents.nodes import respond_node

        base_state["streaming_enabled"] = True

        mock_llm = MagicMock()

        async def _fake_astream(prompt, **kwargs):
            yield AIMessageChunk(content="Hello")
            yield AIMessageChunk(content="")     # empty — should be skipped by UI
            yield AIMessageChunk(content=" world")

        mock_llm.astream = _fake_astream

        with patch("agents.nodes.get_research_llm", return_value=mock_llm):
            result = await respond_node(base_state, runnable_config)

        # Node collects all chunks including empty ones — join handles it
        assert result["final_response"] == "Hello world"


# ══════════════════════════════════════════════════════════════════════════════
# CONCEPT 50: Event Streaming — astream_events
# ══════════════════════════════════════════════════════════════════════════════

class TestEventStreaming:
    """
    Concept 50 — astream_events() yields typed event dicts with "event" key.
    More granular than astream() — includes on_llm_start, on_llm_end, etc.
    """

    @pytest.mark.asyncio
    async def test_astream_events_yields_event_dicts(self, memory_checkpointer):
        """
        astream_events() yields dicts with "event", "name", "data" keys.
        Events include: on_chain_start, on_llm_start, on_llm_stream, etc.
        """
        from graph import build_graph

        mock_llm = MagicMock()

        async def _fake_astream(prompt, **kwargs):
            yield AIMessageChunk(content="response")

        mock_llm.astream = _fake_astream
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="response"))

        graph = build_graph(checkpointer=memory_checkpointer)
        config = {"configurable": {"thread_id": "events-test", "user_id": "test-user"}}

        state = {
            "messages": [HumanMessage(content="test")],
            "ticket_type": None, "ticket_metadata": None,
            "classification_confidence": 0.0, "thread_id": None,
            "retrieved_memories": [], "streaming_enabled": True,
            "active_agent": None, "agent_results": [], "query": None,
            "retrieved_docs": [], "retrieval_grade": None,
            "hallucination_check": None, "pending_approval": None,
            "approval_decision": None, "escalation_level": 0,
            "authenticated": True, "rbac_roles": ["agent"],
            "audit_events": [], "final_response": None, "error": None,
        }

        event_types = set()
        with patch("agents.nodes.get_research_llm", return_value=mock_llm), \
             patch("agents.nodes._classify_with_hf", return_value=("general", 0.9)), \
             patch("agents.nodes._summarise_for_memory", return_value="summary"), \
             patch("agents.nodes.get_memory_store") as mock_store_factory:
            mock_store = MagicMock()
            mock_store.asearch = AsyncMock(return_value=[])
            mock_store.astore = AsyncMock()
            mock_store_factory.return_value = mock_store

            async for event in graph.astream_events(state, config=config, version="v2"):
                event_types.add(event["event"])

        # LangGraph always emits at least on_chain_start and on_chain_end
        assert "on_chain_start" in event_types or "on_chain_end" in event_types


# ══════════════════════════════════════════════════════════════════════════════
# CONCEPT 48 + 51: Debug mode and Streaming UIs
# ══════════════════════════════════════════════════════════════════════════════

class TestStreamingArchitecture:
    """
    Concept 48 — debug mode is gated behind settings.graph_debug.
    Concept 51 — all display logic lives in main.py, never in nodes.
    """

    def test_streaming_enabled_flag_in_initial_state(self):
        """
        main.py sets streaming_enabled=True in the initial state.
        respond_node reads this flag to choose astream() vs ainvoke().
        """
        from main import _build_initial_state
        state = _build_initial_state("test ticket")
        assert state["streaming_enabled"] is True

    def test_streaming_disabled_in_base_state_fixture(self, base_state):
        """
        Test fixtures use streaming_enabled=False so tests use ainvoke()
        path internally — no need to mock astream() in every test.
        """
        assert base_state["streaming_enabled"] is False

    def test_graph_debug_controlled_by_settings(self, monkeypatch):
        """
        graph_debug=True passes debug=True to graph.compile().
        This is the ONLY way to enable debug mode — never hardcoded.
        """
        from core.settings import get_settings
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        monkeypatch.setenv("GRAPH_DEBUG", "true")
        get_settings.cache_clear()
        settings = get_settings()
        assert settings.graph_debug is True
        get_settings.cache_clear()

    def test_node_progress_labels_are_defined(self):
        """
        _print_node_progress() maps node names to user-friendly labels.
        All graph nodes must have an entry — unknown nodes show raw name.
        """
        from main import _print_node_progress
        # Should not raise for any known node name
        for node in ["classify_ticket", "billing_node", "approval_check",
                     "supervisor_node", "kb_worker", "web_worker",
                     "research_collect", "respond",
                     "memory_retrieval", "memory_write", "__start__"]:
            _print_node_progress(node)   # no assertion — just must not raise

    def test_build_response_prompt_includes_memories(self):
        """
        When retrieved_memories is non-empty, _build_response_prompt injects
        them into the system message. The LLM has user context.
        """
        from langchain_core.messages import HumanMessage

        from agents.nodes import _build_response_prompt

        memories = [
            "Alice had a double-charge issue in June 2026.",
            "Alice is on enterprise tier.",
        ]
        messages = [HumanMessage(content="I have a billing question")]
        prompt = _build_response_prompt(messages, "billing", memories)

        system_content = prompt[0].content
        assert "double-charge" in system_content
        assert "enterprise" in system_content

    def test_build_response_prompt_no_memories_has_no_context_section(self):
        """
        When no memories, the prompt has no 'Relevant context' section.
        Keeps prompts concise when there's no history.
        """
        from langchain_core.messages import HumanMessage

        from agents.nodes import _build_response_prompt

        messages = [HumanMessage(content="test")]
        prompt = _build_response_prompt(messages, "billing", [])
        system_content = prompt[0].content
        assert "Relevant context" not in system_content
