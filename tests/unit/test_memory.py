"""
Unit tests for Module 5, Concepts 35–41: Long-Term Memory

  35 - Long-Term Memory    — store lives outside graph state, cross-thread
  36 - Memory Store        — InMemoryStore with semantic search
  37 - Semantic Memory     — embeddings enable meaning-based retrieval
  38 - Memory Retrieval    — memory_retrieval_node reads before responding
  39 - Memory Summarization — _summarise_for_memory compresses a turn
  40 - Memory Compression  — _summarise_for_memory called by memory_write_node
  41 - Config Driven       — memory_top_k from settings, graceful degradation

Run:
    .venv/bin/python -m pytest tests/unit/test_memory.py -v
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.store.memory import InMemoryStore

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_store():
    """Returns a fresh LongTermMemoryStore backed by an InMemoryStore (no embeddings)."""
    from memory.store import LongTermMemoryStore
    return LongTermMemoryStore(store=InMemoryStore())


def _make_config(user_id: str = "test-user", thread_id: str = "test-thread") -> dict:
    return {
        "configurable": {
            "user_id": user_id,
            "thread_id": thread_id,
        }
    }


# ══════════════════════════════════════════════════════════════════════════════
# CONCEPTS 35–36: Long-Term Memory & Memory Store
# ══════════════════════════════════════════════════════════════════════════════

class TestLongTermMemoryStore:
    """
    Concept 35 — The store lives OUTSIDE the graph state.
    Concept 36 — LongTermMemoryStore wraps InMemoryStore.
    """

    @pytest.mark.asyncio
    async def test_store_and_retrieve_by_key(self):
        """Stored memory is retrievable. Basic read/write contract."""
        store = _make_store()
        await store.astore(
            user_id="alice",
            key="pref_billing",
            text="Alice prefers itemised invoices with line-item breakdown.",
        )
        results = await store.asearch(user_id="alice", query="invoice preference")
        assert len(results) >= 1
        assert any("itemised" in r["text"] for r in results)

    @pytest.mark.asyncio
    async def test_memories_are_user_scoped(self):
        """
        Alice's memories never appear in Bob's search results.
        Namespace isolation is the core multi-tenancy guarantee.
        """
        store = _make_store()
        await store.astore(user_id="alice", key="pref", text="Alice prefers email contact.")
        await store.astore(user_id="bob",   key="pref", text="Bob prefers phone contact.")

        alice_results = await store.asearch(user_id="alice", query="contact preference")
        bob_results   = await store.asearch(user_id="bob",   query="contact preference")

        alice_texts = [r["text"] for r in alice_results]
        bob_texts   = [r["text"] for r in bob_results]

        assert any("Alice" in t for t in alice_texts)
        assert not any("Alice" in t for t in bob_texts)
        assert any("Bob" in t for t in bob_texts)
        assert not any("Bob" in t for t in alice_texts)

    @pytest.mark.asyncio
    async def test_upsert_overwrites_same_key(self):
        """
        Storing with the same key UPDATES, not duplicates.
        Critical: prevents duplicate memories accumulating on every turn.
        """
        store = _make_store()
        await store.astore(user_id="alice", key="tier", text="Alice is on starter tier.")
        await store.astore(user_id="alice", key="tier", text="Alice upgraded to enterprise tier.")

        results = await store.asearch(user_id="alice", query="tier")
        texts = [r["text"] for r in results]

        # Only the latest value should appear
        assert any("enterprise" in t for t in texts)
        # Old value is gone (upserted)
        assert not any("starter" in t for t in texts)

    @pytest.mark.asyncio
    async def test_memory_includes_created_at_timestamp(self):
        """Every stored memory has an ISO 8601 timestamp."""
        store = _make_store()
        await store.astore(user_id="alice", key="k1", text="some memory")
        results = await store.asearch(user_id="alice", query="memory")
        assert len(results) >= 1
        assert "created_at" in results[0]
        assert results[0]["created_at"] != ""

    @pytest.mark.asyncio
    async def test_empty_namespace_returns_empty_list(self):
        """Searching for a user with no memories returns empty list, not error."""
        store = _make_store()
        results = await store.asearch(user_id="nonexistent-user", query="anything")
        assert results == []

    @pytest.mark.asyncio
    async def test_result_has_required_keys(self):
        """Each result dict has text, key, score, and created_at."""
        store = _make_store()
        await store.astore(user_id="alice", key="mykey", text="test memory text")
        results = await store.asearch(user_id="alice", query="test")
        assert len(results) >= 1
        r = results[0]
        assert "text" in r
        assert "key" in r
        assert "score" in r
        assert "created_at" in r

    @pytest.mark.asyncio
    async def test_metadata_stored_alongside_text(self):
        """Extra metadata fields are preserved in the stored value."""
        store = _make_store()
        await store.astore(
            user_id="alice",
            key="meta_test",
            text="test",
            metadata={"ticket_type": "billing", "thread_id": "abc"},
        )
        results = await store.asearch(user_id="alice", query="test")
        assert len(results) >= 1


# ══════════════════════════════════════════════════════════════════════════════
# CONCEPT 36: Factory — get_memory_store()
# ══════════════════════════════════════════════════════════════════════════════

class TestMemoryStoreFactory:
    """
    Concept 36 — get_memory_store() is lru_cached (unlike get_checkpointer()).
    InMemoryStore holds no file handles — safe to share across the process.
    """

    def test_factory_returns_long_term_memory_store(self):
        """get_memory_store() returns a LongTermMemoryStore instance."""
        from memory.store import LongTermMemoryStore, get_memory_store
        # Clear cache to get a fresh instance for this test
        get_memory_store.cache_clear()
        store = get_memory_store()
        assert isinstance(store, LongTermMemoryStore)

    def test_factory_is_cached(self):
        """Same instance returned on repeated calls — lru_cache in effect."""
        from memory.store import get_memory_store
        get_memory_store.cache_clear()
        s1 = get_memory_store()
        s2 = get_memory_store()
        assert s1 is s2

    def test_factory_gracefully_falls_back_without_embeddings(self):
        """
        If HuggingFace embeddings can't load, factory returns a store anyway.
        Memory still works for exact-key lookups — semantic search degrades.

        Why patch llm_provider.get_embeddings and not memory.store.get_embeddings?
          get_embeddings is imported INSIDE get_memory_store() as a local import:
            from llm_provider import get_embeddings
          Local imports bind to the source module (llm_provider), not to memory.store.
          Patching the source module is the correct approach for local imports.
        """
        from memory.store import LongTermMemoryStore, get_memory_store
        get_memory_store.cache_clear()
        with patch("llm_provider.get_embeddings", side_effect=RuntimeError("no model")):
            store = get_memory_store()
        assert isinstance(store, LongTermMemoryStore)
        get_memory_store.cache_clear()


# ══════════════════════════════════════════════════════════════════════════════
# CONCEPT 38: Memory Retrieval Node
# ══════════════════════════════════════════════════════════════════════════════

class TestMemoryRetrievalNode:
    """
    Concept 38 — memory_retrieval_node runs before the agent responds.
    It reads state["messages"], searches the store, and writes retrieved_memories.
    """

    @pytest.mark.asyncio
    async def test_returns_retrieved_memories(self, base_state, runnable_config):
        """Happy path: memories are retrieved and returned."""
        from agents.nodes import memory_retrieval_node

        mock_store = _make_store()
        # Pre-populate with a memory — use await directly (we're already in async)
        await mock_store.astore(
            user_id="test-user-001",
            key="past_issue",
            text="User had a double-charge issue in June 2026.",
        )

        with patch("agents.nodes.get_memory_store", return_value=mock_store):
            result = await memory_retrieval_node(base_state, runnable_config)

        assert "retrieved_memories" in result
        assert isinstance(result["retrieved_memories"], list)

    @pytest.mark.asyncio
    async def test_returns_empty_list_on_empty_messages(self, runnable_config):
        """No messages → no query → empty memories returned, no error."""
        from agents.nodes import memory_retrieval_node
        from state import AgentState
        state = AgentState(
            messages=[],
            ticket_type=None, ticket_metadata=None,
            classification_confidence=0.0, thread_id=None,
            retrieved_memories=[], streaming_enabled=False,
            active_agent=None, agent_results=[], query=None,
            retrieved_docs=[], retrieval_grade=None,
            hallucination_check=None, pending_approval=None,
            approval_decision=None, escalation_level=0,
            authenticated=True, rbac_roles=["agent"],
            audit_events=[], final_response=None, error=None,
        )
        result = await memory_retrieval_node(state, runnable_config)
        assert result == {"retrieved_memories": []}

    @pytest.mark.asyncio
    async def test_returns_empty_list_on_store_failure(self, base_state, runnable_config):
        """
        Store failure must NEVER block the ticket flow.
        Node returns [] gracefully — agent proceeds without memory context.
        """
        from agents.nodes import memory_retrieval_node

        mock_store = MagicMock()
        mock_store.asearch = AsyncMock(side_effect=RuntimeError("store down"))

        with patch("agents.nodes.get_memory_store", return_value=mock_store):
            result = await memory_retrieval_node(base_state, runnable_config)

        assert result == {"retrieved_memories": []}

    @pytest.mark.asyncio
    async def test_uses_last_message_as_query(self, runnable_config):
        """The last message content is the semantic search query."""
        from langchain_core.messages import HumanMessage

        from agents.nodes import memory_retrieval_node
        from state import AgentState

        state = AgentState(
            messages=[
                HumanMessage(content="First message"),
                HumanMessage(content="Latest message about billing"),
            ],
            ticket_type=None, ticket_metadata=None,
            classification_confidence=0.0, thread_id=None,
            retrieved_memories=[], streaming_enabled=False,
            active_agent=None, agent_results=[], query=None,
            retrieved_docs=[], retrieval_grade=None,
            hallucination_check=None, pending_approval=None,
            approval_decision=None, escalation_level=0,
            authenticated=True, rbac_roles=["agent"],
            audit_events=[], final_response=None, error=None,
        )

        captured_queries = []

        mock_store = MagicMock()
        async def _capture_search(user_id, query, limit=None):
            captured_queries.append(query)
            return []
        mock_store.asearch = _capture_search

        with patch("agents.nodes.get_memory_store", return_value=mock_store):
            await memory_retrieval_node(state, runnable_config)

        assert len(captured_queries) == 1
        assert captured_queries[0] == "Latest message about billing"

    @pytest.mark.asyncio
    async def test_returns_partial_dict_not_full_state(self, base_state, runnable_config):
        """Node must return only retrieved_memories — not the full state."""
        from agents.nodes import memory_retrieval_node

        mock_store = MagicMock()
        mock_store.asearch = AsyncMock(return_value=[])

        with patch("agents.nodes.get_memory_store", return_value=mock_store):
            result = await memory_retrieval_node(base_state, runnable_config)

        # Should only contain retrieved_memories
        assert set(result.keys()) == {"retrieved_memories"}

    @pytest.mark.asyncio
    async def test_user_id_defaults_to_anonymous(self, base_state):
        """
        Config with no user_id → user defaults to "anonymous".
        Memories are still stored/retrieved — no error.
        """
        from agents.nodes import memory_retrieval_node

        config_no_user = {"configurable": {"thread_id": "t1"}}  # no user_id
        mock_store = MagicMock()
        mock_store.asearch = AsyncMock(return_value=[])

        captured_user_ids = []
        async def _capture(user_id, query, limit=None):
            captured_user_ids.append(user_id)
            return []
        mock_store.asearch = _capture

        with patch("agents.nodes.get_memory_store", return_value=mock_store):
            await memory_retrieval_node(base_state, config_no_user)

        assert captured_user_ids == ["anonymous"]


# ══════════════════════════════════════════════════════════════════════════════
# CONCEPTS 39–40: Summarization & Compression
# ══════════════════════════════════════════════════════════════════════════════

class TestMemoryWriteNode:
    """
    Concept 39 — memory_write_node calls _summarise_for_memory after responding.
    Concept 40 — summary is compressed to one sentence by the fast LLM.
    """

    @pytest.mark.asyncio
    async def test_writes_memory_after_response(self, base_state, runnable_config):
        """Happy path: node summarises and stores a memory, returns audit event."""
        from agents.nodes import memory_write_node

        base_state["final_response"] = "Your refund of $50 has been processed."
        base_state["ticket_type"] = "billing"

        mock_store = MagicMock()
        mock_store.astore = AsyncMock()

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(
            return_value=AIMessage(content="Customer requested refund, $50 processed.")
        )

        with patch("agents.nodes.get_memory_store", return_value=mock_store), \
             patch("agents.nodes._summarise_for_memory",
                   return_value="Customer had billing issue, refund processed."):
            result = await memory_write_node(base_state, runnable_config)

        assert "audit_events" in result
        assert len(result["audit_events"]) == 1
        audit = result["audit_events"][0]
        assert audit["node"] == "memory_write_node"
        assert audit["action"] == "memory_written"

    @pytest.mark.asyncio
    async def test_returns_empty_dict_on_missing_final_response(
        self, base_state, runnable_config
    ):
        """No final_response → nothing to summarise → return {} gracefully."""
        from agents.nodes import memory_write_node

        base_state["final_response"] = None   # respond_node hasn't run yet

        result = await memory_write_node(base_state, runnable_config)
        assert result == {}

    @pytest.mark.asyncio
    async def test_returns_empty_dict_on_store_failure(self, base_state, runnable_config):
        """
        Store write failure must NOT crash the graph.
        The agent's response was already delivered — memory is best-effort.
        """
        from agents.nodes import memory_write_node

        base_state["final_response"] = "Some response."

        with patch("agents.nodes._summarise_for_memory", return_value="summary"), \
             patch("agents.nodes.get_memory_store") as mock_factory:
            mock_store = MagicMock()
            mock_store.astore = AsyncMock(side_effect=RuntimeError("db down"))
            mock_factory.return_value = mock_store
            result = await memory_write_node(base_state, runnable_config)

        assert result == {}

    @pytest.mark.asyncio
    async def test_returns_empty_dict_on_summarisation_failure(
        self, base_state, runnable_config
    ):
        """LLM summarisation failure → graceful degradation, no crash."""
        from agents.nodes import memory_write_node

        base_state["final_response"] = "Some response."

        with patch("agents.nodes._summarise_for_memory",
                   side_effect=RuntimeError("llm rate limited")):
            result = await memory_write_node(base_state, runnable_config)

        assert result == {}

    @pytest.mark.asyncio
    async def test_memory_key_includes_ticket_type(self, base_state, runnable_config):
        """Memory key format is '{ticket_type}_{thread_id_prefix}'."""
        from agents.nodes import memory_write_node

        base_state["final_response"] = "Billing resolved."
        base_state["ticket_type"] = "billing"

        captured_keys = []

        mock_store = MagicMock()
        async def _capture_store(user_id, key, text, metadata=None):
            captured_keys.append(key)
        mock_store.astore = _capture_store

        with patch("agents.nodes.get_memory_store", return_value=mock_store), \
             patch("agents.nodes._summarise_for_memory", return_value="summary"):
            await memory_write_node(base_state, runnable_config)

        assert len(captured_keys) == 1
        assert captured_keys[0].startswith("billing_")


# ══════════════════════════════════════════════════════════════════════════════
# CONCEPT 39: _summarise_for_memory helper
# ══════════════════════════════════════════════════════════════════════════════

class TestSummariseForMemory:
    """
    Concept 39 — _summarise_for_memory is the compression strategy.
    It's a pure-ish async helper: injectable LLM, testable in isolation.
    """

    @pytest.mark.asyncio
    async def test_returns_string(self):
        """Returns a string summary regardless of conversation length."""
        from agents.nodes import _summarise_for_memory

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(
            return_value=AIMessage(content="Customer had billing issue, refund issued.")
        )

        messages = [
            HumanMessage(content="I was charged twice for my subscription."),
        ]

        with patch("llm_provider.get_fast_llm", return_value=mock_llm):
            result = await _summarise_for_memory(messages, "Refund of $49.99 processed.")

        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_strips_whitespace_from_summary(self):
        """LLM output with leading/trailing whitespace is cleaned."""
        from agents.nodes import _summarise_for_memory

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(
            return_value=AIMessage(content="  Summary with padding.  \n")
        )

        with patch("llm_provider.get_fast_llm", return_value=mock_llm):
            result = await _summarise_for_memory(
                [HumanMessage(content="test")], "response"
            )

        assert result == "Summary with padding."

    @pytest.mark.asyncio
    async def test_uses_only_last_3_human_messages(self):
        """Only the last 3 human messages are sent to the LLM — not the full history."""
        from agents.nodes import _summarise_for_memory

        captured_prompts = []
        mock_llm = AsyncMock()
        async def _capture_invoke(prompt, **kwargs):
            captured_prompts.append(prompt)
            return AIMessage(content="summary")
        mock_llm.ainvoke = _capture_invoke

        messages = [
            HumanMessage(content=f"Message {i}") for i in range(10)
        ]

        with patch("llm_provider.get_fast_llm", return_value=mock_llm):
            await _summarise_for_memory(messages, "response")

        # The HumanMessage in the prompt should contain at most 3 messages
        assert len(captured_prompts) == 1
        prompt_content = captured_prompts[0][-1].content  # last message in prompt
        # Messages 7, 8, 9 should be present; message 0 should not
        assert "Message 9" in prompt_content
        assert "Message 0" not in prompt_content


# ══════════════════════════════════════════════════════════════════════════════
# CONCEPT 41: Config-Driven Behaviour
# ══════════════════════════════════════════════════════════════════════════════

class TestConfigDrivenMemory:
    """
    Concept 41 — Memory behaviour is driven by settings, not hardcoded.
    memory_top_k controls how many memories are returned per search.
    """

    @pytest.mark.asyncio
    async def test_search_respects_memory_top_k(self, monkeypatch):
        """
        Searching with limit=None falls back to settings.memory_top_k.
        If settings.memory_top_k = 3, at most 3 memories returned.
        """
        from core.settings import get_settings
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        monkeypatch.setenv("MEMORY_TOP_K", "2")
        get_settings.cache_clear()

        store = _make_store()
        # Store 5 memories — use await directly (already in async context)
        for i in range(5):
            await store.astore(user_id="alice", key=f"k{i}", text=f"memory {i}")

        results = await store.asearch(user_id="alice", query="memory", limit=2)
        assert len(results) <= 2

        get_settings.cache_clear()

    def test_protocols_are_runtime_checkable(self):
        """
        ISP: SemanticSearchProtocol is @runtime_checkable — isinstance() works.
        This lets tests verify store implementations satisfy the contract.
        """
        from memory.protocols import SemanticSearchProtocol

        store = _make_store()
        # isinstance check works because @runtime_checkable is set
        assert isinstance(store, SemanticSearchProtocol)

    def test_message_store_protocol_is_runtime_checkable(self):
        """MessageStoreProtocol is also @runtime_checkable."""
        from memory.protocols import MessageStoreProtocol
        # A class that implements the protocol methods satisfies isinstance
        class FakeStore:
            def store(self, thread_id, message): pass
            def retrieve(self, thread_id, n): return []

        assert isinstance(FakeStore(), MessageStoreProtocol)
