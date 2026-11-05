"""
Unit tests for agents/nodes.py

Tests are structured in three categories:
  1. Pure helper function tests (_prefixed functions) — no mocking needed
  2. Node integration tests — mock the LLM/classifier dependencies
  3. Contract tests — verify nodes obey the partial-dict return rule

Why test pure helpers separately from nodes?
  Pure helpers (_classify_with_hf, _build_response_prompt, _get_thread_id)
  contain the actual business logic. Testing them in isolation:
    - Is fast (no async, no mocks)
    - Pinpoints failures precisely
    - Tests edge cases without constructing full AgentState

Run:
    .venv/bin/python -m pytest tests/unit/test_nodes.py -v
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.nodes import (
    ROUTE_KEYS,
    _build_response_prompt,
    _classify_with_hf,
    _get_thread_id,
    billing_node,
    classify_ticket_node,
    general_node,
    research_node,
    respond_node,
    route_ticket,
    technical_node,
)

# ── Helper: _get_thread_id ────────────────────────────────────────────────────

class TestGetThreadId:
    """Pure function — no async, no mocks needed."""

    def test_extracts_thread_id_from_config(self):
        config = {"configurable": {"thread_id": "abc-123"}}
        assert _get_thread_id(config) == "abc-123"

    def test_returns_unknown_when_configurable_missing(self):
        assert _get_thread_id({}) == "unknown"

    def test_returns_unknown_when_thread_id_missing(self):
        config = {"configurable": {}}
        assert _get_thread_id(config) == "unknown"

    def test_returns_unknown_when_config_is_none_like(self):
        # RunnableConfig is a dict — .get() is always safe
        assert _get_thread_id({"metadata": {}}) == "unknown"


# ── Helper: _build_response_prompt ───────────────────────────────────────────

class TestBuildResponsePrompt:
    """Pure function — verifies prompt construction without any LLM calls."""

    def test_first_message_is_system(self):
        messages = [HumanMessage(content="test")]
        prompt = _build_response_prompt(messages, "billing")
        assert isinstance(prompt[0], SystemMessage)

    def test_system_message_contains_ticket_type(self):
        messages = [HumanMessage(content="test")]
        prompt = _build_response_prompt(messages, "billing")
        assert "billing" in prompt[0].content

    def test_human_message_preserved_after_system(self):
        human = HumanMessage(content="My invoice is wrong")
        prompt = _build_response_prompt([human], "billing")
        assert prompt[1] == human

    def test_full_conversation_history_preserved(self):
        messages = [
            HumanMessage(content="first"),
            AIMessage(content="response"),
            HumanMessage(content="follow up"),
        ]
        prompt = _build_response_prompt(messages, "technical")
        # system + all 3 messages = 4 total
        assert len(prompt) == 4

    def test_different_ticket_types_produce_different_prompts(self):
        messages = [HumanMessage(content="test")]
        billing_prompt = _build_response_prompt(messages, "billing")
        technical_prompt = _build_response_prompt(messages, "technical")
        assert billing_prompt[0].content != technical_prompt[0].content

    def test_returns_list(self):
        result = _build_response_prompt([HumanMessage(content="test")], "general")
        assert isinstance(result, list)


# ── Helper: _classify_with_hf ────────────────────────────────────────────────

class TestClassifyWithHf:
    """Tests for HF zero-shot classifier helper — mocks the classifier pipeline."""

    def _make_classifier_result(self, label: str, score: float) -> dict:
        """Helper: builds the dict shape returned by HF zero-shot pipeline."""
        labels = [label, "other1", "other2"]
        scores = [score, (1 - score) / 2, (1 - score) / 2]
        return {"labels": labels, "scores": scores, "sequence": "test"}

    def test_returns_top_label_and_score(self):
        mock_clf = MagicMock(
            return_value=self._make_classifier_result("billing", 0.92)
        )
        with (
            patch("agents.nodes._classify_with_hf.__wrapped__", create=True),
            patch("llm_provider.get_classifier", return_value=mock_clf),
        ):
            label, score = _classify_with_hf("My invoice is wrong")

        assert label == "billing"
        assert score == pytest.approx(0.92)

    def test_low_confidence_falls_back_to_general(self):
        mock_clf = MagicMock(
            return_value=self._make_classifier_result("billing", 0.2)
        )
        with patch("llm_provider.get_classifier", return_value=mock_clf):
            label, score = _classify_with_hf("some ambiguous text")

        assert label == "general"
        assert score == pytest.approx(0.2)

    def test_unknown_label_falls_back_to_general(self):
        mock_clf = MagicMock(
            return_value={"labels": ["unknown_cat"], "scores": [0.95], "sequence": "t"}
        )
        with patch("llm_provider.get_classifier", return_value=mock_clf):
            label, score = _classify_with_hf("some text")

        assert label == "general"
        assert score == 0.0

    def test_returns_tuple(self):
        mock_clf = MagicMock(
            return_value=self._make_classifier_result("technical", 0.85)
        )
        with patch("llm_provider.get_classifier", return_value=mock_clf):
            result = _classify_with_hf("my app is broken")

        assert isinstance(result, tuple)
        assert len(result) == 2


# ── Node: classify_ticket_node ────────────────────────────────────────────────

class TestClassifyTicketNode:
    """Integration tests for classify_ticket_node — mocks the HF classifier."""

    @pytest.mark.asyncio
    async def test_classifies_billing_ticket(self, base_state, runnable_config):
        mock_clf = MagicMock(
            return_value={
                "labels": ["billing", "technical", "general", "research"],
                "scores": [0.91, 0.05, 0.03, 0.01],
                "sequence": "My invoice is wrong",
            }
        )
        with patch("llm_provider.get_classifier", return_value=mock_clf):
            result = await classify_ticket_node(base_state, runnable_config)

        assert result["ticket_type"] == "billing"
        assert result["classification_confidence"] == pytest.approx(0.91)

    @pytest.mark.asyncio
    async def test_classifies_technical_ticket(self, base_state, runnable_config):
        base_state["messages"] = [HumanMessage(content="My app keeps crashing")]
        mock_clf = MagicMock(
            return_value={
                "labels": ["technical", "billing", "general", "research"],
                "scores": [0.88, 0.06, 0.04, 0.02],
                "sequence": "My app keeps crashing",
            }
        )
        with patch("llm_provider.get_classifier", return_value=mock_clf):
            result = await classify_ticket_node(base_state, runnable_config)

        assert result["ticket_type"] == "technical"

    @pytest.mark.asyncio
    async def test_raises_on_empty_messages(self, base_state, runnable_config):
        """Validation rule: empty messages must raise ValueError, not silently fail."""
        base_state["messages"] = []
        with pytest.raises(ValueError, match="empty messages"):
            await classify_ticket_node(base_state, runnable_config)

    @pytest.mark.asyncio
    async def test_returns_only_partial_dict(self, base_state, runnable_config):
        """
        Architecture contract: node must return ONLY its fields.
        Returning full state would bypass reducers and overwrite other nodes' work.
        """
        mock_clf = MagicMock(
            return_value={
                "labels": ["billing", "technical", "general", "research"],
                "scores": [0.91, 0.05, 0.03, 0.01],
                "sequence": "test",
            }
        )
        with patch("llm_provider.get_classifier", return_value=mock_clf):
            result = await classify_ticket_node(base_state, runnable_config)

        # Should ONLY contain the fields this node writes
        assert set(result.keys()) == {"ticket_type", "classification_confidence"}

        # Should NOT contain fields belonging to other nodes
        assert "messages" not in result
        assert "final_response" not in result
        assert "retrieved_docs" not in result
        assert "authenticated" not in result


# ── Node: respond_node ────────────────────────────────────────────────────────

class TestRespondNode:
    """Integration tests for respond_node — mocks the Groq LLM."""

    @pytest.mark.asyncio
    async def test_returns_final_response(self, base_state, runnable_config, mock_llm):
        base_state["ticket_type"] = "billing"
        with patch("agents.nodes.get_research_llm", return_value=mock_llm):
            result = await respond_node(base_state, runnable_config)

        assert result["final_response"] == "Mock response from test LLM"

    @pytest.mark.asyncio
    async def test_appends_ai_message_to_messages(
        self, base_state, runnable_config, mock_llm
    ):
        """
        respond_node writes an AIMessage to state["messages"].
        The add_messages reducer will APPEND it to the existing messages list.
        This test confirms the node returns it correctly for the reducer to handle.
        """
        base_state["ticket_type"] = "general"
        with patch("agents.nodes.get_research_llm", return_value=mock_llm):
            result = await respond_node(base_state, runnable_config)

        assert "messages" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
        assert result["messages"][0].content == "Mock response from test LLM"

    @pytest.mark.asyncio
    async def test_raises_on_empty_messages(self, base_state, runnable_config, mock_llm):
        base_state["messages"] = []
        with pytest.raises(ValueError, match="empty messages"):
            await respond_node(base_state, runnable_config)

    @pytest.mark.asyncio
    async def test_returns_only_partial_dict(
        self, base_state, runnable_config, mock_llm
    ):
        """Architecture contract: only writes messages and final_response."""
        base_state["ticket_type"] = "technical"
        with patch("agents.nodes.get_research_llm", return_value=mock_llm):
            result = await respond_node(base_state, runnable_config)

        assert set(result.keys()) == {"messages", "final_response"}

        assert "ticket_type" not in result
        assert "retrieved_docs" not in result
        assert "authenticated" not in result

    @pytest.mark.asyncio
    async def test_llm_is_invoked_with_prompt_list(
        self, base_state, runnable_config, mock_llm
    ):
        """Verifies the LLM receives a properly structured prompt."""
        base_state["ticket_type"] = "billing"
        with patch("agents.nodes.get_research_llm", return_value=mock_llm):
            await respond_node(base_state, runnable_config)

        mock_llm.ainvoke.assert_called_once()
        call_args = mock_llm.ainvoke.call_args
        prompt = call_args[0][0]  # first positional arg

        # Prompt must start with SystemMessage
        assert isinstance(prompt[0], SystemMessage)
        # Prompt must include the original HumanMessage
        assert any(isinstance(m, HumanMessage) for m in prompt)

    @pytest.mark.asyncio
    async def test_defaults_ticket_type_to_general_when_none(
        self, base_state, runnable_config, mock_llm
    ):
        """
        If ticket_type is None (classify_ticket_node hasn't run yet),
        respond_node falls back to "general" gracefully.
        """
        base_state["ticket_type"] = None
        with patch("agents.nodes.get_research_llm", return_value=mock_llm):
            result = await respond_node(base_state, runnable_config)

        # Confirm it ran without error
        assert "final_response" in result


# ── Router function: route_ticket ─────────────────────────────────────────────

class TestRouteTicket:
    """
    Tests for the route_ticket router function.

    CRITICAL: route_ticket is a PURE SYNC function.
    These tests have no async, no mocks, no fixtures — just state in, string out.
    If any of these take > 1ms, something is wrong.
    """

    def test_routes_billing_ticket(self, base_state):
        base_state["ticket_type"] = "billing"
        assert route_ticket(base_state) == "billing"

    def test_routes_technical_ticket(self, base_state):
        base_state["ticket_type"] = "technical"
        assert route_ticket(base_state) == "technical"

    def test_routes_general_ticket(self, base_state):
        base_state["ticket_type"] = "general"
        assert route_ticket(base_state) == "general"

    def test_routes_research_ticket(self, base_state):
        base_state["ticket_type"] = "research"
        assert route_ticket(base_state) == "research"

    def test_unknown_ticket_type_falls_back_to_general(self, base_state):
        """
        Any unrecognised ticket_type must fall back to 'general'.
        Never raise — an unhandled router exception crashes the entire graph.
        """
        base_state["ticket_type"] = "legal"   # not a valid TicketType
        assert route_ticket(base_state) == "general"

    def test_none_ticket_type_falls_back_to_general(self, base_state):
        """
        ticket_type=None means classify_ticket_node hasn't set it yet.
        Router must still return a safe value.
        """
        base_state["ticket_type"] = None
        assert route_ticket(base_state) == "general"

    def test_returns_string_not_coroutine(self, base_state):
        """
        Router must NOT be async. Verify the return value is a plain str,
        not a coroutine object (which would happen with async def).
        """
        import inspect
        base_state["ticket_type"] = "billing"
        result = route_ticket(base_state)
        assert isinstance(result, str), "route_ticket must return str, not coroutine"
        assert not inspect.iscoroutine(result), "route_ticket must be sync, not async"

    def test_return_value_exists_in_route_keys(self, base_state):
        """
        Every value returned by route_ticket must be a key in ROUTE_KEYS.
        If it's not, LangGraph raises InvalidUpdateError at runtime.
        """
        for ticket_type in ("billing", "technical", "general", "research"):
            base_state["ticket_type"] = ticket_type
            result = route_ticket(base_state)
            assert result in ROUTE_KEYS, (
                f"route_ticket returned {result!r} which is not in ROUTE_KEYS"
            )

    def test_fallback_general_exists_in_route_keys(self, base_state):
        """The fallback 'general' must be a valid key in ROUTE_KEYS."""
        base_state["ticket_type"] = None
        result = route_ticket(base_state)
        assert result in ROUTE_KEYS


# ── Stub handler nodes ────────────────────────────────────────────────────────

class TestHandlerNodes:
    """
    Tests for the stub billing/technical/general/research handler nodes.
    Verifies they set active_agent and append to audit_events.
    """

    def test_billing_node_sets_active_agent(self, base_state, runnable_config):
        # billing_node is synchronous — no await needed
        result = billing_node(base_state, runnable_config)
        assert result["active_agent"] == "billing"

    @pytest.mark.asyncio
    async def test_technical_node_sets_active_agent(self, base_state, runnable_config):
        result = await technical_node(base_state, runnable_config)
        assert result["active_agent"] == "technical"

    @pytest.mark.asyncio
    async def test_general_node_sets_active_agent(self, base_state, runnable_config):
        result = await general_node(base_state, runnable_config)
        assert result["active_agent"] == "general"

    @pytest.mark.asyncio
    async def test_research_node_sets_query(self, base_state, runnable_config):
        result = await research_node(base_state, runnable_config)
        assert result["active_agent"] == "research"
        assert "query" in result

    @pytest.mark.asyncio
    async def test_handler_nodes_append_audit_event(self, base_state, runnable_config):
        """Each handler node must append exactly one audit event.

        billing_node is synchronous; the others are async.
        """
        # billing_node is sync
        result = billing_node(base_state, runnable_config)
        assert "audit_events" in result
        assert len(result["audit_events"]) == 1
        assert result["audit_events"][0]["action"] == "ticket_received"

        for node_fn in (technical_node, general_node, research_node):
            result = await node_fn(base_state, runnable_config)
            assert "audit_events" in result
            assert len(result["audit_events"]) == 1
            assert result["audit_events"][0]["action"] == "ticket_received"

    @pytest.mark.asyncio
    async def test_handler_nodes_do_not_write_messages(
        self, base_state, runnable_config
    ):
        """Handler stubs must not write messages — that belongs to respond_node."""
        # billing_node is sync
        result = billing_node(base_state, runnable_config)
        assert "messages" not in result
        assert "final_response" not in result

        for node_fn in (technical_node, general_node, research_node):
            result = await node_fn(base_state, runnable_config)
            assert "messages" not in result
            assert "final_response" not in result
