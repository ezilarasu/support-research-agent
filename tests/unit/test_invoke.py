"""
Unit tests for Concept 8: Invoke  (updated for Module 6 Streaming)

Tests cover:
  - _build_config(): correct configurable nesting, recursion_limit placement
  - _build_initial_state(): all 21 AgentState fields present and correctly typed
  - run() session loop: astream called with correct args, tokens displayed
  - Error path: exception in _stream_response surfaces without crashing the loop
  - thread_id: same ID reused across messages in one session
  - _print_node_progress(): node labels and filtering

Run:
    .venv/bin/python -m pytest tests/unit/test_invoke.py -v
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessageChunk, HumanMessage

# ── Helpers ────────────────────────────────────────────────────────────────────

async def _async_updates(*events):
    """Async generator that yields update events (stream_mode='updates')."""
    for event in events:
        yield event


async def _async_messages(*chunks):
    """Async generator that yields (chunk, metadata) tuples (stream_mode='messages')."""
    for chunk in chunks:
        yield chunk


def _make_mock_graph(
    ticket_type: str = "general",
    active_agent: str = "general",
    confidence: float = 0.8,
    response_tokens: list[str] | None = None,
):
    """
    Builds a mock graph whose astream() dispatches on stream_mode.

    Pass 1 (updates): yields one classify_ticket update with ticket metadata.
    Pass 2 (messages): yields AIMessageChunk tokens from the respond node.
    """
    if response_tokens is None:
        response_tokens = ["ok"]

    mock_graph = MagicMock()

    async def fake_astream(state, config=None, stream_mode="updates"):
        if stream_mode == "updates":
            yield {
                "classify_ticket": {
                    "ticket_type": ticket_type,
                    "active_agent": active_agent,
                    "classification_confidence": confidence,
                }
            }
            yield {
                "respond": {
                    "final_response": "".join(response_tokens),
                }
            }
        else:  # stream_mode == "messages"
            for token in response_tokens:
                yield (AIMessageChunk(content=token), {"langgraph_node": "respond"})

    mock_graph.astream = fake_astream
    # get_state().next must be falsy so _stream_response doesn't enter _handle_interrupt.
    # MagicMock().next is truthy by default — mock it explicitly.
    state_mock = MagicMock()
    state_mock.next = ()   # empty tuple = graph completed normally, no interrupt
    mock_graph.get_state = MagicMock(return_value=state_mock)
    return mock_graph


# ── _build_config() ───────────────────────────────────────────────────────────

class TestBuildConfig:
    """_build_config() must produce a valid LangGraph RunnableConfig."""

    def _config(self, thread_id: str = "test-thread-abc"):
        from core.settings import get_settings
        from main import _build_config
        return _build_config(thread_id, get_settings())

    def test_configurable_key_is_present(self):
        """
        LangGraph reads thread_id from config['configurable'].
        config['thread_id'] is silently ignored by LangGraph.
        """
        config = self._config()
        assert "configurable" in config, (
            "thread_id must be nested under 'configurable' — not at the top level"
        )

    def test_thread_id_nested_under_configurable(self):
        """thread_id must live inside configurable, not at the top level."""
        config = self._config("my-thread-123")
        assert config["configurable"]["thread_id"] == "my-thread-123"

    def test_thread_id_not_at_top_level(self):
        """
        Common mistake: config = {'thread_id': '...'} — LangGraph ignores this.
        Ensure thread_id is NOT placed at the top level.
        """
        config = self._config()
        assert "thread_id" not in config, (
            "thread_id at the top level is silently ignored by LangGraph. "
            "It must be inside config['configurable']['thread_id']"
        )

    def test_recursion_limit_at_top_level(self):
        """
        recursion_limit is a LangGraph execution parameter.
        It belongs at the top level of config, NOT inside configurable.
        """
        config = self._config()
        assert "recursion_limit" in config

    def test_recursion_limit_not_in_configurable(self):
        """recursion_limit is execution config, not thread config."""
        config = self._config()
        assert "recursion_limit" not in config.get("configurable", {})

    def test_recursion_limit_is_positive_int(self):
        config = self._config()
        limit = config["recursion_limit"]
        assert isinstance(limit, int)
        assert limit > 0

    def test_recursion_limit_matches_settings(self):
        from core.settings import get_settings
        settings = get_settings()
        config = self._config()
        assert config["recursion_limit"] == settings.graph_recursion_limit

    def test_different_thread_ids_produce_different_configs(self):
        from core.settings import get_settings
        from main import _build_config
        settings = get_settings()
        c1 = _build_config("thread-aaa", settings)
        c2 = _build_config("thread-bbb", settings)
        assert c1["configurable"]["thread_id"] != c2["configurable"]["thread_id"]


# ── _build_initial_state() ────────────────────────────────────────────────────

class TestBuildInitialState:
    """_build_initial_state() must produce a complete, valid initial AgentState."""

    def _state(self, ticket: str = "My invoice is wrong"):
        from main import _build_initial_state
        return _build_initial_state(ticket)

    def test_messages_contains_human_message(self):
        state = self._state("test ticket text")
        assert len(state["messages"]) == 1
        assert isinstance(state["messages"][0], HumanMessage)

    def test_messages_content_matches_ticket(self):
        state = self._state("My laptop won't start")
        assert state["messages"][0].content == "My laptop won't start"

    def test_ticket_type_is_none(self):
        """ticket_type starts as None — populated by classify_ticket_node."""
        assert self._state()["ticket_type"] is None

    def test_ticket_metadata_is_none(self):
        assert self._state()["ticket_metadata"] is None

    def test_classification_confidence_is_zero(self):
        state = self._state()
        assert state["classification_confidence"] == 0.0
        assert isinstance(state["classification_confidence"], float)

    def test_thread_id_is_none(self):
        """thread_id in state starts as None — set via config, not state."""
        assert self._state()["thread_id"] is None

    def test_retrieved_memories_is_empty_list(self):
        state = self._state()
        assert state["retrieved_memories"] == []
        assert isinstance(state["retrieved_memories"], list)

    def test_streaming_enabled_is_true(self):
        """
        Module 6: streaming_enabled=True in CLI sessions activates astream()
        path in respond_node, giving real-time token output.
        """
        assert self._state()["streaming_enabled"] is True

    def test_active_agent_is_none(self):
        assert self._state()["active_agent"] is None

    def test_agent_results_is_empty_list(self):
        state = self._state()
        assert state["agent_results"] == []

    def test_query_is_none(self):
        assert self._state()["query"] is None

    def test_retrieved_docs_is_empty_list(self):
        state = self._state()
        assert state["retrieved_docs"] == []

    def test_retrieval_grade_is_none(self):
        assert self._state()["retrieval_grade"] is None

    def test_hallucination_check_is_none(self):
        assert self._state()["hallucination_check"] is None

    def test_pending_approval_is_none(self):
        assert self._state()["pending_approval"] is None

    def test_approval_decision_is_none(self):
        assert self._state()["approval_decision"] is None

    def test_escalation_level_is_zero(self):
        state = self._state()
        assert state["escalation_level"] == 0
        assert isinstance(state["escalation_level"], int)

    def test_authenticated_is_true(self):
        """Starts True — Module 10 wires real JWT auth."""
        assert self._state()["authenticated"] is True

    def test_rbac_roles_contains_agent(self):
        state = self._state()
        assert "agent" in state["rbac_roles"]

    def test_audit_events_is_empty_list(self):
        state = self._state()
        assert state["audit_events"] == []

    def test_final_response_is_none(self):
        assert self._state()["final_response"] is None

    def test_error_is_none(self):
        assert self._state()["error"] is None

    def test_all_21_fields_present(self):
        """
        All 21 AgentState fields must be present in the initial state.
        Missing fields surface as KeyError mid-graph, not at startup.
        """
        from state import AgentState
        state = self._state()
        expected_fields = set(AgentState.__annotations__.keys())
        actual_fields = set(state.keys())
        missing = expected_fields - actual_fields
        assert not missing, f"Missing fields in initial state: {missing}"


# ── _print_node_progress() ────────────────────────────────────────────────────

class TestPrintNodeProgress:
    """
    _print_node_progress() shows a progress label for meaningful nodes
    and silently skips internal / noisy nodes.

    Replaces TestDisplayResult from pre-Module 6. The streaming architecture
    moves all display logic here (progress) and into _stream_response (tokens).
    """

    def test_classify_ticket_shows_label(self, capsys):
        from main import _print_node_progress
        _print_node_progress("classify_ticket")
        out = capsys.readouterr().out
        assert "Classifying" in out

    def test_respond_shows_label(self, capsys):
        from main import _print_node_progress
        _print_node_progress("respond")
        out = capsys.readouterr().out
        assert "Generating" in out

    def test_billing_node_shows_label(self, capsys):
        from main import _print_node_progress
        _print_node_progress("billing_node")
        out = capsys.readouterr().out
        assert "Billing" in out

    def test_internal_start_node_produces_no_output(self, capsys):
        """
        __start__ is LangGraph internal — not meaningful to users.
        Must produce zero output.
        """
        from main import _print_node_progress
        _print_node_progress("__start__")
        out = capsys.readouterr().out
        assert out == ""

    def test_memory_retrieval_is_silent(self, capsys):
        """
        memory_retrieval is in _SILENT_NODES — no progress indicator.
        Shows for a few ms; displaying it adds noise without value.
        """
        from main import _print_node_progress
        _print_node_progress("memory_retrieval")
        out = capsys.readouterr().out
        assert out == ""

    def test_memory_write_is_silent(self, capsys):
        from main import _print_node_progress
        _print_node_progress("memory_write")
        out = capsys.readouterr().out
        assert out == ""

    def test_unknown_node_falls_back_to_node_name(self, capsys):
        """
        Nodes not in _NODE_LABELS dict use their own name as the label.
        This is future-proofing — new nodes show up without code changes.
        """
        from main import _print_node_progress
        _print_node_progress("some_new_node")
        out = capsys.readouterr().out
        assert "some_new_node" in out


# ── run() session loop ────────────────────────────────────────────────────────

class TestRunSessionLoop:
    """
    Tests for the run() async session loop.

    Module 6 upgrade: run() now calls _stream_response() which uses
    graph.astream() with two passes (updates + messages). Tests mock
    graph.astream() instead of graph.ainvoke().
    """

    async def test_astream_called_with_correct_config_structure(self):
        """
        astream must receive config with thread_id nested under 'configurable'.
        This is the most important contract in Concept 8.
        """
        captured_configs = []

        async def fake_astream(state, config=None, stream_mode="updates"):
            if config:
                captured_configs.append(config)
            if stream_mode == "updates":
                yield {"classify_ticket": {"ticket_type": "general", "active_agent": "general", "classification_confidence": 0.8}}
            else:
                yield (AIMessageChunk(content="ok"), {"langgraph_node": "respond"})

        mock_graph = MagicMock()
        mock_graph.astream = fake_astream
        _s = MagicMock()
        _s.next = ()
        mock_graph.get_state = MagicMock(return_value=_s)

        inputs = iter(["test ticket", EOFError()])

        def fake_input(_prompt=""):
            val = next(inputs)
            if isinstance(val, type) and issubclass(val, Exception):
                raise val()
            if isinstance(val, Exception):
                raise val
            return val

        with patch("main.build_graph", return_value=mock_graph), \
             patch("builtins.input", side_effect=fake_input):
            from main import run
            await run()

        # At least one astream call captured config with correct structure
        assert len(captured_configs) >= 1
        assert "configurable" in captured_configs[0]
        assert "thread_id" in captured_configs[0]["configurable"]
        assert "recursion_limit" in captured_configs[0]

    async def test_same_thread_id_reused_across_messages(self):
        """
        All messages in one session must share the same thread_id.
        This is what makes a session resumable in Module 4.
        """
        captured_thread_ids = []

        async def fake_astream(state, config=None, stream_mode="updates"):
            if config:
                captured_thread_ids.append(config["configurable"]["thread_id"])
            if stream_mode == "updates":
                yield {"classify_ticket": {"ticket_type": "general", "active_agent": "general", "classification_confidence": 0.8}}
            else:
                yield (AIMessageChunk(content="ok"), {"langgraph_node": "respond"})

        mock_graph = MagicMock()
        mock_graph.astream = fake_astream

        call_count = 0

        def fake_input(_prompt=""):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "first message"
            elif call_count == 2:
                return "second message"
            raise EOFError()

        with patch("main.build_graph", return_value=mock_graph), \
             patch("builtins.input", side_effect=fake_input):
            from main import run
            await run()

        # Two messages → two Pass-1 astream calls → two thread_ids captured
        assert len(captured_thread_ids) >= 2, (
            f"Expected at least 2 astream calls (one per message), got {len(captured_thread_ids)}"
        )
        # All thread_ids must be the same session UUID
        unique_ids = set(captured_thread_ids)
        assert len(unique_ids) == 1, (
            "thread_id changed between messages in the same session — "
            "multi-turn history will be broken"
        )

    async def test_graph_exception_does_not_crash_session(self, capsys):
        """
        If _stream_response raises an exception, the session loop must
        continue, not crash. The error is surfaced to the user gracefully.
        """
        call_count = 0

        async def fake_astream(state, config=None, stream_mode="updates"):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Simulated provider failure")
            if stream_mode == "updates":
                yield {"classify_ticket": {"ticket_type": "general", "active_agent": "general", "classification_confidence": 0.5}}
            else:
                yield (AIMessageChunk(content="recovered"), {"langgraph_node": "respond"})

        mock_graph = MagicMock()
        mock_graph.astream = fake_astream

        input_count = 0

        def fake_input(_prompt=""):
            nonlocal input_count
            input_count += 1
            if input_count == 1:
                return "bad ticket"
            elif input_count == 2:
                return "good ticket"
            raise EOFError()

        with patch("main.build_graph", return_value=mock_graph), \
             patch("builtins.input", side_effect=fake_input):
            from main import run
            await run()  # must NOT raise

        out = capsys.readouterr().out
        assert "Simulated provider failure" in out or "went wrong" in out

    async def test_empty_input_skipped(self):
        """Empty input (user pressed Enter) must be skipped — no astream call."""
        astream_calls = []

        async def fake_astream(state, config=None, stream_mode="updates"):
            astream_calls.append(stream_mode)
            if stream_mode == "updates":
                yield {"classify_ticket": {"ticket_type": "general", "active_agent": "general", "classification_confidence": 0.8}}
            else:
                yield (AIMessageChunk(content="ok"), {"langgraph_node": "respond"})

        mock_graph = MagicMock()
        mock_graph.astream = fake_astream

        input_count = 0

        def fake_input(_prompt=""):
            nonlocal input_count
            input_count += 1
            _responses = {1: "", 2: "   ", 3: "real ticket"}
            if input_count in _responses:
                return _responses[input_count]
            raise EOFError()

        with patch("main.build_graph", return_value=mock_graph), \
             patch("builtins.input", side_effect=fake_input):
            from main import run
            await run()

        # Two passes per message: "updates" + "messages" = 2 calls for 1 real ticket
        updates_calls = [c for c in astream_calls if c == "updates"]
        assert len(updates_calls) == 1, (
            f"Expected 1 Pass-1 astream call (for 'real ticket' only), got {len(updates_calls)}"
        )

    async def test_initial_state_contains_ticket_as_human_message(self):
        """astream Pass 1 receives the ticket text wrapped in a HumanMessage."""
        captured_states = []

        async def fake_astream(state, config=None, stream_mode="updates"):
            if stream_mode == "updates":
                captured_states.append(state)
                yield {"classify_ticket": {"ticket_type": "general", "active_agent": "general", "classification_confidence": 0.5}}
            else:
                yield (AIMessageChunk(content="ok"), {"langgraph_node": "respond"})

        mock_graph = MagicMock()
        mock_graph.astream = fake_astream
        _s = MagicMock()
        _s.next = ()
        mock_graph.get_state = MagicMock(return_value=_s)

        input_responses = iter(["Billing question here", EOFError()])

        def fake_input(_prompt=""):
            val = next(input_responses)
            if isinstance(val, BaseException):
                raise val
            return val

        with patch("main.build_graph", return_value=mock_graph), \
             patch("builtins.input", side_effect=fake_input):
            from main import run
            await run()

        # Pass 1 receives the full initial state
        assert len(captured_states) >= 1
        msgs = captured_states[0]["messages"]
        assert len(msgs) == 1
        assert isinstance(msgs[0], HumanMessage)
        assert msgs[0].content == "Billing question here"


# ── build_graph called once ───────────────────────────────────────────────────

class TestGraphCompiledOnce:
    """
    build_graph() must be called exactly once per session.
    Calling it per-message would be expensive (full graph compilation).
    """

    async def test_build_graph_called_once_per_session(self):
        mock_graph = _make_mock_graph()

        input_count = 0

        def fake_input(_prompt=""):
            nonlocal input_count
            input_count += 1
            if input_count <= 3:
                return f"message {input_count}"
            raise EOFError()

        with patch("main.build_graph", return_value=mock_graph) as mock_build, \
             patch("builtins.input", side_effect=fake_input):
            from main import run
            await run()

        mock_build.assert_called_once(), (
            "build_graph() was called more than once — it should be called "
            "once at session start, not on every message"
        )
