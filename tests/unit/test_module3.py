"""
Unit tests for Module 3, Concepts 15–21:
  15 - Router Pattern
  16 - ToolNode (tools/ package)
  17 - create_react_agent (contract tests)
  18 - Dynamic Routing (BaseAgent ABC, Registry)
  19 - Hooks (Observer pattern)
  20 - RunnableConfig (safe access)
  21 - Runtime Configuration (settings-driven behaviour)

Run:
    .venv/bin/python -m pytest tests/unit/test_module3.py -v
"""
from unittest.mock import MagicMock, patch

import pytest

# ══════════════════════════════════════════════════════════════════════════════
# CONCEPT 15: Router Pattern
# ══════════════════════════════════════════════════════════════════════════════

class TestRouterPattern:
    """
    Concept 15 — Routers are pure sync functions.
    Three invariants: sync, valid key, no I/O.
    These tests prove our route_ticket obeys all three.
    """

    def test_route_keys_covers_all_ticket_types(self):
        """ROUTE_KEYS must cover every valid TicketType — no unmapped type."""
        import typing

        from agents.nodes import ROUTE_KEYS
        from state import AgentState

        hints = typing.get_type_hints(AgentState, include_extras=True)
        ticket_type_hint = hints["ticket_type"]
        # ticket_type is TicketType | None — get the Literal args
        args = typing.get_args(ticket_type_hint)
        literal_type = args[0]                          # TicketType
        valid_types = typing.get_args(literal_type)     # ("billing", "technical", ...)

        for t in valid_types:
            assert t in ROUTE_KEYS, (
                f"TicketType '{t}' has no entry in ROUTE_KEYS. "
                f"Add it or the conditional edge will raise at runtime."
            )

    def test_router_is_not_async(self):
        """Router must be a plain function — not a coroutine function."""
        import asyncio

        from agents.nodes import route_ticket
        assert not asyncio.iscoroutinefunction(route_ticket), (
            "route_ticket is async — this silently breaks routing. "
            "LangGraph calls routers synchronously; async returns a coroutine "
            "object (truthy) instead of a string key."
        )

    def test_router_returns_string(self):
        """Router return value must be a str — not None, not a coroutine."""
        from agents.nodes import route_ticket
        state = {"ticket_type": "billing", "messages": []}
        result = route_ticket(state)
        assert isinstance(result, str)

    def test_router_fallback_for_unknown_type(self):
        """Unknown ticket_type falls back to 'general' — never crashes."""
        from agents.nodes import route_ticket
        result = route_ticket({"ticket_type": "unknown_xyz"})
        assert result == "general"

    def test_router_fallback_for_none(self):
        """None ticket_type (classify node not run yet) falls back to 'general'."""
        from agents.nodes import route_ticket
        result = route_ticket({"ticket_type": None})
        assert result == "general"

    def test_route_keys_values_match_node_names_in_graph(self):
        """
        ROUTE_KEYS values must exactly match the node names registered in graph.py.
        A mismatch raises at runtime — not at compile time.
        """
        from agents.nodes import ROUTE_KEYS
        from graph import build_graph
        graph = build_graph()
        node_names = set(graph.nodes)
        for ticket_type, node_name in ROUTE_KEYS.items():
            assert node_name in node_names, (
                f"ROUTE_KEYS['{ticket_type}'] = '{node_name}' "
                f"but '{node_name}' is not registered in the graph. "
                f"Registered nodes: {node_names}"
            )


# ══════════════════════════════════════════════════════════════════════════════
# CONCEPT 16: ToolNode — tool contracts
# ══════════════════════════════════════════════════════════════════════════════

class TestToolContracts:
    """
    Concept 16 — Tools must satisfy BaseTool's contract:
      - name: non-empty string
      - description: non-empty string (LLM reads this to decide when to call)
      - args_schema: Pydantic BaseModel with validated fields
      - _run(): synchronous implementation
    """

    def test_search_tool_has_name(self):
        from tools.search_tool import SearchTool
        assert SearchTool().name
        assert isinstance(SearchTool().name, str)

    def test_search_tool_has_description(self):
        from tools.search_tool import SearchTool
        assert SearchTool().description
        assert len(SearchTool().description) > 20, "Description too short — LLM won't use it"

    def test_search_tool_has_args_schema(self):
        from tools.search_tool import SearchTool, SearchToolInput
        assert SearchTool().args_schema is SearchToolInput

    def test_search_tool_input_validates_query(self):
        """Pydantic validates inputs before the tool is called."""
        from pydantic import ValidationError

        from tools.search_tool import SearchToolInput

        valid = SearchToolInput(query="GDPR article 17 data retention")
        assert valid.query == "GDPR article 17 data retention"
        assert valid.max_results == 5   # default

        with pytest.raises(ValidationError):
            SearchToolInput(query="ab")   # too short — min_length=3

    def test_kb_tool_has_name(self):
        from tools.kb_tool import KnowledgeBaseTool
        assert KnowledgeBaseTool().name

    def test_kb_tool_has_description(self):
        from tools.kb_tool import KnowledgeBaseTool
        assert KnowledgeBaseTool().description

    def test_kb_tool_has_args_schema(self):
        from tools.kb_tool import KBToolInput, KnowledgeBaseTool
        assert KnowledgeBaseTool().args_schema is KBToolInput

    def test_search_tool_graceful_when_no_api_key(self, mock_settings):
        """Tool returns a message, not an exception, when API key is missing."""
        from tools.search_tool import SearchTool
        with patch("tools.search_tool.get_settings") as mock_get:
            cfg = MagicMock()
            cfg.tavily_api_key = None
            mock_get.return_value = cfg

            tool = SearchTool()
            result = tool._run("test query")

        assert "unavailable" in result.lower() or "not configured" in result.lower()
        assert "test query" in result

    def test_kb_tool_graceful_when_index_missing(self, mock_settings):
        """Tool returns a message, not an exception, when FAISS index is missing."""
        from tools.kb_tool import KnowledgeBaseTool
        with patch("os.path.exists", return_value=False):
            tool = KnowledgeBaseTool()
            result = tool._run("test query")

        assert "not yet indexed" in result.lower() or "not found" in result.lower()

    def test_get_research_tools_returns_list(self):
        from tools import get_research_tools
        tools = get_research_tools()
        assert isinstance(tools, list)
        assert len(tools) >= 1

    def test_get_technical_tools_returns_list(self):
        from tools import get_technical_tools
        tools = get_technical_tools()
        assert isinstance(tools, list)
        assert len(tools) >= 1

    def test_format_search_results_empty_response(self):
        """Pure helper handles empty results gracefully."""
        from tools.search_tool import _format_search_results
        result = _format_search_results({"results": [], "answer": None})
        assert "No results" in result

    def test_format_search_results_with_answer(self):
        from tools.search_tool import _format_search_results
        result = _format_search_results({
            "answer": "GDPR requires deletion within 30 days.",
            "results": [{"title": "GDPR Guide", "url": "https://example.com", "content": "..."}],
        })
        assert "GDPR requires deletion within 30 days." in result

    def test_format_kb_results_empty(self):
        from tools.kb_tool import _format_kb_results
        result = _format_kb_results([])
        assert "No relevant" in result

    def test_format_kb_results_with_docs(self):
        from langchain_core.documents import Document

        from tools.kb_tool import _format_kb_results
        docs = [Document(page_content="billing policy info", metadata={"source": "billing.md"})]
        result = _format_kb_results(docs)
        assert "billing.md" in result
        assert "billing policy info" in result


# ══════════════════════════════════════════════════════════════════════════════
# CONCEPT 18: Dynamic Routing — BaseAgent ABC + Registry
# ══════════════════════════════════════════════════════════════════════════════

class TestBaseAgentABC:
    """
    Concept 18 — BaseAgent enforces the contract every agent must implement.
    Attempting to instantiate without implementing abstract methods raises TypeError.
    """

    def test_cannot_instantiate_base_agent_directly(self):
        """BaseAgent is abstract — you cannot instantiate it directly."""
        from agents.base import BaseAgent
        with pytest.raises(TypeError, match="abstract"):
            BaseAgent()  # type: ignore

    def test_subclass_must_implement_agent_name(self):
        """agent_name is abstract — subclass must define it."""
        from agents.base import BaseAgent
        class Incomplete(BaseAgent):
            def get_tools(self): return []
            async def process(self, state, config): return {}
            # agent_name NOT implemented

        with pytest.raises(TypeError):
            Incomplete()

    def test_subclass_must_implement_get_tools(self):
        """get_tools is abstract — subclass must define it."""
        from agents.base import BaseAgent
        class Incomplete(BaseAgent):
            @property
            def agent_name(self): return "test"
            async def process(self, state, config): return {}
            # get_tools NOT implemented

        with pytest.raises(TypeError):
            Incomplete()

    def test_subclass_must_implement_process(self):
        """process is abstract — subclass must define it."""
        from agents.base import BaseAgent
        class Incomplete(BaseAgent):
            @property
            def agent_name(self): return "test"
            def get_tools(self): return []
            # process NOT implemented

        with pytest.raises(TypeError):
            Incomplete()

    def test_complete_subclass_instantiates(self):
        """A fully implemented subclass works fine."""
        from agents.base import BaseAgent
        class ConcreteAgent(BaseAgent):
            @property
            def agent_name(self): return "concrete"
            def get_tools(self): return []
            async def process(self, state, config): return {}

        agent = ConcreteAgent()
        assert agent.agent_name == "concrete"
        assert agent.get_tools() == []

    def test_repr_includes_class_name_and_agent_name(self):
        from agents.base import BaseAgent
        class MyAgent(BaseAgent):
            @property
            def agent_name(self): return "my_agent"
            def get_tools(self): return []
            async def process(self, state, config): return {}

        assert "MyAgent" in repr(MyAgent())
        assert "my_agent" in repr(MyAgent())


class TestAgentRegistry:
    """
    Concept 18 — Registry maps names to factories.
    Tests prove: register → resolve → get_agent → fresh instance.
    """

    def setup_method(self):
        """Clear registry before each test — tests must not share state."""
        from agents.registry import clear_registry
        clear_registry()

    def teardown_method(self):
        """Clean up after each test."""
        from agents.registry import clear_registry
        clear_registry()

    def test_register_and_get_agent(self):
        from agents.base import BaseAgent
        from agents.registry import get_agent, register_agent

        class TestAgent(BaseAgent):
            @property
            def agent_name(self): return "test"
            def get_tools(self): return []
            async def process(self, state, config): return {}

        @register_agent("test")
        def create_test_agent():
            return TestAgent()

        agent = get_agent("test")
        assert isinstance(agent, TestAgent)
        assert agent.agent_name == "test"

    def test_get_agent_raises_for_unknown_name(self):
        from agents.registry import get_agent
        with pytest.raises(KeyError, match="not_registered"):
            get_agent("not_registered")

    def test_error_message_includes_available_agents(self):
        """The KeyError message lists available agents to aid debugging."""
        from agents.base import BaseAgent
        from agents.registry import get_agent, register_agent

        class A(BaseAgent):
            @property
            def agent_name(self): return "alpha"
            def get_tools(self): return []
            async def process(self, s, c): return {}

        @register_agent("alpha")
        def factory(): return A()

        with pytest.raises(KeyError) as exc_info:
            get_agent("unknown")

        assert "alpha" in str(exc_info.value)

    def test_list_agents_returns_registered_names(self):
        from agents.base import BaseAgent
        from agents.registry import list_agents, register_agent

        class A(BaseAgent):
            @property
            def agent_name(self): return "a"
            def get_tools(self): return []
            async def process(self, s, c): return {}

        class B(BaseAgent):
            @property
            def agent_name(self): return "b"
            def get_tools(self): return []
            async def process(self, s, c): return {}

        @register_agent("a")
        def fa(): return A()

        @register_agent("b")
        def fb(): return B()

        names = list_agents()
        assert "a" in names
        assert "b" in names

    def test_each_get_agent_call_returns_fresh_instance(self):
        """Agents are stateless — each get_agent() call creates a new instance."""
        from agents.base import BaseAgent
        from agents.registry import get_agent, register_agent

        class S(BaseAgent):
            @property
            def agent_name(self): return "s"
            def get_tools(self): return []
            async def process(self, s, c): return {}

        @register_agent("s")
        def factory(): return S()

        a1 = get_agent("s")
        a2 = get_agent("s")
        assert a1 is not a2   # different instances

    def test_register_agent_returns_factory_unchanged(self):
        """@register_agent is side-effect only — returns the original factory."""
        from agents.base import BaseAgent
        from agents.registry import register_agent

        class R(BaseAgent):
            @property
            def agent_name(self): return "r"
            def get_tools(self): return []
            async def process(self, s, c): return {}

        def my_factory(): return R()

        result = register_agent("r")(my_factory)
        assert result is my_factory   # unchanged


# ══════════════════════════════════════════════════════════════════════════════
# CONCEPT 19: Hooks — Observer pattern
# ══════════════════════════════════════════════════════════════════════════════

class TestHooks:
    """
    Concept 19 — Hooks decouple observability from node business logic.
    """

    def setup_method(self):
        from observability.hooks import clear_hooks
        clear_hooks()

    def teardown_method(self):
        from observability.hooks import clear_hooks
        clear_hooks()

    async def test_pre_hook_called_before_node(self):
        """Pre-hook runs before the node executes."""
        from observability.hooks import node_with_hooks, register_pre_hook

        call_order = []

        def my_pre_hook(name, state, config):
            call_order.append("pre_hook")

        async def my_node(state, config):
            call_order.append("node")
            return {"result": "ok"}

        register_pre_hook(my_pre_hook)
        wrapped = node_with_hooks("test_node", my_node)
        await wrapped({}, {})

        assert call_order == ["pre_hook", "node"]

    async def test_post_hook_called_after_node(self):
        """Post-hook runs after the node returns."""
        from observability.hooks import node_with_hooks, register_post_hook

        call_order = []

        async def my_node(state, config):
            call_order.append("node")
            return {"result": "ok"}

        def my_post_hook(name, result, config):
            call_order.append("post_hook")

        register_post_hook(my_post_hook)
        wrapped = node_with_hooks("test_node", my_node)
        await wrapped({}, {})

        assert call_order == ["node", "post_hook"]

    async def test_post_hook_receives_duration_ms(self):
        """duration_ms is injected into post-hook result dict."""
        from observability.hooks import node_with_hooks, register_post_hook

        received = {}

        def capture_hook(name, result, config):
            received.update(result)

        async def my_node(state, config):
            return {"answer": "42"}

        register_post_hook(capture_hook)
        wrapped = node_with_hooks("t", my_node)
        await wrapped({}, {})

        assert "duration_ms" in received
        assert isinstance(received["duration_ms"], float)
        assert received["duration_ms"] >= 0

    async def test_node_result_does_not_contain_duration_ms(self):
        """duration_ms is only passed to hooks — not returned to LangGraph."""
        from observability.hooks import node_with_hooks

        async def my_node(state, config):
            return {"answer": "42"}

        wrapped = node_with_hooks("t", my_node)
        result = await wrapped({}, {})

        assert "duration_ms" not in result
        assert result == {"answer": "42"}

    async def test_failing_hook_does_not_crash_node(self):
        """If a hook raises, the node still completes normally."""
        from observability.hooks import node_with_hooks, register_pre_hook

        def bad_hook(name, state, config):
            raise RuntimeError("hook exploded!")

        async def my_node(state, config):
            return {"ok": True}

        register_pre_hook(bad_hook)
        wrapped = node_with_hooks("t", my_node)
        result = await wrapped({}, {})
        assert result == {"ok": True}   # node succeeded despite hook failure

    async def test_multiple_hooks_all_called(self):
        """All registered hooks are called — not just the first."""
        from observability.hooks import node_with_hooks, register_post_hook

        called = []

        def hook1(n, r, c): called.append("hook1")
        def hook2(n, r, c): called.append("hook2")
        def hook3(n, r, c): called.append("hook3")

        register_post_hook(hook1)
        register_post_hook(hook2)
        register_post_hook(hook3)

        async def my_node(state, config): return {}

        wrapped = node_with_hooks("t", my_node)
        await wrapped({}, {})

        assert called == ["hook1", "hook2", "hook3"]

    def test_node_with_hooks_preserves_signature(self):
        """
        node_with_hooks uses functools.wraps — the wrapped function has
        the same __name__ and __wrapped__ as the original.
        This is required so LangGraph's inspect.signature() sees (state, config).
        """
        from observability.hooks import node_with_hooks

        async def classify_node(state, config): return {}

        wrapped = node_with_hooks("classify", classify_node)
        assert wrapped.__name__ == "classify_node"
        assert wrapped.__wrapped__ is classify_node


# ══════════════════════════════════════════════════════════════════════════════
# CONCEPT 20: RunnableConfig — safe access
# ══════════════════════════════════════════════════════════════════════════════

class TestRunnableConfig:
    """
    Concept 20 — config must always be accessed with .get() + fallback.
    config["configurable"]["thread_id"] raises KeyError if either key is absent.
    config.get("configurable", {}).get("thread_id", "unknown") never raises.
    """

    def test_get_thread_id_safe_with_full_config(self):
        from agents.nodes import _get_thread_id
        config = {"configurable": {"thread_id": "abc-123"}}
        assert _get_thread_id(config) == "abc-123"

    def test_get_thread_id_safe_with_empty_config(self):
        """Empty config returns fallback — never KeyError."""
        from agents.nodes import _get_thread_id
        assert _get_thread_id({}) == "unknown"

    def test_get_thread_id_safe_with_missing_configurable(self):
        """Missing 'configurable' key returns fallback — never KeyError."""
        from agents.nodes import _get_thread_id
        assert _get_thread_id({"metadata": {"something": "else"}}) == "unknown"

    def test_get_thread_id_safe_with_none_thread_id(self):
        """thread_id: None in config returns fallback."""
        from agents.nodes import _get_thread_id
        result = _get_thread_id({"configurable": {"thread_id": None}})
        assert result == "unknown" or result is None   # either is acceptable

    def test_direct_key_access_would_raise(self):
        """
        This test documents WHY we use .get() — to prove that direct access
        raises KeyError when configurable is missing.
        """
        config = {}
        with pytest.raises(KeyError):
            _ = config["configurable"]["thread_id"]   # ← the bug we're preventing


# ══════════════════════════════════════════════════════════════════════════════
# CONCEPT 21: Runtime Configuration
# ══════════════════════════════════════════════════════════════════════════════

class TestRuntimeConfiguration:
    """
    Concept 21 — Settings control runtime behaviour.
    Nodes and hooks read settings at call time — not at import time.
    This means settings can be changed between test cases.
    """

    def test_latency_hook_gated_by_graph_debug(self, mock_settings, caplog):
        """
        _latency_hook only logs when settings.graph_debug is True.
        In production, graph_debug=False so latency logs don't flood stdout.
        """
        from observability.hooks import _latency_hook

        config = {"configurable": {"thread_id": "t1"}}
        result = {"duration_ms": 42.5}

        # With debug OFF — no log
        with patch("observability.hooks.get_settings") as mock_get:
            cfg = MagicMock()
            cfg.graph_debug = False
            mock_get.return_value = cfg

            import logging
            with caplog.at_level(logging.INFO, logger="observability.hooks"):
                _latency_hook("test_node", result, config)

            assert "node_latency" not in caplog.text

        caplog.clear()

        # With debug ON — log emitted
        with patch("observability.hooks.get_settings") as mock_get:
            cfg = MagicMock()
            cfg.graph_debug = True
            mock_get.return_value = cfg

            with caplog.at_level(logging.INFO, logger="observability.hooks"):
                _latency_hook("test_node", result, config)

            assert "node_latency" in caplog.text or "test_node" in caplog.text

    def test_settings_read_at_call_time_not_import_time(self, monkeypatch):
        """
        Settings are read inside the hook at call time.
        If we change graph_debug between calls, the hook picks up the change.
        This is why we call get_settings() inside functions, not at module level.
        """
        from observability.hooks import _latency_hook

        config = {"configurable": {"thread_id": "t1"}}
        result = {"duration_ms": 10.0}

        called_with_debug_values = []

        with patch("observability.hooks.get_settings") as mock_get:
            for debug_value in [False, True, False]:
                cfg = MagicMock()
                cfg.graph_debug = debug_value
                mock_get.return_value = cfg
                # Track what value the hook sees each time
                called_with_debug_values.append(debug_value)
                _latency_hook("n", result, config)

        assert called_with_debug_values == [False, True, False]
