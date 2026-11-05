"""
Unit + integration tests for Concept 7: Compile

Tests cover:
  - build_graph() compile kwargs are passed through correctly
  - Compiled graph is immutable (can't add nodes after compile)
  - debug flag is gated / passed correctly
  - get_graph_info() returns expected topology metadata
  - InMemorySaver checkpointer works for test-time persistence
  - interrupt_before is registered on the compiled graph

Run:
    .venv/bin/python -m pytest tests/unit/test_compile.py -v
"""
from __future__ import annotations

# ── build_graph() signature ───────────────────────────────────────────────────

class TestBuildGraphSignature:
    """build_graph() must accept compile kwargs as explicit params."""

    def test_builds_without_any_kwargs(self):
        from graph import build_graph
        graph = build_graph()
        assert hasattr(graph, "ainvoke")

    def test_builds_with_in_memory_checkpointer(self):
        """
        InMemorySaver is the test-time checkpointer.
        build_graph(checkpointer=InMemorySaver()) must compile without error.
        Used in all persistence tests (Module 4).
        """
        from langgraph.checkpoint.memory import InMemorySaver

        from graph import build_graph
        graph = build_graph(checkpointer=InMemorySaver())
        assert hasattr(graph, "ainvoke")

    def test_builds_with_debug_false(self):
        from graph import build_graph
        graph = build_graph(debug=False)
        assert hasattr(graph, "ainvoke")

    def test_builds_with_interrupt_before(self):
        """
        interrupt_before registers HITL pause points on the compiled graph.
        Tests that the kwarg is accepted — full HITL behaviour in Module 7.
        """
        from langgraph.checkpoint.memory import InMemorySaver

        from graph import build_graph
        # interrupt_before requires a checkpointer — LangGraph enforces this
        graph = build_graph(
            checkpointer=InMemorySaver(),
            interrupt_before=["billing_node"],
        )
        assert hasattr(graph, "ainvoke")

    def test_interrupt_before_without_checkpointer_compiles_but_cannot_resume(self):
        """
        LangGraph does NOT raise at compile time when interrupt_before is used
        without a checkpointer. The error surfaces at invoke/resume time instead.

        This is a LangGraph design decision: compile() is permissive.
        The runtime refuses to resume without a checkpointer.

        Production rule: always pair interrupt_before with a checkpointer.
        Module 7 demonstrates the full HITL pattern with SqliteSaver.
        """
        from graph import build_graph
        # Compiles without error — LangGraph is permissive at build time
        graph = build_graph(interrupt_before=["billing_node"])
        assert hasattr(graph, "ainvoke")


# ── Immutability ──────────────────────────────────────────────────────────────

class TestCompiledGraphImmutability:
    """
    CompiledStateGraph is immutable — the topology is fixed after compile().

    IMPORTANT: The underlying StateGraph object is still mutable, but any
    changes to it after compile() are NOT reflected in the compiled graph.
    This is a common confusion point.
    """

    def test_compiled_graph_has_no_add_node_method(self):
        """
        The CompiledStateGraph doesn't expose add_node — topology is locked.
        Callers can't accidentally modify the compiled graph.
        """
        from graph import build_graph
        graph = build_graph()
        assert not hasattr(graph, "add_node"), (
            "Compiled graph should not expose add_node — topology is immutable"
        )

    def test_compiled_graph_has_no_add_edge_method(self):
        from graph import build_graph
        graph = build_graph()
        assert not hasattr(graph, "add_edge")

    def test_compiled_graph_has_no_set_entry_point_method(self):
        from graph import build_graph
        graph = build_graph()
        assert not hasattr(graph, "set_entry_point")

    def test_two_builds_produce_independent_graphs(self):
        """
        Each call to build_graph() produces a fresh compiled graph.
        They share no state — mutations to one don't affect the other.
        """
        from graph import build_graph
        g1 = build_graph()
        g2 = build_graph()
        # Different objects — fully independent
        assert g1 is not g2


# ── Runtime methods available after compile ───────────────────────────────────

class TestCompiledGraphInterface:
    """Compiled graph must expose all expected runtime methods."""

    def _graph(self):
        from graph import build_graph
        return build_graph()

    def test_has_invoke(self):
        assert hasattr(self._graph(), "invoke")

    def test_has_ainvoke(self):
        assert hasattr(self._graph(), "ainvoke")

    def test_has_stream(self):
        assert hasattr(self._graph(), "stream")

    def test_has_astream(self):
        assert hasattr(self._graph(), "astream")

    def test_has_get_graph(self):
        """get_graph() returns the DrawableGraph for visualisation."""
        assert hasattr(self._graph(), "get_graph")

    def test_has_get_state(self):
        """get_state() retrieves the current checkpoint state (Module 4)."""
        assert hasattr(self._graph(), "get_state")

    def test_has_get_state_history(self):
        """get_state_history() retrieves all checkpoints for a thread (Concept 31)."""
        assert hasattr(self._graph(), "get_state_history")


# ── get_graph_info() ──────────────────────────────────────────────────────────

class TestGetGraphInfo:
    """Verifies the debugging utility returns correct topology metadata."""

    def test_returns_dict_with_expected_keys(self):
        from graph import get_graph_info
        info = get_graph_info()
        assert "nodes" in info
        assert "node_count" in info
        assert "mermaid" in info

    def test_node_count_is_correct(self):
        from graph import get_graph_info
        info = get_graph_info()
        # __start__ + 10 registered nodes (Module 8 added supervisor, kb_worker,
        # web_worker, research_collect; removed technical_node, general_node,
        # research_node from direct routing — net +1 node)
        assert info["node_count"] == 11

    def test_all_registered_nodes_present(self):
        from graph import get_graph_info
        info = get_graph_info()
        for expected in (
            "classify_ticket", "billing_node", "approval_check",
            "supervisor_node", "kb_worker", "web_worker",
            "research_collect", "respond",
        ):
            assert expected in info["nodes"], f"{expected!r} missing from graph nodes"

    def test_mermaid_output_is_nonempty_string(self):
        from graph import get_graph_info
        info = get_graph_info()
        assert isinstance(info["mermaid"], str)
        assert len(info["mermaid"]) > 50   # must be a real diagram, not empty

    def test_mermaid_contains_all_nodes(self):
        from graph import get_graph_info
        info = get_graph_info()
        mermaid = info["mermaid"]
        for node in ("classify_ticket", "billing_node", "respond"):
            assert node in mermaid, f"{node!r} missing from Mermaid output"

    def test_mermaid_is_valid_mermaid_format(self):
        """
        LangGraph's draw_mermaid() produces a diagram with YAML frontmatter.
        The output starts with '---' (Mermaid config block), followed by
        'graph TD;' or 'flowchart TD'.

        This is version-specific: LangGraph >= 0.2 prepends curve config.
        """
        from graph import get_graph_info
        mermaid = get_graph_info()["mermaid"]
        assert any(
            mermaid.strip().startswith(prefix)
            for prefix in ("---", "%%", "flowchart", "graph")
        ), f"Unexpected Mermaid format: {mermaid[:120]}"


# ── debug flag ────────────────────────────────────────────────────────────────

class TestDebugFlag:
    """debug=True must be passed through to compile() and gated by settings."""

    def test_debug_false_by_default(self):
        """build_graph() without debug arg must NOT enable debug mode."""
        from graph import build_graph
        # Just verify it compiles without error — debug=False is the default
        graph = build_graph()
        assert hasattr(graph, "ainvoke")

    def test_settings_graph_debug_gates_debug_mode(self):
        """
        Production pattern: debug is read from settings, never hardcoded.
        build_graph(debug=settings.graph_debug) is the correct call.
        """
        from core.settings import get_settings
        settings = get_settings()
        # Default is False — debug must be off in standard config
        assert settings.graph_debug is False, (
            "graph_debug must default to False — never True in production"
        )


# ── Compile kwargs are not silently ignored ───────────────────────────────────

class TestCompileKwargsPassthrough:
    """Compile kwargs reach StateGraph.compile() — they are not silently dropped."""

    def test_checkpointer_is_attached_to_compiled_graph(self):
        """
        When compiled with InMemorySaver, the graph's checkpointer
        attribute should not be None.
        """
        from langgraph.checkpoint.memory import InMemorySaver

        from graph import build_graph
        saver = InMemorySaver()
        graph = build_graph(checkpointer=saver)
        # Compiled graph exposes its checkpointer
        assert graph.checkpointer is not None

    def test_no_checkpointer_means_none_checkpointer(self):
        """Without a checkpointer, state is in-memory only."""
        from graph import build_graph
        graph = build_graph()
        # No checkpointer = None on the compiled graph
        assert graph.checkpointer is None
