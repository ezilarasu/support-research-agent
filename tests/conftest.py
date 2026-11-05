"""
Shared pytest fixtures for all test files.

These fixtures are auto-discovered by pytest — no import needed in test files.
Available to every file under tests/.

Fixtures provided:
    mock_settings        — test-safe settings with fake API keys
    base_state           — minimal valid AgentState for node tests
    runnable_config      — standard RunnableConfig dict for test invocations
    mock_llm             — async mock LLM, returns deterministic AIMessage
    clear_settings_cache — autouse: clears lru_cache before/after every test

Usage in a test:
    async def test_my_node(base_state, runnable_config, mock_llm):
        with patch("agents.nodes.get_research_llm", return_value=mock_llm):
            result = await my_node(base_state, runnable_config)
        assert result["ticket_type"] == "billing"
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from state import AgentState


@pytest.fixture(autouse=True)
def clear_settings_cache():
    """
    Clears the Settings lru_cache before and after every test.

    Why autouse=True?
      Settings is a singleton — if one test patches env vars without
      clearing the cache, the patched value leaks into subsequent tests.
      autouse ensures every test starts with a clean slate.
    """
    from core.settings import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def mock_settings(monkeypatch):
    """
    Injects test-safe settings — no real API keys needed in tests.

    Sets the minimum required env vars to construct a valid Settings instance.
    Use this fixture in any test that imports from core.settings.

    Args:
        monkeypatch: pytest's monkeypatch fixture (auto-injected).

    Returns:
        Settings: The test Settings instance.
    """
    from core.settings import get_settings
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key-not-real")
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    get_settings.cache_clear()
    return get_settings()


@pytest.fixture
def base_state() -> AgentState:
    """
    Minimal valid AgentState for use in node tests.

    All fields initialised to their zero/None values.
    Tests that need specific fields set should do so after receiving this fixture:
        def test_something(base_state):
            base_state["ticket_type"] = "billing"
            ...

    Returns:
        AgentState: A fully populated state with safe default values.
    """
    return AgentState(
        messages=[HumanMessage(content="My invoice is wrong, I was charged twice")],
        ticket_type=None,
        ticket_metadata=None,
        classification_confidence=0.0,
        thread_id="test-thread-001",
        retrieved_memories=[],
        streaming_enabled=False,
        active_agent=None,
        agent_results=[],
        query=None,
        retrieved_docs=[],
        retrieval_grade=None,
        hallucination_check=None,
        pending_approval=None,
        approval_decision=None,
        escalation_level=0,
        authenticated=True,
        rbac_roles=["agent"],
        audit_events=[],
        final_response=None,
        error=None,
    )


@pytest.fixture
def runnable_config() -> dict:
    """
    Standard RunnableConfig for test invocations.

    The "configurable" namespace is where LangGraph looks for thread_id,
    user_id, and other per-invocation metadata (Concept 20).

    Returns:
        dict: A valid RunnableConfig with test thread/user/tenant IDs.
    """
    return {
        "configurable": {
            "thread_id": "test-thread-001",
            "user_id": "test-user-001",
            "tenant_id": "test-tenant",
        },
        "metadata": {"test": True},
    }


@pytest.fixture
def memory_checkpointer():
    """
    InMemorySaver checkpointer for integration tests.

    Why InMemorySaver instead of SqliteSaver in tests?
      - No disk I/O — tests run faster
      - No cleanup needed — garbage collected after the test
      - No "database is locked" errors from parallel test runs
      - Identical API to SqliteSaver (LSP — both implement BaseCheckpointSaver)

    Usage:
        def test_persistence(memory_checkpointer):
            graph = build_graph(checkpointer=memory_checkpointer)
            result = graph.invoke(state, config={"configurable": {"thread_id": "t1"}})

    Returns:
        MemorySaver: A fresh in-memory checkpointer for this test.
    """
    return MemorySaver()


@pytest.fixture
def mock_llm():
    """
    Mock LLM that returns a deterministic AIMessage. No real API calls made.

    Use this wherever a node calls get_research_llm() or get_router_llm().
    Patch the factory function in the node's module, not the LLM class:

        with patch("agents.nodes.get_research_llm", return_value=mock_llm):
            result = await respond_node(state, config)

    Returns:
        AsyncMock: A mock with .ainvoke() returning AIMessage("Mock response").
    """
    mock = AsyncMock()
    mock.ainvoke = AsyncMock(
        return_value=AIMessage(content="Mock response from test LLM")
    )
    return mock
