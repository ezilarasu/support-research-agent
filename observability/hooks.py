"""
Module: observability.hooks
Layer:  Infrastructure
Owner:  observability/

Purpose:
    Observer pattern implementation for LangGraph node observability.

    Problem without hooks:
        Every node that needs latency logging, audit recording, or tracing
        would have to import and call observability code directly. This:
          - Couples business logic to infrastructure concerns
          - Makes it hard to add/remove observability across all nodes at once
          - Violates Single Responsibility — nodes should process, not instrument

    Solution: hooks
        Nodes are wrapped once with node_with_hooks().
        Pre-hooks run before the node. Post-hooks run after.
        To add latency logging to ALL nodes: register one post-hook.
        No node code changes.

    Design: Observer Pattern
        Observers (_pre_hooks, _post_hooks) are registered at startup.
        The wrapped node calls all observers automatically.
        Observers and the node are completely decoupled.

Public API:
    register_pre_hook(hook)     — called BEFORE every wrapped node
    register_post_hook(hook)    — called AFTER every wrapped node
    node_with_hooks(name, fn)   — wraps a node function with hooks
    configure_hooks()           — registers built-in latency + audit hooks
    clear_hooks()               — resets hook lists (tests only)

LangGraph Concepts demonstrated:
    - Hooks (Concept 19) — Observer pattern for decoupled observability
    - RunnableConfig (Concept 20) — hooks receive config for thread_id extraction
    - Runtime Configuration (Concept 21) — hooks gated by settings flags
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable

from langchain_core.runnables import RunnableConfig

from core.settings import get_settings

logger = logging.getLogger(__name__)


# ── Hook type ─────────────────────────────────────────────────────────────────
# A hook is any callable that accepts (node_name, state_dict, config).
# It returns nothing — hooks are side-effect only (logging, metrics, tracing).
# Hooks must be FAST — they run synchronously on every node execution.

NodeHook = Callable[[str, dict, RunnableConfig], None]

# ── Hook registries ───────────────────────────────────────────────────────────
# Module-level lists — hooks registered here persist for the app lifetime.
# Tests use clear_hooks() to reset between test cases.

_pre_hooks:  list[NodeHook] = []
_post_hooks: list[NodeHook] = []


# ── Public registration API ───────────────────────────────────────────────────

def register_pre_hook(hook: NodeHook) -> None:
    """
    Register a function to be called BEFORE every node executes.

    Called with (node_name, state, config) just before the node runs.
    Use for: input validation logging, auth checks, request tracing.

    Args:
        hook: Callable(node_name: str, state: dict, config: RunnableConfig) -> None
    """
    _pre_hooks.append(hook)
    logger.debug("register_pre_hook: registered %s", hook.__name__)


def register_post_hook(hook: NodeHook) -> None:
    """
    Register a function to be called AFTER every node returns.

    Called with (node_name, result_dict, config) after the node returns.
    The result_dict includes a "duration_ms" key added by node_with_hooks.
    Use for: latency logging, metrics emission, result auditing.

    Args:
        hook: Callable(node_name: str, result: dict, config: RunnableConfig) -> None
    """
    _post_hooks.append(hook)
    logger.debug("register_post_hook: registered %s", hook.__name__)


# ── Node wrapper ──────────────────────────────────────────────────────────────

def node_with_hooks(name: str, fn: Callable) -> Callable:
    """
    Wraps a node function with all registered pre/post hooks.

    Usage in graph.py:
        builder.node("classify", node_with_hooks("classify", classify_ticket_node))

    The wrapper:
      1. Calls all pre-hooks with (name, state, config)
      2. Records start time
      3. Calls the original node function
      4. Computes duration_ms
      5. Calls all post-hooks with (name, {**result, duration_ms=...}, config)
      6. Returns the original node result (hooks don't modify it)

    Args:
        name: Node name — used in hook calls and log messages.
        fn:   The async node function to wrap.

    Returns:
        An async node function with hooks applied.
    """
    import functools

    @functools.wraps(fn)   # preserves signature so LangGraph sees (state, config)
    async def wrapped(state: dict, config: RunnableConfig) -> dict:
        # ── Pre-hooks ─────────────────────────────────────────────────────────
        for hook in _pre_hooks:
            try:
                hook(name, state, config)
            except Exception as e:
                # Hooks must never crash the node — log and continue
                logger.warning("pre_hook failed: hook=%s node=%s error=%s", hook.__name__, name, e)

        # ── Node execution ────────────────────────────────────────────────────
        start = time.perf_counter()
        result = await fn(state, config)
        duration_ms = (time.perf_counter() - start) * 1000

        # ── Post-hooks ────────────────────────────────────────────────────────
        # Inject duration_ms into the result dict so post-hooks can read it.
        # This doesn't change what LangGraph sees — it's only passed to hooks.
        result_with_timing = {**result, "duration_ms": duration_ms}

        for hook in _post_hooks:
            try:
                hook(name, result_with_timing, config)
            except Exception as e:
                logger.warning("post_hook failed: hook=%s node=%s error=%s", hook.__name__, name, e)

        return result   # return the original result — duration_ms not in state

    return wrapped


# ── Built-in hooks ────────────────────────────────────────────────────────────

def _latency_hook(node_name: str, result: dict, config: RunnableConfig) -> None:
    """
    Post-hook: logs node execution latency.

    Gated behind settings.graph_debug — only emits in debug mode.
    In production, latency is captured by LangSmith tracing (Module 10).
    """
    settings = get_settings()
    if not settings.graph_debug:
        return

    duration_ms = result.get("duration_ms", 0.0)
    thread_id = config.get("configurable", {}).get("thread_id", "?")
    logger.info(
        "node_latency: node=%s thread=%s duration_ms=%.1f",
        node_name,
        thread_id,
        duration_ms,
    )


def _audit_hook(node_name: str, state: dict, config: RunnableConfig) -> None:
    """
    Pre-hook: logs that a node is about to execute.

    Always active — audit trail should not be gated behind debug mode.
    Low overhead: single log line per node execution.
    """
    thread_id = config.get("configurable", {}).get("thread_id", "?")
    tenant_id = config.get("configurable", {}).get("tenant_id", "default")
    logger.debug(
        "node_enter: node=%s thread=%s tenant=%s",
        node_name,
        thread_id,
        tenant_id,
    )


# ── Configure built-in hooks ──────────────────────────────────────────────────

def configure_hooks() -> None:
    """
    Registers all built-in hooks.

    Call once at app startup in main.py, after configure_logging().
    Order matters: hooks are called in registration order.

    Example (main.py):
        configure_logging()
        configure_hooks()    ← add this
        graph = build_graph()
    """
    register_pre_hook(_audit_hook)
    register_post_hook(_latency_hook)
    logger.info("configure_hooks: registered %d pre-hooks, %d post-hooks",
                len(_pre_hooks), len(_post_hooks))


def clear_hooks() -> None:
    """
    Clears all registered hooks.
    Tests only — never call in production code.
    """
    _pre_hooks.clear()
    _post_hooks.clear()
