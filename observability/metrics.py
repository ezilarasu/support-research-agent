"""
Module: observability.metrics
Layer:  Service (observability)
Owner:  observability/

Purpose:
    Lightweight in-process metrics for support-research-agent.

    Collects per-session and per-node performance data without an
    external metrics backend. Data is accumulated in memory and
    accessible via get_session_metrics() after each graph run.

    Production extension point: replace _store with a Prometheus
    push or a Datadog client by swapping the MetricsStore implementation.

    Three metric types:
      1. Node latency   — time each node takes to execute (ms)
      2. Token usage    — input/output tokens per LLM call
      3. Retrieval hits — whether RAG returned relevant documents

    Integration with hooks (Concept 19 — Observer Pattern):
      Node latency is collected via a post-hook registered at startup.
      This separates metric collection from business logic — nodes
      never import this module.

Public API:
    configure_metrics()          — register hooks at startup
    get_session_metrics(thread_id) — returns collected metrics
    reset_metrics(thread_id)     — clear metrics for a session
    record_node_latency(...)     — called by the post-hook
    record_token_usage(...)      — called from respond_node (optional)

Dependencies:
    observability.hooks  — pre/post hook registration
    core.settings        — graph_debug gates verbose output

LangGraph Concepts demonstrated:
    - Graph Observability (Concept 117) — callback-based metrics
    - Hooks (Concept 19)               — Observer Pattern integration
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)

# ── In-memory metrics store ───────────────────────────────────────────────────
# thread_id → list of metric events
# defaultdict(list) means unknown thread IDs auto-create an empty list
_node_latencies: dict[str, list[dict]] = defaultdict(list)
_token_usages: dict[str, list[dict]] = defaultdict(list)
_retrieval_events: dict[str, list[dict]] = defaultdict(list)


# ── Recording functions (called by hooks and nodes) ───────────────────────────

def record_node_latency(
    node_name: str,
    duration_ms: float,
    thread_id: str = "unknown",
) -> None:
    """
    Records how long a node took to execute.

    Called by the post-hook registered in configure_metrics().
    Never called directly from node code — nodes stay pure.

    Args:
        node_name:   Name of the node that completed.
        duration_ms: Execution time in milliseconds.
        thread_id:   Session identifier for grouping.
    """
    _node_latencies[thread_id].append({
        "node": node_name,
        "duration_ms": round(duration_ms, 2),
        "ts": time.time(),
    })
    logger.debug("metrics: node=%s duration_ms=%.1f thread=%s", node_name, duration_ms, thread_id)


def record_token_usage(
    node_name: str,
    input_tokens: int,
    output_tokens: int,
    thread_id: str = "unknown",
    model: str = "unknown",
) -> None:
    """
    Records token consumption for an LLM call.

    Nodes that use `llm.astream()` can extract token counts from
    `AIMessageChunk.usage_metadata` and call this function.
    Optional — token counting is available in LangSmith traces too.

    Args:
        node_name:     Node that made the LLM call.
        input_tokens:  Prompt tokens consumed.
        output_tokens: Completion tokens generated.
        thread_id:     Session identifier.
        model:         Model ID (for cost attribution).
    """
    _token_usages[thread_id].append({
        "node": node_name,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "model": model,
        "ts": time.time(),
    })
    logger.debug(
        "metrics: tokens node=%s in=%d out=%d model=%s",
        node_name,
        input_tokens,
        output_tokens,
        model,
    )


def record_retrieval_event(
    query: str,
    doc_count: int,
    grade: str,
    thread_id: str = "unknown",
) -> None:
    """
    Records a RAG retrieval outcome.

    Called from kb_worker_node after retrieval completes.

    Args:
        query:     The search query used.
        doc_count: Number of documents retrieved.
        grade:     Retrieval grade: "relevant" | "irrelevant" | "partial".
        thread_id: Session identifier.
    """
    _retrieval_events[thread_id].append({
        "query_len": len(query),
        "doc_count": doc_count,
        "grade": grade,
        "ts": time.time(),
    })


# ── Session metrics summary ───────────────────────────────────────────────────

def get_session_metrics(thread_id: str) -> dict[str, Any]:
    """
    Returns aggregated metrics for a given session.

    Args:
        thread_id: Session identifier to retrieve metrics for.

    Returns:
        Dict with:
          node_latencies:    list of {node, duration_ms} per node execution
          total_duration_ms: sum of all node latencies
          slowest_node:      (node_name, duration_ms) or None
          token_usages:      list of token usage events
          total_tokens:      sum of all tokens consumed
          retrieval_events:  list of retrieval outcomes
    """
    latencies = _node_latencies.get(thread_id, [])
    tokens = _token_usages.get(thread_id, [])
    retrievals = _retrieval_events.get(thread_id, [])

    total_duration = sum(e["duration_ms"] for e in latencies)
    total_tokens = sum(e["total_tokens"] for e in tokens)

    slowest = None
    if latencies:
        slowest_event = max(latencies, key=lambda e: e["duration_ms"])
        slowest = (slowest_event["node"], slowest_event["duration_ms"])

    return {
        "thread_id": thread_id,
        "node_latencies": latencies,
        "total_duration_ms": round(total_duration, 2),
        "slowest_node": slowest,
        "token_usages": tokens,
        "total_tokens": total_tokens,
        "retrieval_events": retrievals,
    }


def reset_metrics(thread_id: str) -> None:
    """Clears all collected metrics for a session. Use between test runs."""
    _node_latencies.pop(thread_id, None)
    _token_usages.pop(thread_id, None)
    _retrieval_events.pop(thread_id, None)


# ── Hook registration ─────────────────────────────────────────────────────────

def configure_metrics() -> None:
    """
    Registers the node latency post-hook.

    Call ONCE at app startup (main.py or api/routes.py).
    After this, every node execution automatically records its duration.

    Observer Pattern (Concept 19):
      The post-hook receives (node_name, state_with_duration, config)
      after every node completes. `node_with_hooks()` in hooks.py
      adds `duration_ms` to the state dict passed to post-hooks.
    """
    from observability.hooks import register_post_hook

    def _latency_post_hook(node_name: str, state: dict, config: dict) -> None:
        """Post-hook: extracts duration_ms and records it as a metric."""
        duration_ms = state.get("duration_ms", 0.0)
        thread_id = config.get("configurable", {}).get("thread_id", "unknown")
        if duration_ms > 0:
            record_node_latency(node_name, duration_ms, thread_id)

    register_post_hook(_latency_post_hook)
    logger.debug("metrics: latency post-hook registered")
