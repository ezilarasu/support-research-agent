"""
Module: memory.store
Layer:  Service
Owner:  memory/

Purpose:
    Long-term semantic memory store for the support-research-agent.

    Uses LangGraph's built-in InMemoryStore with an embed function for
    semantic search. In production, swap to AsyncPostgresStore or a
    custom ChromaDB-backed store by changing get_memory_store() in
    core/factories.py — zero node code changes required (Dependency Inversion).

    Namespace design:
      ("memories", user_id) scopes every user's memories independently.
      ("memories", "alice") and ("memories", "bob") never interfere.
      This is multi-tenancy at the memory layer.

    Why InMemoryStore instead of a custom vector DB?
      LangGraph's InMemoryStore supports both exact-key lookup AND semantic
      search (when an embed function is provided). It's the simplest
      implementation that demonstrates the full concept — swapping backends
      later is a one-line factory change.

Public API:
    LongTermMemoryStore    — wraps LangGraph's store with our SemanticSearchProtocol
    get_memory_store()     — factory: returns the singleton store (lru_cached)

LangGraph Concepts demonstrated:
    - Long-Term Memory (Concept 35)   — store lives outside graph state
    - Memory Store (Concept 36)       — InMemoryStore + embed function
    - Semantic Memory (Concept 37)    — embeddings enable meaning-based search
    - Memory Retrieval (Concept 38)   — asearch() used by retrieval node
    - Memory Summarization (Concept 39) — write_memory() called after summarisation
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from functools import lru_cache

from langgraph.store.memory import InMemoryStore

from core.settings import get_settings

logger = logging.getLogger(__name__)


class LongTermMemoryStore:
    """
    Semantic long-term memory store for a single deployment.

    Wraps LangGraph's InMemoryStore and adds:
      - namespace scoping per user_id (multi-tenant isolation)
      - ISO 8601 timestamps on every write
      - structured logging on reads and writes
      - SemanticSearchProtocol-compatible interface

    In production, replace the inner _store with AsyncPostgresStore
    by changing the factory — this class and all nodes stay the same.
    """

    def __init__(self, store: InMemoryStore) -> None:
        """
        Args:
            store: The underlying LangGraph store. Injected — never created here.
                   This enables testing with a mock store without subclassing.
        """
        self._store = store

    # ── Write ──────────────────────────────────────────────────────────────────

    async def astore(
        self,
        user_id: str,
        key: str,
        text: str,
        metadata: dict | None = None,
    ) -> None:
        """
        Embeds and stores a memory for a user.

        The namespace ("memories", user_id) ensures Alice's memories are
        completely isolated from Bob's — they share the same store instance
        but never see each other's data.

        Args:
            user_id:  The user this memory belongs to.
            key:      A unique identifier, e.g. "preference_billing_format".
                      Upserting with the same key OVERWRITES the previous value.
            text:     The raw memory text to embed and store.
            metadata: Optional extra fields stored alongside the text.
        """
        namespace = ("memories", user_id)
        value = {
            "text": text,
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
            **(metadata or {}),
        }
        await self._store.aput(namespace, key, value)
        logger.info(
            "memory_stored user=%s key=%s text_len=%d",
            user_id,
            key,
            len(text),
        )

    # ── Read ───────────────────────────────────────────────────────────────────

    async def asearch(
        self,
        user_id: str,
        query: str,
        limit: int | None = None,
    ) -> list[dict]:
        """
        Returns the top-k memories most semantically similar to query.

        The embed function attached to the store at construction time is used
        to convert the query to a vector, then cosine similarity ranks results.

        Args:
            user_id: Scopes the search to this user's memories only.
            query:   The natural-language search query.
            limit:   Max results. Defaults to settings.memory_top_k.

        Returns:
            list[dict]: Each dict has "text", "key", "score", and "created_at".
                        Sorted by relevance score, highest first.
        """
        settings = get_settings()
        k = limit if limit is not None else settings.memory_top_k
        namespace = ("memories", user_id)

        # namespace_prefix is positional-only in InMemoryStore.asearch
        results = await self._store.asearch(namespace, query=query, limit=k)

        memories = [
            {
                "text": item.value.get("text", ""),
                "key": item.key,
                "score": getattr(item, "score", 1.0),
                "created_at": item.value.get("created_at", ""),
            }
            for item in results
        ]

        logger.debug(
            "memory_search user=%s query=%r results=%d",
            user_id,
            query[:50],
            len(memories),
        )
        return memories


# ── Factory ───────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_memory_store() -> LongTermMemoryStore:
    """
    Returns the singleton LongTermMemoryStore.

    Why lru_cache here but NOT on get_checkpointer()?
      LongTermMemoryStore wraps an InMemoryStore — a plain Python dict.
      No file handles, no OS-level locks. Safe to share across tests.
      The checkpointer holds an open SQLite file handle — NOT safe to share.

    In production, swap InMemoryStore for AsyncPostgresStore here.
    Zero node code changes required — nodes depend on LongTermMemoryStore,
    not on the underlying store implementation.

    Returns:
        LongTermMemoryStore: The singleton store for this process.
    """
    try:
        # Attempt to create the store with an embed function for semantic search.
        # The embed function converts text → float vector using HuggingFace.
        from llm_provider import get_embeddings
        embeddings = get_embeddings()

        def _embed_fn(texts: list[str]) -> list[list[float]]:
            """
            Adapter: converts LangChain embeddings to the list[list[float]]
            format that LangGraph's InMemoryStore expects.

            LangChain's embed_documents() returns list[list[float]] already —
            this function is just the named bridge between the two APIs.
            """
            return embeddings.embed_documents(texts)

        store = InMemoryStore(index={"embed": _embed_fn, "dims": 384})
        logger.info("get_memory_store: semantic search enabled (HuggingFace embeddings)")

    except Exception as e:
        # Embeddings unavailable (e.g. test environment, no HF model downloaded).
        # Fall back to a store without semantic search — exact-key lookup still works.
        logger.warning(
            "get_memory_store: embeddings unavailable (%s) — "
            "falling back to exact-key store (no semantic search)",
            e,
        )
        store = InMemoryStore()

    return LongTermMemoryStore(store=store)
