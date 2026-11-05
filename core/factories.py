"""
Module: core.factories
Layer:  Infrastructure
Owner:  core/

Purpose:
    Factory functions for all infrastructure dependencies:
      - Checkpointers (InMemorySaver, SqliteSaver)
      - Vector store (FAISS — Module 9)
      - Reranker     (cross-encoder — Module 9)

    Factory Pattern:
      Each factory encapsulates construction logic for one dependency.
      Callers (nodes, graph.py, main.py) never import concrete classes —
      they call the factory, which resolves the right implementation from settings.

    Why factories instead of direct imports?
      - Single place to change implementations (swap SQLite → Postgres in one line)
      - lru_cache makes each factory a singleton — same instance reused everywhere
      - Tests inject mocks by patching the factory, not every call site
      - Dependency Inversion: callers depend on the factory contract, not the class

    Why NOT lru_cache on the checkpointer?
      SqliteSaver holds an open database connection. lru_cache would reuse
      the same connection across tests, causing "database is locked" errors.
      get_checkpointer() creates a fresh instance each call for correctness.
      In production, the caller (main.py) holds the reference for the session.

Public API:
    get_checkpointer()  — returns the active checkpointer backend
    get_vector_store()  — returns the active vector store (Module 9 stub)

LangGraph Concepts demonstrated:
    - Graph Memory (Concept 22)      — checkpointers are the storage layer
    - Checkpointing (Concept 23)     — get_checkpointer() wires it all together
    - InMemorySaver (Concept 24)     — memory backend for tests
    - SQLite Checkpointer (Concept 25) — sqlite backend for local dev
"""
from __future__ import annotations

import logging

from core.settings import get_settings

logger = logging.getLogger(__name__)


# ── Checkpointer factory ──────────────────────────────────────────────────────

def get_checkpointer():
    """
    Returns the active LangGraph checkpointer based on settings.

    The checkpointer_backend setting controls which implementation is returned:
      "memory" → MemorySaver   (no disk I/O, lost on process exit)
      "sqlite" → SqliteSaver   (persists to SQLite, survives restarts)

    Why NOT lru_cache here?
      SqliteSaver holds an open SQLite connection. Caching it would reuse
      the same connection object across test cases, causing:
        sqlite3.OperationalError: database is locked
      main.py holds the checkpointer reference for the session lifetime.
      Each call creates a fresh, correctly-isolated instance.

    Why NOT create_checkpointer() as a context manager?
      SqliteSaver works both as a context manager AND standalone.
      We use standalone for simplicity; the connection is closed when
      the object is garbage-collected (sufficient for CLI use).

    Returns:
        BaseCheckpointSaver: InMemorySaver or SqliteSaver instance.

    Raises:
        ValueError: If checkpointer_backend is an unknown value.
    """
    settings = get_settings()
    backend = settings.checkpointer_backend

    if backend == "memory":
        from langgraph.checkpoint.memory import MemorySaver
        logger.debug("get_checkpointer: backend=memory (no persistence)")
        return MemorySaver()

    if backend == "sqlite":
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver
        logger.info(
            "get_checkpointer: backend=sqlite path=%s",
            settings.sqlite_path,
        )
        conn = sqlite3.connect(settings.sqlite_path, check_same_thread=False)
        return SqliteSaver(conn)

    raise ValueError(
        f"Unknown checkpointer_backend: {backend!r}. "
        f"Valid values: 'memory', 'sqlite'."
    )


# ── Vector store factory (Module 9) ───────────────────────────────────────────

def get_vector_store():
    """
    Returns the active vector store for RAG retrieval.

    Resolves the implementation from settings.rag_backend:
      "faiss" → FAISSVectorStore loaded from settings.rag_index_path

    Returns None (not an error) when the index hasn't been built yet.
    This allows the app to start before the ingest pipeline has been run.
    The retriever_node handles None gracefully — returns empty results.

    Why NOT @lru_cache here?
      The FAISS index may be rebuilt between calls (during ingest runs).
      Caching would serve stale data. Callers that need a persistent reference
      (e.g. kb_worker_node) should hold their own reference.

    Returns:
        FAISSVectorStore instance, or None if index not built yet.

    Raises:
        ValueError: If rag_backend is an unknown value.
    """
    import os

    settings = get_settings()
    backend = settings.rag_backend

    if backend == "faiss":
        index_path = settings.rag_index_path
        if not os.path.exists(index_path):
            logger.debug(
                "get_vector_store: no FAISS index at path=%s "
                "— run: python -m rag.ingest",
                index_path,
            )
            return None

        try:
            from llm_provider import get_embeddings
            from rag.stores.faiss_store import FAISSVectorStore

            embeddings = get_embeddings()
            store = FAISSVectorStore.load(index_path, embeddings)
            logger.info("get_vector_store: loaded FAISS store from path=%s", index_path)
            return store
        except Exception as e:
            logger.warning("get_vector_store: failed to load FAISS store: %s", e)
            return None

    raise ValueError(
        f"Unknown rag_backend: {backend!r}. Valid values: 'faiss'."
    )
