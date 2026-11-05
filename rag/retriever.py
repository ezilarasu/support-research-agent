"""
Module: rag.retriever
Layer:  Service
Owner:  rag/

Purpose:
    Retrieval node factory. make_retriever_node() returns an async LangGraph
    node function with dependencies injected — making it testable without
    a real FAISS index.

    Two-stage retrieval:
        Stage 1: Similarity search — retrieves top_k_retrieve candidates
                 from the vector store using cosine similarity.
        Stage 2: Reranking — narrows to top_k_rerank using a cross-encoder
                 (ms-marco-MiniLM). Cross-encoders score query+doc jointly,
                 not independently, giving much better precision than
                 bi-encoder similarity alone.

Public API:
    make_retriever_node(store, reranker) — DI factory returning a node fn
    retriever_node                       — default production node (lazy-loaded)

Dependencies:
    core.settings     — rag_top_k_retrieve, rag_top_k_rerank, rag_score_threshold
    core.factories    — get_vector_store()
    llm_provider      — get_reranker()
    rag.protocols     — VectorStoreProtocol, RetrievedDocument, RetrievalResult
    state             — AgentState

LangGraph Concepts demonstrated:
    - Retrieval Nodes (Concept 90)  — make_retriever_node() DI factory
    - Reranking (Concept 102)       — cross-encoder post-retrieval reranking
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable  # noqa: TC003
from typing import Any

from langchain_core.runnables import RunnableConfig  # noqa: TC002

from core.settings import get_settings
from rag.protocols import RetrievedDocument
from state import AgentState  # noqa: TC001

logger = logging.getLogger(__name__)


def make_retriever_node(
    store: Any | None = None,
    reranker: Any | None = None,
) -> Callable:
    """
    Factory: creates a retrieval+reranking node with injected dependencies.

    Dependency Inversion in action: the node depends on VectorStoreProtocol
    and a reranker, not on FAISSVectorStore or CrossEncoder directly. In tests,
    inject mocks. In production, call with no args — factories resolve them.

    Args:
        store:    VectorStoreProtocol instance. Defaults to get_vector_store().
        reranker: CrossEncoder instance. Defaults to get_reranker().

    Returns:
        An async node function compatible with graph.add_node().

    Example:
        # Production
        builder.node("retrieve", make_retriever_node())

        # Test
        builder.node("retrieve", make_retriever_node(store=mock_store, reranker=mock_reranker))
    """
    def _get_store() -> Any:
        if store is not None:
            return store
        from core.factories import get_vector_store
        return get_vector_store()

    def _get_reranker() -> Any:
        if reranker is not None:
            return reranker
        from llm_provider import get_reranker
        return get_reranker()

    async def retriever_node(state: AgentState, config: RunnableConfig) -> dict:
        """
        Retrieve and rerank documents from the knowledge base.

        Reads:  state["query"]         — the search query (may be rewritten)
        Writes: state["retrieved_docs"] — top-k reranked RetrievedDocuments

        Args:
            state:  Current AgentState. Reads query field.
            config: RunnableConfig with thread_id in configurable.

        Returns:
            Partial state dict with retrieved_docs and retrieval_grade.

        Raises:
            ValueError: If query is empty or missing.
        """
        query = state.get("query") or ""
        thread_id = config.get("configurable", {}).get("thread_id", "unknown")
        settings = get_settings()

        if not query.strip():
            logger.warning("retriever_node: empty query thread=%s", thread_id)
            return {"retrieved_docs": [], "retrieval_grade": "irrelevant"}

        logger.debug("retriever_node: thread=%s query=%r", thread_id, query[:80])
        start = time.perf_counter()

        # ── Stage 1: Similarity search ──────────────────────────────────────
        _store = _get_store()

        try:
            if _store is None:
                logger.warning("retriever_node: no vector store loaded thread=%s", thread_id)
                return {"retrieved_docs": [], "retrieval_grade": "irrelevant"}

            raw_docs = _store.similarity_search(query, k=settings.rag_top_k_retrieve)
        except RuntimeError as e:
            logger.warning("retriever_node: store not ready thread=%s error=%s", thread_id, e)
            return {"retrieved_docs": [], "retrieval_grade": "irrelevant"}

        if not raw_docs:
            logger.info("retriever_node: no docs found thread=%s query=%r", thread_id, query[:50])
            return {"retrieved_docs": [], "retrieval_grade": "irrelevant"}

        # ── Stage 2: Reranking ───────────────────────────────────────────────
        _reranker = _get_reranker()
        reranked = _rerank(query, raw_docs, _reranker, settings.rag_top_k_rerank)

        duration_ms = (time.perf_counter() - start) * 1000

        # Convert to typed RetrievedDocument dicts
        retrieved: list[RetrievedDocument] = [
            RetrievedDocument(
                content=doc.page_content,
                source=doc.metadata.get("source", "unknown"),
                score=float(score),  # cast numpy.float32 → Python float (msgpack-safe)
                metadata=doc.metadata,
            )
            for doc, score in reranked
        ]

        grade = "relevant" if retrieved else "irrelevant"
        logger.info(
            "retriever_node: thread=%s doc_count=%d duration_ms=%.1f grade=%s",
            thread_id,
            len(retrieved),
            duration_ms,
            grade,
        )

        return {
            "retrieved_docs": retrieved,
            "retrieval_grade": grade,
            "query": query,
        }

    return retriever_node


def _rerank(
    query: str,
    docs: list[Any],
    reranker: Any,
    top_k: int,
) -> list[tuple[Any, float]]:
    """
    Rerank documents using a cross-encoder model.

    Cross-encoders score (query, document) pairs jointly — much more accurate
    than bi-encoder similarity alone, at the cost of O(n) model calls.

    Args:
        query:    Search query string.
        docs:     Documents from similarity search.
        reranker: CrossEncoder instance (from get_reranker()).
        top_k:    Number of documents to return after reranking.

    Returns:
        List of (document, score) tuples, sorted by score descending.
    """
    if not docs:
        return []

    try:
        # CrossEncoder.predict takes (query, passage) pairs
        pairs = [[query, doc.page_content] for doc in docs]
        scores = reranker.predict(pairs)

        scored = sorted(
            zip(docs, scores, strict=False),
            key=lambda x: x[1],
            reverse=True,  # highest score = most relevant
        )
        return scored[:top_k]

    except Exception as e:
        logger.warning("reranker: failed, returning raw similarity order. error=%s", e)
        # Fall back to the original similarity order if reranker fails
        return [(doc, 1.0) for doc in docs[:top_k]]


# Default production node (lazy-loads dependencies on first call)
retriever_node = make_retriever_node()
