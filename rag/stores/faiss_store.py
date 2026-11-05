"""
Module: rag.stores.faiss_store
Layer:  Infrastructure
Owner:  rag/stores/

Purpose:
    FAISSVectorStore — a concrete VectorStoreProtocol implementation backed by
    FAISS (Facebook AI Similarity Search). FAISS runs entirely locally: no
    network calls, no cost, sub-millisecond retrieval up to ~100k documents.

    Wraps langchain_community's FAISS class and exposes the clean
    VectorStoreProtocol interface. All consumers access this via
    `get_vector_store()` factory — never instantiate directly.

Public API:
    FAISSVectorStore       — wraps LangChain FAISS, satisfies VectorStoreProtocol
    FAISSVectorStore.load  — classmethod: load a persisted index from disk

Dependencies:
    langchain_community.vectorstores  — FAISS wrapper
    llm_provider                      — get_embeddings() (HF sentence-transformers)
    rag.protocols                     — VectorStoreProtocol, RetrievedDocument

LangGraph Concepts demonstrated:
    - Vector Stores (Concept 100) — VectorStoreProtocol + Liskov Substitution
    - Embeddings (Concept 101)    — get_embeddings() only, never instantiate directly
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """
    FAISS-backed vector store satisfying VectorStoreProtocol.

    Construction:
        - From scratch: FAISSVectorStore(embeddings)
        - From persisted index: FAISSVectorStore.load(path, embeddings)

    Why FAISS for local/dev:
        - Zero-cost: no API calls, no rate limits
        - Fast: millisecond retrieval for KB-sized corpora (<100k docs)
        - Simple: single .faiss file, easy to gitignore and rebuild

    Limitation: not suitable for multi-process writes. Use ChromaDB or
    Pinecone for production multi-instance deployments.
    """

    def __init__(self, embeddings: Any) -> None:
        """
        Create an empty FAISS store.

        Args:
            embeddings: HuggingFaceEmbeddings instance from get_embeddings().
                        Never pass a raw model name — always use get_embeddings().
        """
        self._embeddings = embeddings
        self._store: Any | None = None  # langchain_community FAISS instance
        logger.debug("faiss_store: initialized (empty, no index loaded)")

    @classmethod
    def load(cls, path: str, embeddings: Any) -> FAISSVectorStore:
        """
        Load a persisted FAISS index from disk.

        Args:
            path:       Directory path to the saved index (e.g. "indexes/kb.faiss").
            embeddings: HuggingFaceEmbeddings from get_embeddings().

        Returns:
            FAISSVectorStore with the loaded index ready for similarity_search.

        Raises:
            FileNotFoundError: If the index directory does not exist.
            RuntimeError:      If the FAISS index files are corrupted.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"FAISSVectorStore.load: index not found at '{path}'. "
                "Run the ingestion pipeline first: python -m rag.ingest"
            )

        try:
            from langchain_community.vectorstores import FAISS

            instance = cls(embeddings)
            instance._store = FAISS.load_local(
                path,
                embeddings,
                allow_dangerous_deserialization=True,  # safe: we built this index ourselves
            )
            logger.info("faiss_store: loaded index from path=%s", path)
            return instance
        except Exception as e:
            raise RuntimeError(f"FAISSVectorStore.load: failed to load index: {e}") from e

    def add_documents(self, docs: list[Document]) -> None:
        """
        Add documents to the FAISS index.

        If the store is empty, creates a new index from the documents.
        If the store already has documents, merges them in.

        Args:
            docs: LangChain Documents to embed and index.

        Raises:
            ValueError: If docs list is empty.
        """
        if not docs:
            raise ValueError("FAISSVectorStore.add_documents: docs list is empty")

        from langchain_community.vectorstores import FAISS

        logger.info("faiss_store: adding doc_count=%d documents", len(docs))

        if self._store is None:
            self._store = FAISS.from_documents(docs, self._embeddings)
            logger.debug("faiss_store: created new index from %d docs", len(docs))
        else:
            new_store = FAISS.from_documents(docs, self._embeddings)
            self._store.merge_from(new_store)
            logger.debug("faiss_store: merged %d docs into existing index", len(docs))

    def similarity_search(self, query: str, k: int = 20) -> list[Document]:
        """
        Return top-k most similar documents.

        Args:
            query: Natural-language query string.
            k:     Number of documents to return. Default 20 (before reranking).

        Returns:
            List of LangChain Documents ordered by similarity (highest first).

        Raises:
            RuntimeError: If no index has been loaded or built yet.
        """
        if self._store is None:
            raise RuntimeError(
                "FAISSVectorStore: no index loaded. "
                "Call add_documents() or FAISSVectorStore.load() first."
            )
        results = self._store.similarity_search(query, k=k)
        logger.debug("faiss_store: query=%r returned doc_count=%d", query[:50], len(results))
        return results

    def similarity_search_with_score(
        self, query: str, k: int = 20
    ) -> list[tuple[Document, float]]:
        """
        Return top-k documents with their similarity scores.

        Args:
            query: Natural-language query string.
            k:     Number of documents to return.

        Returns:
            List of (Document, score) tuples. Score is L2 distance —
            lower is MORE similar for FAISS (unlike cosine similarity).
        """
        if self._store is None:
            raise RuntimeError(
                "FAISSVectorStore: no index loaded. "
                "Call add_documents() or FAISSVectorStore.load() first."
            )
        return self._store.similarity_search_with_score(query, k=k)

    def save(self, path: str) -> None:
        """
        Persist the FAISS index to disk.

        Args:
            path: Directory path (created if it doesn't exist).

        Raises:
            RuntimeError: If no index has been built yet.
        """
        if self._store is None:
            raise RuntimeError(
                "FAISSVectorStore.save: no index to save. "
                "Call add_documents() first."
            )
        os.makedirs(path, exist_ok=True)
        self._store.save_local(path)
        logger.info("faiss_store: saved index to path=%s", path)

    @property
    def is_loaded(self) -> bool:
        """True if an index has been built or loaded."""
        return self._store is not None

    def to_retriever(self, search_kwargs: dict | None = None) -> Any:
        """
        Return a LangChain Retriever interface for use with LCEL chains.

        Args:
            search_kwargs: e.g. {"k": 5} to override the default k.

        Returns:
            VectorStoreRetriever compatible with .invoke() / .batch().
        """
        if self._store is None:
            raise RuntimeError("FAISSVectorStore: no index loaded.")
        kwargs = search_kwargs or {"k": 20}
        return self._store.as_retriever(search_kwargs=kwargs)
