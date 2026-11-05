"""
Module: rag.protocols
Layer:  Service
Owner:  rag/

Purpose:
    Defines the Protocols (structural interfaces) for the RAG pipeline.
    Every concrete implementation (FAISSVectorStore, FixedSizeChunker, etc.)
    must satisfy these contracts. Callers depend on these abstractions — never
    on concrete classes directly (Dependency Inversion Principle).

    This file is the contract definition layer. No I/O or business logic here.

Public API:
    VectorStoreProtocol    — any vector store (FAISS, Chroma, etc.)
    ChunkingStrategy       — any document chunker
    RetrievalResult        — typed return value for retrieval operations
    GradeResult            — typed return value for grading operations

LangGraph Concepts demonstrated:
    - RAG Subgraphs (Concept 89) — protocol-based decoupling of RAG stages
    - Vector Stores (Concept 100) — VectorStoreProtocol + LSP
    - Document Ingestion Pipeline (Concept 99) — ChunkingStrategy protocol
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypedDict, runtime_checkable

if TYPE_CHECKING:
    from langchain_core.documents import Document

# ── Typed return structures ─────────────────────────────────────────────────

class RetrievedDocument(TypedDict):
    """A single document retrieved from the knowledge base."""
    content: str
    source: str
    score: float
    metadata: dict[str, Any]


class RetrievalResult(TypedDict):
    """Structured result from a full retrieval operation (retrieve + rerank)."""
    documents: list[RetrievedDocument]
    query_used: str
    strategy: str
    duration_ms: float


class GradeResult(TypedDict):
    """Result from a retrieval grader or hallucination grader."""
    grade: str                   # "relevant" | "irrelevant" | "partial"
    reason: str
    scores: list[float]


# ── Vector store protocol ────────────────────────────────────────────────────

@runtime_checkable
class VectorStoreProtocol(Protocol):
    """
    Structural interface for vector stores.

    Any class implementing these three methods satisfies this protocol —
    no inheritance required. @runtime_checkable means isinstance() works.

    Why Protocol instead of ABC:
    - LangChain's built-in stores (FAISS, Chroma) don't extend our ABC.
    - Protocol lets us wrap them without modifying third-party code.
    - `isinstance(faiss_store, VectorStoreProtocol)` → True if it matches.
    """

    def similarity_search(self, query: str, k: int) -> list[Document]:
        """Return the top-k most similar documents for the query."""
        ...

    def add_documents(self, docs: list[Document]) -> None:
        """Add documents to the vector store."""
        ...

    def save(self, path: str) -> None:
        """Persist the vector store to disk."""
        ...


# ── Chunking strategy protocol ───────────────────────────────────────────────

@runtime_checkable
class ChunkingStrategy(Protocol):
    """
    Structural interface for document chunkers.

    Strategy Pattern: different chunking algorithms (fixed-size, semantic,
    sentence-boundary) all satisfy this single interface. The ingestion
    pipeline receives a ChunkingStrategy and calls .split() — it never
    knows which concrete implementation it's using.
    """

    def split(self, documents: list[Document]) -> list[Document]:
        """
        Split documents into chunks.

        Args:
            documents: Source documents (may be large, multi-page content).

        Returns:
            Chunked documents — same type, smaller content, metadata preserved.
        """
        ...
