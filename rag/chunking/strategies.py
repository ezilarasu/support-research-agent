"""
Module: rag.chunking.strategies
Layer:  Service
Owner:  rag/chunking/

Purpose:
    Concrete ChunkingStrategy implementations.

    The Strategy Pattern is applied here: the ingestion pipeline receives
    a ChunkingStrategy and calls .split() without knowing which concrete
    implementation is active. Swapping from FixedSizeChunker to
    SemanticChunker requires only a settings change, zero code changes.

Public API:
    FixedSizeChunker   — splits by character count with overlap (default, fast)
    SemanticChunker    — splits at semantic boundaries (slower, higher quality)

Dependencies:
    langchain_text_splitters  — RecursiveCharacterTextSplitter
    rag.protocols             — ChunkingStrategy Protocol

LangGraph Concepts demonstrated:
    - Document Ingestion Pipeline (Concept 99) — Strategy pattern for chunking
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

if TYPE_CHECKING:
    from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class FixedSizeChunker:
    """
    Chunks documents by character count with configurable overlap.

    Why this is the default:
    - Deterministic: same input → same chunks every time (good for tests)
    - Fast: O(n) in document length, no model calls
    - Sufficient: for support KB content (structured markdown), fixed-size
      chunks work well because each section is naturally self-contained

    RecursiveCharacterTextSplitter tries delimiters in order:
    ["\n\n", "\n", " ", ""] — preferring paragraph splits over word splits.
    This means chunk boundaries land at natural text breaks, not mid-word.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50) -> None:
        """
        Args:
            chunk_size:    Maximum characters per chunk. Maps to ~128 tokens
                           for English text (1 token ≈ 4 chars).
            chunk_overlap: Characters repeated between consecutive chunks.
                           Prevents context loss at chunk boundaries.
        """
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        logger.debug(
            "chunker: FixedSizeChunker init chunk_size=%d overlap=%d",
            chunk_size,
            chunk_overlap,
        )

    def split(self, documents: list[Document]) -> list[Document]:
        """
        Split documents into fixed-size chunks.

        Metadata is inherited by each chunk from its parent document.
        Chunk index is added as metadata["chunk_index"] for traceability.

        Args:
            documents: Source documents to split.

        Returns:
            Chunked documents with preserved metadata.
        """
        if not documents:
            return []

        chunks = self._splitter.split_documents(documents)

        # Add chunk index metadata for traceability in grading/debugging
        source_counts: dict[str, int] = {}
        for chunk in chunks:
            src = chunk.metadata.get("source", "unknown")
            source_counts[src] = source_counts.get(src, 0) + 1
            chunk.metadata["chunk_index"] = source_counts[src] - 1

        logger.info(
            "chunker: fixed_size split doc_count=%d → chunk_count=%d",
            len(documents),
            len(chunks),
        )
        return chunks


class SemanticChunker:
    """
    Chunks documents at semantic boundaries using an embedding model.

    Why prefer this over FixedSizeChunker:
    - Chunks don't cut mid-thought — each chunk is semantically complete
    - Retrieval precision improves because chunks match query semantics
    - Particularly valuable for long-form documentation

    Trade-off: requires an embedding model call per document (slower ingest).
    Use only when ingest latency is acceptable and retrieval quality is critical.

    NOTE: This chunker requires langchain_experimental to be installed.
    It is NOT the default — enable with CHUNKING_STRATEGY=semantic in .env.
    """

    def __init__(self, embeddings: Any) -> None:
        """
        Args:
            embeddings: HuggingFaceEmbeddings from get_embeddings().
                        Never pass a raw model name.
        """
        try:
            from langchain_experimental.text_splitter import SemanticChunker as _SemanticChunker
            self._splitter = _SemanticChunker(embeddings=embeddings)
        except ImportError as e:
            raise ImportError(
                "SemanticChunker requires langchain_experimental. "
                "Install it: pip install langchain-experimental"
            ) from e
        logger.debug("chunker: SemanticChunker initialized with embedding model")

    def split(self, documents: list[Document]) -> list[Document]:
        """
        Split documents at semantic boundaries.

        Args:
            documents: Source documents to split.

        Returns:
            Semantically-bounded chunks with preserved metadata.
        """
        if not documents:
            return []

        chunks = self._splitter.split_documents(documents)
        logger.info(
            "chunker: semantic split doc_count=%d → chunk_count=%d",
            len(documents),
            len(chunks),
        )
        return chunks
