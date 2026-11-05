"""
Module: rag.ingest
Layer:  Service
Owner:  rag/

Purpose:
    Document ingestion pipeline. Loads raw KB documents, chunks them via
    the configured ChunkingStrategy, embeds them, and saves a FAISS index.

    This is a one-time (or periodic) offline operation, not part of the
    live request path. Run as: python -m rag.ingest

    Pipeline stages:
        1. Load — read files from data/kb/
        2. Chunk — split via ChunkingStrategy (FixedSizeChunker by default)
        3. Embed + Index — build FAISS index via FAISSVectorStore.add_documents()
        4. Persist — save index to indexes/kb.faiss

Public API:
    build_index(docs, strategy, index_path)  — main entry point
    load_kb_documents(data_dir)              — loads raw markdown files

Dependencies:
    core.settings         — rag_* configuration values
    core.factories        — get_vector_store() to avoid direct instantiation
    rag.chunking          — ChunkingStrategy protocol + FixedSizeChunker
    rag.protocols         — ChunkingStrategy type annotation
    llm_provider          — get_embeddings() for HF sentence-transformers

LangGraph Concepts demonstrated:
    - Document Ingestion Pipeline (Concept 99) — Strategy pattern for chunking
    - Embeddings (Concept 101)                 — get_embeddings() abstraction
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.documents import Document

from core.settings import get_settings

if TYPE_CHECKING:
    from rag.protocols import ChunkingStrategy

logger = logging.getLogger(__name__)


def load_kb_documents(data_dir: str = "data/kb") -> list[Document]:
    """
    Load all markdown files from the KB directory as LangChain Documents.

    Each file becomes one Document. Metadata includes the source path and
    filename so chunks can be traced back to their origin document.

    Args:
        data_dir: Path to the knowledge base directory.

    Returns:
        List of Documents. Empty list if directory is missing or empty.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.warning("ingest: data directory not found at path=%s", data_dir)
        return []

    documents: list[Document] = []
    for file_path in sorted(data_path.glob("**/*.md")):
        try:
            content = file_path.read_text(encoding="utf-8")
            doc = Document(
                page_content=content,
                metadata={
                    "source": str(file_path),
                    "filename": file_path.name,
                    "file_stem": file_path.stem,
                },
            )
            documents.append(doc)
            logger.debug("ingest: loaded file=%s chars=%d", file_path.name, len(content))
        except OSError as e:
            logger.warning("ingest: failed to read file=%s error=%s", file_path, e)

    logger.info("ingest: loaded doc_count=%d from dir=%s", len(documents), data_dir)
    return documents


def build_index(
    docs: list[Document] | None = None,
    strategy: ChunkingStrategy | None = None,
    index_path: str | None = None,
) -> None:
    """
    Build and persist a FAISS index from KB documents.

    This is the main ingestion entry point. Call with no arguments to use
    all defaults from settings.

    Args:
        docs:       Documents to index. If None, loads from data/kb/.
        strategy:   ChunkingStrategy to use. If None, uses FixedSizeChunker
                    with settings.rag_chunk_size and settings.rag_chunk_overlap.
        index_path: Path to save the index. If None, uses settings.rag_index_path.

    Raises:
        ValueError:  If no documents are found after loading.
        RuntimeError: If embedding or FAISS indexing fails.
    """
    settings = get_settings()
    _index_path = index_path or settings.rag_index_path

    # ── 1. Load documents ───────────────────────────────────────────────────
    if docs is None:
        docs = load_kb_documents()
    if not docs:
        raise ValueError(
            "build_index: no documents to index. "
            "Add markdown files to data/kb/ first."
        )
    logger.info("ingest: starting build_index doc_count=%d", len(docs))

    # ── 2. Chunk ────────────────────────────────────────────────────────────
    if strategy is None:
        from rag.chunking.strategies import FixedSizeChunker
        strategy = FixedSizeChunker(
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
        )

    chunks = strategy.split(docs)
    logger.info("ingest: chunked doc_count=%d → chunk_count=%d", len(docs), len(chunks))

    # ── 3. Embed + index ────────────────────────────────────────────────────
    from llm_provider import get_embeddings
    from rag.stores.faiss_store import FAISSVectorStore

    embeddings = get_embeddings()
    store = FAISSVectorStore(embeddings)

    try:
        store.add_documents(chunks)
    except Exception as e:
        raise RuntimeError(f"build_index: embedding/indexing failed: {e}") from e

    # ── 4. Persist ──────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(_index_path) or ".", exist_ok=True)
    store.save(_index_path)
    logger.info("ingest: index saved to path=%s", _index_path)


if __name__ == "__main__":
    import sys

    from core.logging import configure_logging

    configure_logging()
    logger.info("ingest: starting KB index build")
    try:
        build_index()
        logger.info("ingest: completed successfully")
    except Exception as e:
        logger.error("ingest: failed — %s", e)
        sys.exit(1)
