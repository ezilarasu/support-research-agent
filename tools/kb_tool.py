"""
Module: tools.kb_tool
Layer:  Service
Owner:  tools/

Purpose:
    Internal knowledge base search tool.
    Searches the FAISS vector index over internal support documents.
    Used by technical and research agents before falling back to web search.

    The full RAG pipeline (chunking, embedding, reranking) is built in Module 9.
    This tool wraps it in a LangChain tool interface for use with ToolNode.

Public API:
    KnowledgeBaseTool — LangChain tool wrapping the KB retriever

LangGraph Concepts demonstrated:
    - ToolNode (Concept 16) — this tool is passed to ToolNode
    - Dependency Inversion  — calls get_vector_store() factory, not FAISS directly
"""
from __future__ import annotations

import logging

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from core.settings import get_settings

logger = logging.getLogger(__name__)


# ── Input schema ──────────────────────────────────────────────────────────────

class KBToolInput(BaseModel):
    """Input schema for the knowledge base search tool."""

    query: str = Field(
        ...,
        description="The search query for the internal knowledge base. "
                    "Include specific product names, error codes, or policy names.",
        min_length=3,
        max_length=500,
    )
    top_k: int = Field(
        default=5,
        description="Number of knowledge base documents to retrieve.",
        ge=1,
        le=20,
    )


# ── Tool class ────────────────────────────────────────────────────────────────

class KnowledgeBaseTool(BaseTool):
    """
    Internal knowledge base search tool.

    Searches the FAISS vector index built from internal support documents:
      - billing_policies.md
      - api_reference.md
      - troubleshooting.md
      - gdpr_policy.md
      - faq.md

    Used as the first retrieval step. If results are below the relevance
    threshold, the research agent falls back to SearchTool (web search).

    The full FAISS index is built in Module 9.
    Until then, returns a placeholder indicating the KB is not yet indexed.
    """

    name: str = "knowledge_base_search"
    description: str = (
        "Search the internal support knowledge base. "
        "Use this FIRST before web search — it contains official policies, "
        "API documentation, and troubleshooting guides. "
        "Input: a specific search query. Output: relevant document excerpts."
    )
    args_schema: type[BaseModel] = KBToolInput

    def _run(self, query: str, top_k: int = 5) -> str:
        """
        Searches the FAISS knowledge base index.

        Falls back gracefully if the index hasn't been built yet (pre-Module 9).

        Args:
            query: Search query string.
            top_k: Number of results to return.

        Returns:
            Formatted string of retrieved document excerpts.
        """
        settings = get_settings()

        import os
        if not os.path.exists(settings.rag_index_path):
            logger.info(
                "kb_tool: FAISS index not found at %s — "
                "run build_index() to create it (Module 9)",
                settings.rag_index_path,
            )
            return (
                f"[Knowledge base not yet indexed — build_index() not run]\n"
                f"Query was: {query}\n"
                f"Tip: run scripts/build_kb_index.py to index the knowledge base."
            )

        try:
            from core.factories import get_vector_store
            store = get_vector_store()
            docs = store.similarity_search(query, k=top_k)
            return _format_kb_results(docs)

        except Exception as e:
            logger.error("kb_tool: retrieval failed query=%r error=%s", query, e)
            return f"[KB search failed: {e}]\nQuery was: {query}"

    async def _arun(self, query: str, top_k: int = 5) -> str:
        """Async delegation to sync _run."""
        return self._run(query=query, top_k=top_k)


# ── Pure helper ───────────────────────────────────────────────────────────────

def _format_kb_results(docs: list) -> str:
    """
    Formats retrieved LangChain Documents into a clean string for the LLM.

    Pure function — takes a list of Documents, returns a string.

    Args:
        docs: List of LangChain Document objects from similarity_search.

    Returns:
        Formatted string with source and content for each document.
    """
    if not docs:
        return "No relevant documents found in the knowledge base."

    lines: list[str] = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        content = doc.page_content[:600]   # truncate long chunks
        lines.append(f"[{i}] Source: {source}")
        lines.append(f"    {content}\n")

    return "\n".join(lines)
