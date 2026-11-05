"""
Module: tools.search_tool
Layer:  Service
Owner:  tools/

Purpose:
    Tavily web search tool for the research agent.
    Used when the internal knowledge base doesn't have sufficient information.
    Tavily is purpose-built for LLM-friendly web search — returns clean,
    structured results rather than raw HTML.

Public API:
    SearchTool  — LangChain tool wrapping Tavily search API

LangGraph Concepts demonstrated:
    - ToolNode (Concept 16) — this tool is passed to ToolNode in research_node
    - Dependency Inversion  — tool depends on TavilySearchProtocol, not SDK directly
"""
from __future__ import annotations

import logging

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from core.settings import get_settings

logger = logging.getLogger(__name__)


# ── Input schema ──────────────────────────────────────────────────────────────
# Pydantic model describing what arguments this tool accepts.
# LangGraph/LangChain uses this to:
#   1. Validate arguments from the LLM before calling the tool
#   2. Generate a JSON schema the LLM sees in its system prompt
#   3. Give you autocompletion when writing tests

class SearchToolInput(BaseModel):
    """Input schema for the web search tool."""

    query: str = Field(
        ...,
        description="The search query to send to Tavily. Be specific — include"
                    " product names, error codes, or policy names for better results.",
        min_length=3,
        max_length=500,
    )
    max_results: int = Field(
        default=5,
        description="Maximum number of search results to return.",
        ge=1,
        le=10,
    )


# ── Tool class ────────────────────────────────────────────────────────────────

class SearchTool(BaseTool):
    """
    Web search tool powered by Tavily.

    Used by the research agent when the internal knowledge base does not
    contain sufficient information to answer a compliance or research question.

    Why Tavily over Google/Bing?
      - Returns clean, structured results (title, content, url) — not raw HTML
      - Purpose-built for LLM consumption — no parsing needed
      - Relevance-ranked by default — top results are most relevant

    Usage in a LangGraph node:
        from langchain_core.tools import ToolNode
        from tools import get_research_tools

        tools = get_research_tools()
        tool_node = ToolNode(tools)

        # Or bind tools to an LLM for ReAct-style agents:
        llm_with_tools = get_research_llm().bind_tools(tools)
    """

    name: str = "web_search"
    description: str = (
        "Search the web for up-to-date information. "
        "Use this when the question requires current information, regulatory updates, "
        "external API documentation, or anything not in the internal knowledge base. "
        "Input: a specific search query. Output: list of relevant web results."
    )
    args_schema: type[BaseModel] = SearchToolInput

    # ── Optional: override to disable parallel tool use ──────────────────────
    # return_direct: bool = False  (default — result goes back to LLM)

    def _run(self, query: str, max_results: int = 5) -> str:
        """
        Synchronous execution of web search.

        LangChain calls _run for sync tool use, _arun for async.
        We implement _run here; _arun is handled by BaseTool default
        (runs _run in a thread pool).

        Args:
            query:       The search query string.
            max_results: Maximum results to return.

        Returns:
            Formatted string of search results for the LLM to read.
        """
        settings = get_settings()

        # Guard: tool is useless without a Tavily API key
        if not settings.tavily_api_key:
            logger.warning("search_tool: TAVILY_API_KEY not set — returning placeholder")
            return (
                f"[Search unavailable — TAVILY_API_KEY not configured]\n"
                f"Query was: {query}"
            )

        try:
            from tavily import TavilyClient

            client = TavilyClient(
                api_key=settings.tavily_api_key.get_secret_value()
            )
            response = client.search(
                query=query,
                max_results=max_results,
                search_depth="advanced",    # advanced = more thorough, slightly slower
                include_answer=True,        # Tavily's own AI summary of results
            )

            return _format_search_results(response)

        except Exception as e:
            logger.error("search_tool: tavily call failed query=%r error=%s", query, e)
            # Return a graceful degradation message rather than crashing the agent
            return f"[Search failed: {e}]\nQuery was: {query}"

    async def _arun(self, query: str, max_results: int = 5) -> str:
        """
        Async execution — delegates to sync _run.

        For production, replace with an async Tavily client call.
        BaseTool's default _arun runs _run in a thread pool — acceptable
        for now, but adds threading overhead.
        """
        return self._run(query=query, max_results=max_results)


# ── Pure helper ───────────────────────────────────────────────────────────────

def _format_search_results(response: dict) -> str:
    """
    Formats Tavily's raw response into a clean string for the LLM.

    Pure function — takes a dict, returns a string. Easy to unit test.

    Args:
        response: Raw Tavily API response dict.

    Returns:
        Formatted multi-line string the LLM can read.
    """
    lines: list[str] = []

    # Tavily's own summary (if available)
    if answer := response.get("answer"):
        lines.append(f"Summary: {answer}\n")

    # Individual results
    results = response.get("results", [])
    for i, result in enumerate(results, 1):
        title   = result.get("title", "No title")
        url     = result.get("url", "")
        content = result.get("content", "")[:500]  # truncate to 500 chars
        lines.append(f"[{i}] {title}")
        lines.append(f"    URL: {url}")
        lines.append(f"    {content}\n")

    return "\n".join(lines) if lines else "No results found."
