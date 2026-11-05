"""
Module: tools
Layer:  Service
Owner:  tools/

Purpose:
    Tool definitions for LangGraph ToolNode.
    Each tool is a class with a Pydantic args_schema for type-safe inputs.
    Tools are imported by agent nodes that need external capabilities.

Public API:
    SearchTool          — Tavily web search (used by research_node)
    KnowledgeBaseTool   — FAISS KB search (used by technical/research nodes)
    get_research_tools  — returns tool list for research agent
    get_technical_tools — returns tool list for technical agent

LangGraph Concepts demonstrated:
    - ToolNode (Concept 16)
"""
from tools.kb_tool import KnowledgeBaseTool
from tools.search_tool import SearchTool

__all__ = [
    "SearchTool",
    "KnowledgeBaseTool",
    "get_research_tools",
    "get_technical_tools",
]


def get_research_tools() -> list:
    """Returns the tool list for the research agent."""
    return [SearchTool(), KnowledgeBaseTool()]


def get_technical_tools() -> list:
    """Returns the tool list for the technical agent."""
    return [KnowledgeBaseTool()]
