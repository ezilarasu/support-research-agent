"""
RAG pipeline for support-research-agent.

Public API:
    get_rag_graph()        — returns the compiled CRAG subgraph
    get_retriever_node()   — returns the standalone retriever node function
    build_index(docs)      — ingests documents and builds the FAISS index
"""
from rag.graph import get_rag_graph, get_retriever_node
from rag.ingest import build_index

__all__ = ["get_rag_graph", "get_retriever_node", "build_index"]
