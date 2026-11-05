"""
Module: rag.graph
Layer:  Graph
Owner:  rag/

Purpose:
    CRAG (Corrective RAG) subgraph — Concept 95.

    CRAG adds a self-correction loop around standard RAG:
      1. retrieve        — FAISS similarity search + cross-encoder reranking
      2. grade_retrieval — LLM grades each doc as relevant/irrelevant
      3. Route:
           "relevant"   → generate  (use KB docs to answer)
           "partial"    → generate  (use what we have)
           "irrelevant" → web_fallback → generate  (fall back to web search)
      4. generate        — LLM synthesizes answer from retrieved context
      5. hallucination   — checks if answer is grounded in retrieved docs
      6. Route:
           grounded      → END  (return answer)
           hallucinated  → END  (return answer with warning)

    Why a subgraph (not inline nodes)?
      CRAG is a self-contained retrieval pipeline. Packaging it as a subgraph:
        - Encapsulates all retrieval logic in one place
        - Can be swapped for Self-RAG or Adaptive RAG by changing one import
        - Reusable: kb_worker_node calls get_rag_graph() as a black box
        - Testable: run the subgraph in isolation with a test FAISS index

    Subgraph boundary (Concept 73):
      InputState:  requires "query" field
      OutputState: provides "retrieved_docs", "final_response",
                   "retrieval_grade", "hallucination_check"
      The parent graph maps these at the boundary.

Public API:
    get_rag_graph()        — returns the compiled CRAG subgraph
    get_retriever_node()   — returns the standalone retriever node function

Dependencies:
    core.graph_builder    — fluent GraphBuilder
    core.factories        — get_vector_store(), get_checkpointer()
    rag.retriever         — make_retriever_node()
    rag.grader            — grade_retrieval, check_hallucination (pure fns)
    llm_provider          — get_router_llm(), get_research_llm()
    state                 — AgentState

LangGraph Concepts demonstrated:
    - RAG Subgraphs (Concept 89)      — CRAG as a compiled subgraph
    - Query Rewriting (Concept 91)    — rewrite_query_node
    - Retrieval Graders (Concept 93)  — retrieval_grade_node
    - Hallucination Graders (Concept 94) — hallucination_check_node
    - CRAG (Concept 95)               — full corrective loop
"""
from __future__ import annotations

import logging
from functools import lru_cache

from langchain_core.runnables import RunnableConfig  # noqa: TC002

from llm_provider import get_research_llm, get_router_llm, with_retry
from state import AgentState

logger = logging.getLogger(__name__)


# ── Query rewriting node ─────────────────────────────────────────────────────

@with_retry
async def rewrite_query_node(state: AgentState, config: RunnableConfig) -> dict:
    """
    Rewrites the query to improve retrieval when initial results are poor.

    CRAG calls this when retrieval_grade == "irrelevant". The rewriter
    rephrases the query to be more specific and KB-friendly.

    Reads:  state["query"], state["messages"]
    Writes: state["query"] (rewritten version)
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    original_query = state.get("query") or ""
    thread_id = config.get("configurable", {}).get("thread_id", "unknown")

    logger.info("rewrite_query_node: thread=%s original=%r", thread_id, original_query[:60])

    llm = get_router_llm()
    prompt = [
        SystemMessage(content=(
            "You are a query optimizer for a support knowledge base. "
            "Rewrite the user's question to be more specific and keyword-rich "
            "for better document retrieval. Return ONLY the rewritten query, nothing else."
        )),
        HumanMessage(content=f"Original question: {original_query}"),
    ]

    try:
        response = await llm.ainvoke(prompt, config=config)
        rewritten = response.content.strip()
        logger.info(
            "rewrite_query_node: thread=%s rewritten=%r",
            thread_id,
            rewritten[:60],
        )
        return {"query": rewritten}
    except Exception as e:
        logger.warning("rewrite_query_node: rewrite failed, keeping original. error=%s", e)
        return {"query": original_query}


# ── Retrieval grading node ────────────────────────────────────────────────────

async def retrieval_grade_node(state: AgentState, config: RunnableConfig) -> dict:
    """
    Grades retrieved documents and sets retrieval_grade.

    Pure grading logic lives in rag.grader.grade_retrieval().
    This node bridges state → grader → state.

    Reads:  state["query"], state["retrieved_docs"]
    Writes: state["retrieval_grade"]
    """
    from rag.grader import grade_retrieval

    query = state.get("query") or ""
    docs = state.get("retrieved_docs", [])
    thread_id = config.get("configurable", {}).get("thread_id", "unknown")

    llm = get_router_llm()
    grade = grade_retrieval(query, docs, llm)

    logger.info(
        "retrieval_grade_node: thread=%s grade=%s doc_count=%d",
        thread_id,
        grade,
        len(docs),
    )
    return {"retrieval_grade": grade}


# ── Generate answer node ──────────────────────────────────────────────────────

@with_retry
async def generate_node(state: AgentState, config: RunnableConfig) -> dict:
    """
    Generates the final answer using retrieved context.

    Synthesizes KB documents (and optionally web results) into a coherent
    support response. Uses get_research_llm() — the most capable model
    for reasoning and synthesis.

    Reads:  state["query"], state["retrieved_docs"], state["messages"]
    Writes: state["final_response"]
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    query = state.get("query") or ""
    docs = state.get("retrieved_docs", [])
    thread_id = config.get("configurable", {}).get("thread_id", "unknown")

    logger.info(
        "generate_node: thread=%s query=%r doc_count=%d",
        thread_id,
        query[:60],
        len(docs),
    )

    # Build context from retrieved documents
    if docs:
        context_parts = []
        for i, doc in enumerate(docs[:5], 1):  # cap at 5 docs
            context_parts.append(
                f"[Source {i}: {doc.get('source', 'unknown')}]\n{doc['content'][:600]}"
            )
        context = "\n\n".join(context_parts)
    else:
        context = "No relevant documents found in the knowledge base."

    llm = get_research_llm()
    prompt = [
        SystemMessage(content=(
            "You are a helpful customer support assistant. "
            "Answer the user's question using the provided context. "
            "If the context doesn't fully answer the question, say so clearly. "
            "Be concise, accurate, and friendly."
        )),
        HumanMessage(content=(
            f"Context from knowledge base:\n{context}\n\n"
            f"User question: {query}"
        )),
    ]

    try:
        response = await llm.ainvoke(prompt, config=config)
        answer = response.content.strip()
        logger.info(
            "generate_node: thread=%s answer_len=%d",
            thread_id,
            len(answer),
        )
        return {"final_response": answer}
    except Exception as e:
        logger.error("generate_node: failed thread=%s error=%s", thread_id, e)
        return {
            "final_response": (
                "I'm sorry, I encountered an issue generating a response. "
                "Please try again or contact support directly."
            ),
            "error": str(e),
        }


# ── Hallucination check node ──────────────────────────────────────────────────

async def hallucination_check_node(state: AgentState, config: RunnableConfig) -> dict:
    """
    Checks whether the generated answer is grounded in retrieved docs.

    Pure check logic lives in rag.grader.check_hallucination().
    This node bridges state → grader → state.

    Reads:  state["final_response"], state["retrieved_docs"]
    Writes: state["hallucination_check"]
    """
    from rag.grader import check_hallucination

    answer = state.get("final_response") or ""
    docs = state.get("retrieved_docs", [])
    thread_id = config.get("configurable", {}).get("thread_id", "unknown")

    if not answer:
        return {"hallucination_check": True}  # no answer = nothing to hallucinate

    llm = get_router_llm()
    is_grounded = check_hallucination(answer, docs, llm)

    logger.info(
        "hallucination_check_node: thread=%s grounded=%s",
        thread_id,
        is_grounded,
    )
    return {"hallucination_check": is_grounded}


# ── CRAG routing functions ────────────────────────────────────────────────────

def route_by_grade(state: AgentState) -> str:
    """Routes after retrieval grading: relevant/partial → generate, irrelevant → rewrite."""
    grade = state.get("retrieval_grade", "irrelevant")
    if grade in ("relevant", "partial"):
        return "generate"
    return "rewrite_query"


def route_after_generate(state: AgentState) -> str:
    """Routes after generation: always end — we do one hallucination check, not a loop."""
    return "hallucination_check"


# ── CRAG subgraph factory ─────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_rag_graph():
    """
    Build and compile the CRAG subgraph.

    Cached with lru_cache — the same compiled graph is reused across
    all invocations (building a StateGraph is expensive).

    Why no checkpointer on the subgraph?
      The parent graph owns the checkpointer. Subgraphs share the parent's
      checkpoint context — they don't need (and shouldn't have) their own.

    Returns:
        Compiled LangGraph StateGraph for CRAG retrieval.
    """
    from core.graph_builder import GraphBuilder
    from rag.retriever import make_retriever_node

    retriever = make_retriever_node()

    graph = (
        GraphBuilder(AgentState)
        .node("retrieve", retriever)
        .node("grade_retrieval", retrieval_grade_node)
        .node("rewrite_query", rewrite_query_node)
        .node("generate", generate_node)
        .node("hallucination_check", hallucination_check_node)
        .edge("retrieve", "grade_retrieval")
        .conditional_edge(
            "grade_retrieval",
            route_by_grade,
            {"generate": "generate", "rewrite_query": "rewrite_query"},
        )
        .edge("rewrite_query", "retrieve")          # CRAG loop: rewrite → re-retrieve
        .edge("generate", "hallucination_check")
        .entry("retrieve")
        .build()
    )

    logger.info("get_rag_graph: CRAG subgraph compiled")
    return graph


def get_retriever_node():
    """
    Returns the standalone retriever node function.

    Used by kb_worker_node when it needs retrieval without the full CRAG loop.

    Returns:
        Async node function from make_retriever_node().
    """
    from rag.retriever import make_retriever_node
    return make_retriever_node()
