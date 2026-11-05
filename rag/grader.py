"""
Module: rag.grader
Layer:  Service
Owner:  rag/

Purpose:
    Retrieval graders and hallucination grader for the CRAG pipeline.
    All graders are PURE FUNCTIONS — they take inputs and return typed results
    without touching state directly. This makes them trivially unit-testable.

    Two grader types:
      1. Retrieval grader — grades each retrieved document as relevant/irrelevant
         to the query. Drives CRAG's decision: use KB, rewrite query, or fall back.
      2. Hallucination grader — checks whether the LLM's answer is grounded in
         the retrieved documents. Returns True (grounded) or False (hallucinated).

    Why pure functions (not nodes)?
      Graders are called FROM nodes but are not nodes themselves. Keeping them
      as pure functions means:
        - Unit tests: no mocking of LangGraph state or RunnableConfig
        - Reuse: any node can call a grader without graph coupling
        - Composability: graders can be chained or parallelized easily

    CRAG decision logic (Concept 95):
      retrieve → grade_retrieval →
        if all irrelevant   → rewrite_query → retrieve (loop)
        if partial          → web_fallback → merge → answer
        if all relevant     → answer → hallucination_check
          if hallucinated   → rewrite_query (loop)
          if grounded       → final_response

Public API:
    grade_document(query, content, llm)           — grade one document
    grade_retrieval(query, docs, llm)             — grade all retrieved docs
    check_hallucination(answer, docs, llm)        — check if answer is grounded

Dependencies:
    core.settings     — (accessed via passed llm, no direct dependency)
    llm_provider      — llm is injected by the caller node
    rag.protocols     — RetrievedDocument, GradeResult

LangGraph Concepts demonstrated:
    - Retrieval Graders (Concept 93)      — grade_document / grade_retrieval
    - Hallucination Graders (Concept 94)  — check_hallucination
    - CRAG (Concept 95)                   — graders drive the CRAG loop
"""
from __future__ import annotations

import logging
from typing import Any

from rag.protocols import GradeResult, RetrievedDocument

logger = logging.getLogger(__name__)

# Grading scale — used in prompts and returned in grade field
_GRADE_RELEVANT = "relevant"
_GRADE_IRRELEVANT = "irrelevant"
_GRADE_PARTIAL = "partial"


def grade_document(
    query: str,
    content: str,
    llm: Any,
) -> GradeResult:
    """
    Grade whether a single document is relevant to the query.

    Pure function — no state, no I/O beyond the LLM call.
    Used by grade_retrieval to evaluate each document individually.

    Args:
        query:   The search query (may be the original ticket text or rewritten).
        content: The document chunk content to evaluate.
        llm:     LLM instance (from get_router_llm() — cheap, fast model).

    Returns:
        GradeResult with grade ("relevant" | "irrelevant"), reason, and scores.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    prompt = [
        SystemMessage(content=(
            "You are a document relevance grader. "
            "Your task: decide if a document is relevant to the user's question.\n"
            "Respond with ONLY one of: relevant / irrelevant\n"
            "Then a newline, then one sentence explaining why."
        )),
        HumanMessage(content=(
            f"Question: {query}\n\n"
            f"Document:\n{content[:1000]}"  # cap at 1000 chars to save tokens
        )),
    ]

    try:
        response = llm.invoke(prompt)
        lines = response.content.strip().split("\n", 1)
        raw_grade = lines[0].strip().lower()

        # Check "irrelevant" before "relevant" — the latter is a substring of the former
        grade = _GRADE_IRRELEVANT if "irrelevant" in raw_grade else _GRADE_RELEVANT
        reason = lines[1].strip() if len(lines) > 1 else "No reason provided."

        logger.debug(
            "grade_document: grade=%s query=%r content_len=%d",
            grade,
            query[:40],
            len(content),
        )
        return GradeResult(grade=grade, reason=reason, scores=[1.0 if grade == _GRADE_RELEVANT else 0.0])

    except Exception as e:
        logger.warning("grade_document: grading failed, defaulting to relevant. error=%s", e)
        # Fail safe: prefer showing potentially irrelevant content over missing relevant content
        return GradeResult(grade=_GRADE_RELEVANT, reason=f"Grading failed: {e}", scores=[0.5])


def grade_retrieval(
    query: str,
    docs: list[RetrievedDocument],
    llm: Any,
) -> str:
    """
    Grade a set of retrieved documents to determine CRAG routing.

    Evaluates each document and returns an overall grade:
      "relevant"   — majority of docs are relevant → use KB directly
      "irrelevant" — majority of docs are irrelevant → trigger web fallback
      "partial"    — mixed results → merge KB + web

    Args:
        query: The search query.
        docs:  Retrieved documents (RetrievedDocument TypedDicts).
        llm:   LLM instance for individual document grading.

    Returns:
        One of: "relevant", "irrelevant", "partial"
    """
    if not docs:
        logger.info("grade_retrieval: no docs to grade → irrelevant")
        return _GRADE_IRRELEVANT

    grades = [
        grade_document(query, doc["content"], llm)
        for doc in docs
    ]

    relevant_count = sum(1 for g in grades if g["grade"] == _GRADE_RELEVANT)
    total = len(grades)
    ratio = relevant_count / total

    if ratio >= 0.7:
        overall = _GRADE_RELEVANT
    elif ratio <= 0.3:
        overall = _GRADE_IRRELEVANT
    else:
        overall = _GRADE_PARTIAL

    logger.info(
        "grade_retrieval: overall=%s relevant=%d/%d ratio=%.2f",
        overall,
        relevant_count,
        total,
        ratio,
    )
    return overall


def check_hallucination(
    answer: str,
    docs: list[RetrievedDocument],
    llm: Any,
) -> bool:
    """
    Check whether the LLM's answer is grounded in the retrieved documents.

    Hallucination definition here: the answer contains claims that are NOT
    supported by the retrieved documents. This is a binary check — the answer
    is either grounded or it is not.

    Pure function — takes the answer and docs, returns bool.
    Called AFTER the LLM generates an answer, before returning to the user.

    Args:
        answer: The LLM-generated response to check.
        docs:   The documents used as context for the answer.
        llm:    LLM instance for evaluation.

    Returns:
        True if the answer is grounded (no hallucination detected).
        False if the answer contains unsupported claims.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    if not docs:
        # No retrieved docs → cannot verify → assume grounded (no claims to check)
        logger.debug("check_hallucination: no docs provided → assuming grounded")
        return True

    # Build context string from retrieved docs (cap each to save tokens)
    context = "\n\n---\n\n".join(
        f"Source: {doc['source']}\n{doc['content'][:500]}"
        for doc in docs[:5]  # use at most 5 docs for the check
    )

    prompt = [
        SystemMessage(content=(
            "You are a hallucination checker. "
            "Determine if the answer is supported by the provided context documents.\n"
            "An answer is hallucinated if it makes specific claims not found in the context.\n"
            "Respond with ONLY: grounded OR hallucinated\n"
            "Then a newline, then one sentence explaining why."
        )),
        HumanMessage(content=(
            f"Context:\n{context}\n\n"
            f"Answer:\n{answer}"
        )),
    ]

    try:
        response = llm.invoke(prompt)
        first_line = response.content.strip().split("\n", 1)[0].lower()
        is_grounded = "grounded" in first_line

        logger.info(
            "check_hallucination: grounded=%s answer_len=%d",
            is_grounded,
            len(answer),
        )
        return is_grounded

    except Exception as e:
        logger.warning("check_hallucination: check failed → assuming grounded. error=%s", e)
        return True  # fail safe: don't suppress answers due to grader failure
