"""
Module: rag.evaluator
Layer:  Service
Owner:  rag/

Purpose:
    RAG evaluation metrics — pure functions for offline quality measurement.
    Not part of the live request path; used for testing and tuning.

    Three standard RAG evaluation dimensions:
      1. Retrieval Recall    — did we retrieve the relevant documents?
      2. Answer Faithfulness — is the answer grounded in retrieved docs?
      3. Answer Relevance    — does the answer actually address the question?

    Why evaluate offline (not in the graph)?
      Evaluating during live requests adds latency and cost for every query.
      Run evaluations periodically on a fixed test set to detect regressions
      after KB updates, model changes, or prompt edits.

Public API:
    evaluate_retrieval_recall(query, retrieved, relevant_sources)
    evaluate_faithfulness(answer, docs, llm)
    evaluate_answer_relevance(question, answer, llm)
    run_eval_suite(test_cases, llm)

Dependencies:
    rag.grader    — check_hallucination (faithfulness proxy)
    rag.protocols — RetrievedDocument

LangGraph Concepts demonstrated:
    - RAG Evaluation (Concept 106) — three grader nodes as pure functions
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rag.protocols import RetrievedDocument

logger = logging.getLogger(__name__)


def evaluate_retrieval_recall(
    query: str,
    retrieved: list[RetrievedDocument],
    relevant_sources: list[str],
) -> float:
    """
    Recall@k: fraction of known-relevant sources that were retrieved.

    Args:
        query:            The search query (used for logging).
        retrieved:        Documents returned by the retriever.
        relevant_sources: Known-relevant source paths for this query.

    Returns:
        Float in [0.0, 1.0]. 1.0 means all relevant sources were retrieved.
    """
    if not relevant_sources:
        logger.warning("evaluate_retrieval_recall: no relevant_sources provided — cannot compute")
        return 0.0

    retrieved_sources = {doc["source"] for doc in retrieved}
    hits = sum(1 for src in relevant_sources if src in retrieved_sources)
    recall = hits / len(relevant_sources)

    logger.debug(
        "eval_recall: query=%r hits=%d/%d recall=%.2f",
        query[:40],
        hits,
        len(relevant_sources),
        recall,
    )
    return recall


def evaluate_faithfulness(
    answer: str,
    docs: list[RetrievedDocument],
    llm: Any,
) -> float:
    """
    Faithfulness: 1.0 if the answer is grounded in retrieved docs, else 0.0.

    Uses check_hallucination as the underlying grader.

    Args:
        answer: LLM-generated response.
        docs:   Retrieved context documents.
        llm:    Router LLM for grading.

    Returns:
        1.0 (faithful) or 0.0 (hallucinated).
    """
    from rag.grader import check_hallucination

    is_grounded = check_hallucination(answer, docs, llm)
    score = 1.0 if is_grounded else 0.0

    logger.debug("eval_faithfulness: grounded=%s score=%.1f", is_grounded, score)
    return score


def evaluate_answer_relevance(
    question: str,
    answer: str,
    llm: Any,
) -> float:
    """
    Answer relevance: does the answer actually address the question?

    Prompts the LLM to score relevance 1–5, then normalises to [0.0, 1.0].

    Args:
        question: The original user question.
        answer:   The generated answer.
        llm:      Router LLM for evaluation.

    Returns:
        Float in [0.0, 1.0]. 1.0 means perfectly relevant.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    prompt = [
        SystemMessage(content=(
            "Rate how well the answer addresses the question on a scale of 1 to 5.\n"
            "1 = completely off-topic\n"
            "5 = directly and completely answers the question\n"
            "Respond with ONLY a single integer: 1, 2, 3, 4, or 5."
        )),
        HumanMessage(content=f"Question: {question}\n\nAnswer: {answer}"),
    ]

    try:
        response = llm.invoke(prompt)
        raw = response.content.strip()
        score_int = int(raw[0]) if raw and raw[0].isdigit() else 3
        score_int = max(1, min(5, score_int))  # clamp to [1, 5]
        score = (score_int - 1) / 4.0         # normalise to [0.0, 1.0]

        logger.debug(
            "eval_relevance: question=%r raw_score=%d normalised=%.2f",
            question[:40],
            score_int,
            score,
        )
        return score

    except Exception as e:
        logger.warning("eval_relevance: scoring failed, defaulting to 0.5. error=%s", e)
        return 0.5


def run_eval_suite(
    test_cases: list[dict],
    llm: Any,
) -> dict:
    """
    Run a full evaluation suite over a list of test cases.

    Each test case is a dict:
        {
            "query": str,
            "expected_sources": list[str],   # known-relevant doc sources
            "retrieved": list[RetrievedDocument],
            "answer": str,
        }

    Args:
        test_cases: List of test case dicts.
        llm:        LLM for faithfulness and relevance grading.

    Returns:
        Summary dict:
            {
                "recall_mean": float,
                "faithfulness_mean": float,
                "relevance_mean": float,
                "n": int,
                "by_case": list[dict],
            }
    """
    if not test_cases:
        return {"recall_mean": 0.0, "faithfulness_mean": 0.0, "relevance_mean": 0.0, "n": 0, "by_case": []}

    results = []
    for case in test_cases:
        query = case.get("query", "")
        expected = case.get("expected_sources", [])
        retrieved = case.get("retrieved", [])
        answer = case.get("answer", "")

        recall = evaluate_retrieval_recall(query, retrieved, expected)
        faithfulness = evaluate_faithfulness(answer, retrieved, llm)
        relevance = evaluate_answer_relevance(query, answer, llm)

        results.append({
            "query": query,
            "recall": recall,
            "faithfulness": faithfulness,
            "relevance": relevance,
        })

    n = len(results)
    return {
        "recall_mean": sum(r["recall"] for r in results) / n,
        "faithfulness_mean": sum(r["faithfulness"] for r in results) / n,
        "relevance_mean": sum(r["relevance"] for r in results) / n,
        "n": n,
        "by_case": results,
    }
