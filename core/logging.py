"""
Module: core.logging
Layer:  Infrastructure
Owner:  core/

Purpose:
    Configures structured logging for the entire application.
    Called exactly once in main.py before any other imports execute.
    All other modules use logging.getLogger(__name__) — never print().

    Why structured logging over print():
      - Respects log levels (DEBUG, INFO, WARNING, ERROR) — filter noise in prod
      - Captured by log aggregation tools (Datadog, CloudWatch, etc.)
      - Lazy evaluation: logger.info("val=%s", x) only formats if INFO is enabled
      - Named loggers: logger.getLogger(__name__) lets you silence noisy libraries

Public API:
    configure_logging()  — call once at app startup

LangGraph Concepts demonstrated:
    - Foundation for observability hooks (Concept 19)
"""
from __future__ import annotations

import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    """
    Configures structured logging for the entire application.

    Call this once in main.py before importing anything else.
    Uses stdout (not stderr) so logs appear in order with Rich output.

    Args:
        level: Log level string. One of: DEBUG, INFO, WARNING, ERROR, CRITICAL.
               Override with LOG_LEVEL env var or pass explicitly.

    Example:
        # main.py — first lines of the file
        from core.logging import configure_logging
        configure_logging()  # must be before any other imports
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        # Format: timestamp · level · module name · message
        # Example: 2026-06-17 10:23:45,123 INFO agents.nodes node=classify thread=abc-123
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,  # override any handlers set by imported libraries
    )

    # ── Suppress noisy third-party loggers ───────────────────────────────────
    # These libraries log at DEBUG/INFO constantly — silence them in all envs.
    _noisy = [
        "httpx",          # HTTP client used by Groq/HF SDKs
        "httpcore",       # HTTP core used by httpx
        "faiss",          # FAISS vector store
        "sentence_transformers",  # HF sentence-transformers
        "transformers",   # HuggingFace transformers
        "urllib3",        # HTTP pool
        "langchain",      # LangChain base (too verbose at DEBUG)
    ]
    for name in _noisy:
        logging.getLogger(name).setLevel(logging.WARNING)
