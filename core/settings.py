"""
Module: core.settings
Layer:  Infrastructure
Owner:  core/

Purpose:
    Single source of truth for all application configuration.
    Loaded once at startup, cached as a singleton, imported everywhere.
    Never use os.getenv() in application code — always use get_settings().

    Why pydantic-settings over raw os.getenv():
      - SecretStr masks API keys in logs and repr output automatically
      - Validated on startup — bad config crashes early with a clear error
      - lru_cache makes it a singleton — no repeated .env file reads
      - Field(...) documents every setting inline — the file IS the docs
      - Easy to override in tests: get_settings.cache_clear() + monkeypatch

Public API:
    get_settings()  — returns the singleton Settings instance

LangGraph Concepts demonstrated:
    - Settings as the foundation for Compile (Concept 7) and Checkpointing (Concept 23)
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central application configuration for support-research-agent.

    All values are loaded from environment variables or a .env file.
    Access pattern everywhere in the codebase:
        from core.settings import get_settings
        s = get_settings()
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,  # LLM_PROVIDER and llm_provider both work
        extra="ignore",        # unknown env vars don't crash the app
    )

    # ── Provider ─────────────────────────────────────────────────────────────
    llm_provider: Literal["groq", "huggingface", "hf_local"] = Field(
        default="groq",
        description="Active LLM backend. Switch with LLM_PROVIDER env var.",
    )

    # ── API Keys ─────────────────────────────────────────────────────────────
    # SecretStr: the value is masked in logs (shows as '**********')
    # Field(...) means REQUIRED — missing key crashes at startup, not mid-run
    groq_api_key: SecretStr = Field(..., description="Groq API key. Required.")
    huggingfacehub_api_token: SecretStr | None = Field(
        default=None,
        description="HuggingFace Hub token. Required for huggingface provider.",
    )
    tavily_api_key: SecretStr | None = Field(
        default=None,
        description="Tavily search API key. Required for web research tool.",
    )
    langsmith_api_key: SecretStr | None = Field(
        default=None,
        description="LangSmith tracing key. Optional — tracing disabled if absent.",
    )

    # ── LangSmith ────────────────────────────────────────────────────────────
    langchain_tracing_v2: bool = Field(default=False)
    langchain_project: str = Field(default="support-research-agent")

    # ── Groq Models ──────────────────────────────────────────────────────────
    # Different models for different node roles — see CLAUDE.md for the full
    # assignment strategy. Using the right model per role saves tokens and latency.
    groq_model_router: str = Field(
        default="gemma2-9b-it",
        description="Fast model for routing and classification. 15k tok/min free.",
    )
    groq_model_research: str = Field(
        default="llama-3.3-70b-versatile",
        description="Most capable model for reasoning and synthesis. 6k tok/min free.",
    )
    groq_model_fast: str = Field(
        default="gpt-oss-20b",
        description="Fastest model for tool calls and extraction.",
    )
    groq_model_longdoc: str = Field(
        default="mixtral-8x7b-32768",
        description="32k context model for long-document summarisation.",
    )

    # ── HuggingFace Models ────────────────────────────────────────────────────
    hf_model_chat: str = Field(default="HuggingFaceH4/zephyr-7b-beta")
    hf_model_local: str = Field(default="microsoft/phi-3-mini-4k-instruct")
    hf_embeddings_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Local sentence-transformer. Used for RAG + semantic memory.",
    )
    hf_classifier_model: str = Field(
        default="facebook/bart-large-mnli",
        description="Zero-shot classifier. Used for ticket pre-classification (0 tokens).",
    )
    hf_reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder reranker. Used for RAG result reranking.",
    )

    # ── RAG ──────────────────────────────────────────────────────────────────
    rag_backend: Literal["faiss"] = Field(
        default="faiss",
        description="Active vector store backend. 'faiss' = local FAISS index.",
    )
    rag_top_k_retrieve: int = Field(
        default=20,
        description="Number of docs to retrieve from vector store (before reranking).",
    )
    rag_top_k_rerank: int = Field(
        default=5,
        description="Number of docs to pass to LLM after reranking.",
    )
    rag_score_threshold: float = Field(
        default=0.3,
        description="Minimum similarity score to include a retrieved document.",
    )
    rag_chunk_size: int = Field(default=512)
    rag_chunk_overlap: int = Field(default=50)
    rag_index_path: str = Field(default="indexes/kb.faiss")

    # ── Memory ───────────────────────────────────────────────────────────────
    memory_top_k: int = Field(
        default=5,
        description="Number of long-term memories to retrieve per turn.",
    )
    memory_summarize_after_turns: int = Field(
        default=10,
        description="Compress conversation history after this many turns.",
    )

    # ── Graph ─────────────────────────────────────────────────────────────────
    graph_recursion_limit: int = Field(
        default=50,
        description="Max graph steps before raising RecursionError. Guards against infinite loops.",
    )
    graph_debug: bool = Field(
        default=False,
        description="Enable LangGraph debug mode. Gates verbose node output.",
    )
    checkpointer_backend: Literal["memory", "sqlite"] = Field(
        default="sqlite",
        description=(
            "Checkpointer backend. "
            "'memory' = InMemorySaver (dev/test, no persistence). "
            "'sqlite' = SqliteSaver (local dev, survives restarts). "
        ),
    )
    sqlite_path: str = Field(
        default="memory.db",
        description="Path to SQLite file for graph state checkpointing.",
    )

    # ── Security ──────────────────────────────────────────────────────────────
    jwt_secret: SecretStr | None = Field(
        default=None,
        description="Secret for JWT signing. Required in Module 10.",
    )
    jwt_algorithm: str = Field(default="HS256")
    default_tenant_id: str = Field(
        default="default",
        description="Tenant ID used when no auth context is present (dev mode).",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Returns the singleton Settings instance.

    Cached with lru_cache — the .env file is read exactly once per process.
    Every module calls this; they all get the same object.

    In tests, clear the cache before patching env vars:
        get_settings.cache_clear()
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        s = get_settings()  # fresh instance with test values

    Returns:
        Settings: The validated, singleton settings object.

    Raises:
        ValidationError: If required fields (e.g. GROQ_API_KEY) are missing.
    """
    return Settings()
