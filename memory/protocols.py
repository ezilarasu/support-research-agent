"""
Module: memory.protocols
Layer:  Service
Owner:  memory/

Purpose:
    Interface definitions for the long-term memory system.
    Applying ISP (Interface Segregation Principle): two small, focused
    protocols instead of one large interface that forces all memory
    implementations to carry methods they don't use.

    Why two protocols?
      MessageStoreProtocol  — short-term: store/retrieve messages by thread
      SemanticSearchProtocol — long-term: embed and search by meaning

      Not every memory backend supports semantic search (e.g. a simple Redis
      message cache doesn't need embeddings). Segregating the interfaces means
      you can implement only what you need.

Public API:
    MessageStoreProtocol   — store and retrieve BaseMessage lists
    SemanticSearchProtocol — embed text into a vector and search by meaning

LangGraph Concepts demonstrated:
    - Long-Term Memory (Concept 35) — the contract these protocols define
    - Memory Store (Concept 36)     — SemanticSearchProtocol is implemented by store.py
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage


@runtime_checkable
class MessageStoreProtocol(Protocol):
    """
    Protocol for short-term message storage (per-thread).

    Implemented by anything that stores and retrieves sequences of
    BaseMessage objects keyed by thread_id.
    """

    def store(self, thread_id: str, message: BaseMessage) -> None:
        """
        Persists a single message for a thread.

        Args:
            thread_id: The conversation thread this message belongs to.
            message:   The message to persist.
        """
        ...

    def retrieve(self, thread_id: str, n: int) -> list[BaseMessage]:
        """
        Returns the last n messages for a thread, oldest-first.

        Args:
            thread_id: The conversation thread to retrieve from.
            n:         Maximum number of messages to return.

        Returns:
            list[BaseMessage]: Up to n messages, oldest first.
        """
        ...


@runtime_checkable
class SemanticSearchProtocol(Protocol):
    """
    Protocol for long-term semantic memory (per-user, cross-thread).

    Implemented by any store that can embed text into a vector space
    and retrieve the most semantically similar entries by query.
    """

    async def astore(
        self,
        user_id: str,
        key: str,
        text: str,
        metadata: dict | None = None,
    ) -> None:
        """
        Embeds and stores a memory for a user.

        Args:
            user_id:  The user this memory belongs to. Scopes the namespace.
            key:      A unique identifier for this memory (e.g. "preference_billing").
            text:     The raw text to embed and store.
            metadata: Optional dict of extra fields (created_at, source, etc.).
        """
        ...

    async def asearch(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
    ) -> list[dict]:
        """
        Returns the top-k memories most semantically similar to query.

        Args:
            user_id: Scopes the search to this user's memories only.
            query:   The search query (embedded and compared to stored vectors).
            limit:   Maximum number of results to return.

        Returns:
            list[dict]: Each dict has "text", "key", "score", and "metadata".
        """
        ...
