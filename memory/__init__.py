"""
Module: memory
Layer:  Service
Owner:  memory/

Purpose:
    Long-term semantic memory system for support-research-agent.
    Stores user preferences, account facts, and past-issue summaries
    ACROSS conversation threads — not just within one session.

    Contrast with checkpointing (Module 4):
      Checkpointing = short-term, per-thread, survives restarts
      This package = long-term, per-user, cross-thread, meaning-based search

Public API:
    LongTermMemoryStore    — the store class (inject in tests via constructor)
    get_memory_store()     — factory: returns the singleton store
    MessageStoreProtocol   — interface for short-term message stores
    SemanticSearchProtocol — interface for semantic memory stores
"""
from memory.protocols import MessageStoreProtocol, SemanticSearchProtocol
from memory.store import LongTermMemoryStore, get_memory_store

__all__ = [
    "LongTermMemoryStore",
    "get_memory_store",
    "MessageStoreProtocol",
    "SemanticSearchProtocol",
]
