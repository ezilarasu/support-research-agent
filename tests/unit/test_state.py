"""
Unit tests for state.py — AgentState schema and TypeAliases.

These tests verify:
  - AgentState has the expected fields (schema contract)
  - TypeAlias values contain only the allowed literals
  - Sub-TypedDicts (TicketMetadata, RetrievedDocument) have correct fields
  - __all__ exports are complete

Why test a TypedDict?
  TypedDicts have zero runtime enforcement — any dict passes at runtime.
  These tests serve as the RUNTIME CONTRACT. If someone renames a field,
  adds a typo, or removes a key, these tests break loudly before anything
  else in the codebase can silently consume a wrong field name.

Run:
    .venv/bin/python -m pytest tests/unit/test_state.py -v
"""
from __future__ import annotations

import typing

import state as state_module
from state import (
    AgentName,
    AgentState,
    ApprovalDecision,
    GradeLabel,
    Priority,
    RetrievedDocument,
    TicketMetadata,
    TicketType,
)

# ── AgentState field contract ─────────────────────────────────────────────────

class TestAgentStateFields:
    """Verifies AgentState has exactly the fields we depend on."""

    def _fields(self) -> set[str]:
        return set(AgentState.__annotations__.keys())

    # Module 1 core fields
    def test_has_messages(self):
        assert "messages" in self._fields()

    def test_has_ticket_type(self):
        assert "ticket_type" in self._fields()

    def test_has_final_response(self):
        assert "final_response" in self._fields()

    def test_has_error(self):
        assert "error" in self._fields()

    # Module 2/3 routing fields
    def test_has_ticket_metadata(self):
        assert "ticket_metadata" in self._fields()

    def test_has_classification_confidence(self):
        assert "classification_confidence" in self._fields()

    # Module 4 persistence fields
    def test_has_thread_id(self):
        assert "thread_id" in self._fields()

    # Module 5 memory fields
    def test_has_retrieved_memories(self):
        assert "retrieved_memories" in self._fields()

    # Module 6 streaming fields
    def test_has_streaming_enabled(self):
        assert "streaming_enabled" in self._fields()

    # Module 7 HITL fields
    def test_has_pending_approval(self):
        assert "pending_approval" in self._fields()

    def test_has_approval_decision(self):
        assert "approval_decision" in self._fields()

    def test_has_escalation_level(self):
        assert "escalation_level" in self._fields()

    # Module 8 multi-agent fields
    def test_has_active_agent(self):
        assert "active_agent" in self._fields()

    def test_has_agent_results(self):
        assert "agent_results" in self._fields()

    # Module 9 RAG fields
    def test_has_query(self):
        assert "query" in self._fields()

    def test_has_retrieved_docs(self):
        assert "retrieved_docs" in self._fields()

    def test_has_retrieval_grade(self):
        assert "retrieval_grade" in self._fields()

    def test_has_hallucination_check(self):
        assert "hallucination_check" in self._fields()

    # Module 10 security fields
    def test_has_authenticated(self):
        assert "authenticated" in self._fields()

    def test_has_rbac_roles(self):
        assert "rbac_roles" in self._fields()

    def test_has_audit_events(self):
        assert "audit_events" in self._fields()

    def test_total_field_count(self):
        """Guard against accidental field removal — update this when adding new fields."""
        # 21 fields across all 10 modules
        assert len(self._fields()) == 21, (
            f"Expected 21 AgentState fields, got {len(self._fields())}. "
            f"Fields: {sorted(self._fields())}"
        )


# ── Sub-TypedDict field contracts ─────────────────────────────────────────────

class TestTicketMetadataFields:
    def test_required_fields(self):
        fields = set(TicketMetadata.__annotations__.keys())
        assert fields == {"ticket_id", "tenant_id", "user_id", "priority", "created_at", "sla_deadline"}


class TestRetrievedDocumentFields:
    def test_required_fields(self):
        fields = set(RetrievedDocument.__annotations__.keys())
        assert fields == {"content", "source", "score", "metadata"}


# ── TypeAlias literal values ──────────────────────────────────────────────────

class TestTypeAliases:
    """
    Verifies the allowed values of each Literal TypeAlias.
    These are the values routing functions and conditional edges depend on.
    A typo in a TypeAlias silently breaks all routing logic.
    """

    def _literal_values(self, alias) -> set:
        """Extracts the allowed values from a Literal TypeAlias."""
        args = typing.get_args(alias)
        return set(args)

    def test_ticket_type_values(self):
        assert self._literal_values(TicketType) == {"billing", "technical", "general", "research"}

    def test_agent_name_values(self):
        assert self._literal_values(AgentName) == {
            "billing", "technical", "general", "research", "supervisor"
        }

    def test_grade_label_values(self):
        assert self._literal_values(GradeLabel) == {"relevant", "irrelevant", "partial"}

    def test_approval_decision_values(self):
        assert self._literal_values(ApprovalDecision) == {"approved", "rejected", "escalated"}

    def test_priority_values(self):
        assert self._literal_values(Priority) == {"low", "medium", "high", "critical"}


# ── __all__ completeness ──────────────────────────────────────────────────────

class TestPublicAPI:
    """Verifies __all__ exports are complete and contain no typos."""

    def test_all_is_defined(self):
        assert hasattr(state_module, "__all__"), "state.py must define __all__"

    def test_agent_state_exported(self):
        assert "AgentState" in state_module.__all__

    def test_ticket_metadata_exported(self):
        assert "TicketMetadata" in state_module.__all__

    def test_retrieved_document_exported(self):
        assert "RetrievedDocument" in state_module.__all__

    def test_type_aliases_exported(self):
        for name in ("TicketType", "AgentName", "GradeLabel", "ApprovalDecision", "Priority"):
            assert name in state_module.__all__, f"{name} missing from __all__"

    def test_all_names_importable(self):
        """Every name in __all__ must actually exist in the module."""
        for name in state_module.__all__:
            assert hasattr(state_module, name), f"__all__ lists '{name}' but it doesn't exist"


# ── Runtime dict behaviour ────────────────────────────────────────────────────

class TestTypedDictRuntimeBehaviour:
    """
    TypedDict is a plain dict at runtime — these tests confirm that.
    This matters because LangGraph serialises/deserialises state as dicts.
    """

    def test_agent_state_is_dict_subclass(self):
        # TypedDict creates a class that IS a dict — no special wrapper
        assert issubclass(AgentState, dict)

    def test_can_construct_as_dict(self):
        # TypedDict instances are constructed like dicts
        from langchain_core.messages import HumanMessage
        state = AgentState(
            messages=[HumanMessage(content="test")],
            ticket_type=None,
            ticket_metadata=None,
            classification_confidence=0.0,
            thread_id="t-001",
            retrieved_memories=[],
            streaming_enabled=False,
            active_agent=None,
            agent_results=[],
            query=None,
            retrieved_docs=[],
            retrieval_grade=None,
            hallucination_check=None,
            pending_approval=None,
            approval_decision=None,
            escalation_level=0,
            authenticated=False,
            rbac_roles=[],
            audit_events=[],
            final_response=None,
            error=None,
        )
        # It's a dict — supports all dict operations
        assert isinstance(state, dict)
        assert state["ticket_type"] is None
        assert state["escalation_level"] == 0

    def test_partial_dict_is_valid_update_signal(self):
        """
        Nodes return partial dicts, not full AgentState instances.
        This test documents that partial dicts are intentional — they
        contain only the fields a node writes.
        """
        partial_update = {"ticket_type": "billing", "classification_confidence": 0.95}
        # A partial dict is just a plain dict — valid to return from a node
        assert isinstance(partial_update, dict)
        assert "messages" not in partial_update  # fields not written by this node absent
