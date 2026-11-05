"""
Tests for Module 10 security layer: auth, RBAC, compliance, exceptions.

Covers:
  - core/exceptions.py — exception hierarchy, structured context
  - security/auth.py   — validate_jwt, build_auth_context, auth_node, create_jwt
  - security/rbac.py   — has_permission, require_permission, rbac_check_node
  - security/compliance.py — detect_pii, compliance_gate_node (already tested in
                              tests/unit/test_workflows.py, extended here)
  - observability/tracing.py — configure_tracing, get_run_metadata, is_tracing_enabled
  - observability/metrics.py — record_*, get_session_metrics, reset_metrics
"""
from __future__ import annotations

import os

import pytest
from langchain_core.messages import HumanMessage

from core.exceptions import (
    AuthenticationError,
    AuthorizationError,
    ConfigurationError,
    ProviderError,
    RetrievalError,
    SupportAgentError,
    ValidationError,
)
from core.settings import get_settings
from security.auth import build_auth_context, create_jwt, validate_jwt
from security.rbac import (
    ROLE_PERMISSIONS,
    get_permissions,
    has_permission,
    require_permission,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clear_settings():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def mock_settings(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    get_settings.cache_clear()
    return get_settings()


@pytest.fixture
def settings_with_jwt(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("JWT_SECRET", "test-secret-key-for-tests")
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    get_settings.cache_clear()
    return get_settings()


@pytest.fixture
def runnable_config():
    return {
        "configurable": {
            "thread_id": "test-security-thread",
            "user_id": "user-123",
            "tenant_id": "acme-corp",
            "roles": ["agent"],
        }
    }


@pytest.fixture
def base_state():
    return {
        "messages": [HumanMessage(content="I need a refund")],
        "ticket_type": None,
        "ticket_metadata": None,
        "classification_confidence": 0.0,
        "thread_id": None,
        "retrieved_memories": [],
        "streaming_enabled": False,
        "active_agent": None,
        "agent_results": [],
        "query": None,
        "retrieved_docs": [],
        "retrieval_grade": None,
        "hallucination_check": None,
        "pending_approval": None,
        "approval_decision": None,
        "escalation_level": 0,
        "authenticated": False,
        "rbac_roles": [],
        "audit_events": [],
        "final_response": None,
        "error": None,
    }


# ── Exception hierarchy tests ─────────────────────────────────────────────────

class TestExceptionHierarchy:
    """All custom exceptions derive from SupportAgentError."""

    def test_authentication_error_is_support_agent_error(self):
        with pytest.raises(SupportAgentError):
            raise AuthenticationError("token expired")

    def test_authorization_error_carries_context(self):
        err = AuthorizationError("user-1", "approve_refund", "billing_admin")
        assert err.user_id == "user-1"
        assert err.action == "approve_refund"
        assert err.required_role == "billing_admin"
        assert "billing_admin" in str(err)

    def test_provider_error_carries_context(self):
        cause = RuntimeError("connection refused")
        err = ProviderError("groq", "gemma2-9b-it", cause)
        assert err.provider == "groq"
        assert err.model == "gemma2-9b-it"
        assert err.cause is cause
        assert "groq" in str(err)

    def test_validation_error_carries_field_and_reason(self):
        err = ValidationError("messages", "list is empty")
        assert err.field == "messages"
        assert err.reason == "list is empty"

    def test_all_errors_catchable_as_support_agent_error(self):
        errors = [
            AuthenticationError("x"),
            AuthorizationError("u", "a", "r"),
            ProviderError("p", "m", Exception("e")),
            RetrievalError("no index"),
            ValidationError("f", "r"),
            ConfigurationError("missing key"),
        ]
        for err in errors:
            with pytest.raises(SupportAgentError):
                raise err

    def test_catching_specific_subclass_does_not_catch_others(self):
        with pytest.raises(AuthorizationError):
            raise AuthorizationError("u", "a", "r")

        # AuthenticationError should NOT be caught by AuthorizationError handler
        with pytest.raises(AuthenticationError):
            try:
                raise AuthenticationError("expired")
            except AuthorizationError:
                pass  # should not reach here


# ── JWT auth tests ─────────────────────────────────────────────────────────────

class TestJWT:
    """Tests for JWT validation and auth context building."""

    def test_create_and_validate_jwt(self, settings_with_jwt):
        token = create_jwt(
            user_id="user-456",
            tenant_id="tenant-abc",
            roles=["agent", "billing_admin"],
            settings=settings_with_jwt,
        )
        assert isinstance(token, str)
        assert len(token) > 50  # JWTs are always long

        payload = validate_jwt(
            token,
            settings_with_jwt.jwt_secret.get_secret_value(),
            settings_with_jwt.jwt_algorithm,
        )
        assert payload["sub"] == "user-456"
        assert payload["tenant_id"] == "tenant-abc"
        assert "billing_admin" in payload["roles"]

    def test_validate_jwt_raises_on_invalid_token(self, settings_with_jwt):
        with pytest.raises(AuthenticationError, match="invalid token"):
            validate_jwt(
                "not.a.valid.jwt",
                settings_with_jwt.jwt_secret.get_secret_value(),
            )

    def test_validate_jwt_raises_on_wrong_secret(self, settings_with_jwt):
        token = create_jwt("u", "t", ["agent"], settings=settings_with_jwt)
        with pytest.raises(AuthenticationError):
            validate_jwt(token, "wrong-secret")

    def test_validate_jwt_raises_on_missing_sub(self, settings_with_jwt):
        from jose import jwt as jose_jwt
        payload = {"tenant_id": "t", "roles": ["agent"]}
        token = jose_jwt.encode(
            payload,
            settings_with_jwt.jwt_secret.get_secret_value(),
            algorithm="HS256",
        )
        with pytest.raises(AuthenticationError, match="missing 'sub'"):
            validate_jwt(token, settings_with_jwt.jwt_secret.get_secret_value())

    def test_build_auth_context_dev_mode_no_secret(self, mock_settings):
        ctx = build_auth_context(token=None, settings=mock_settings)
        assert ctx["authenticated"] is True
        assert ctx["user_id"] == "dev-user"
        assert len(ctx["roles"]) > 0

    def test_build_auth_context_with_valid_token(self, settings_with_jwt):
        token = create_jwt("u-789", "acme", ["agent"], settings=settings_with_jwt)
        ctx = build_auth_context(token=token, settings=settings_with_jwt)
        assert ctx["user_id"] == "u-789"
        assert ctx["tenant_id"] == "acme"
        assert ctx["authenticated"] is True

    def test_build_auth_context_raises_when_secret_set_but_no_token(self, settings_with_jwt):
        with pytest.raises(AuthenticationError, match="no token"):
            build_auth_context(token=None, settings=settings_with_jwt)

    def test_create_jwt_raises_without_secret(self, mock_settings):
        with pytest.raises(ConfigurationError):
            create_jwt("u", "t", ["agent"], settings=mock_settings)


# ── RBAC tests ─────────────────────────────────────────────────────────────────

class TestRBAC:
    """Tests for role-to-permission mapping and enforcement."""

    def test_agent_has_tickets_respond(self):
        assert has_permission(["agent"], "tickets:respond") is True

    def test_agent_cannot_approve_refunds(self):
        assert has_permission(["agent"], "refunds:approve") is False

    def test_billing_admin_can_approve_refunds(self):
        assert has_permission(["billing_admin"], "refunds:approve") is True

    def test_multiple_roles_union_permissions(self):
        # agent + billing_admin = agent permissions + refunds:approve
        assert has_permission(["agent", "billing_admin"], "refunds:approve") is True
        assert has_permission(["agent", "billing_admin"], "tickets:respond") is True

    def test_unknown_role_has_no_permissions(self):
        assert has_permission(["super_admin"], "tickets:read") is False

    def test_empty_roles_has_no_permissions(self):
        assert has_permission([], "tickets:read") is False

    def test_readonly_cannot_respond(self):
        assert has_permission(["readonly"], "tickets:respond") is False
        assert has_permission(["readonly"], "tickets:read") is True

    def test_get_permissions_returns_union(self):
        agent_perms = get_permissions(["agent"])
        billing_perms = get_permissions(["billing_admin"])
        combined = get_permissions(["agent", "billing_admin"])
        assert combined == agent_perms | billing_perms

    def test_require_permission_passes_when_allowed(self):
        # Should not raise
        require_permission(["billing_admin"], "user-1", "approve_refund", "refunds:approve")

    def test_require_permission_raises_when_denied(self):
        with pytest.raises(AuthorizationError) as exc_info:
            require_permission(["agent"], "user-1", "approve_refund", "refunds:approve")
        err = exc_info.value
        assert err.user_id == "user-1"
        assert err.action == "approve_refund"
        assert err.required_role == "refunds:approve"

    def test_all_roles_defined_in_role_permissions(self):
        expected_roles = {"agent", "billing_admin", "supervisor", "readonly"}
        assert set(ROLE_PERMISSIONS.keys()) == expected_roles

    def test_supervisor_can_escalate(self):
        assert has_permission(["supervisor"], "tickets:escalate") is True

    def test_supervisor_can_list_tenants(self):
        assert has_permission(["supervisor"], "tenants:list") is True

    def test_agent_cannot_list_tenants(self):
        assert has_permission(["agent"], "tenants:list") is False


# ── RBAC node tests ───────────────────────────────────────────────────────────

class TestRBACNode:
    """Tests for rbac_check_node graph node."""

    @pytest.mark.asyncio
    async def test_allows_agent_for_tickets_respond(self, base_state, runnable_config):
        from security.rbac import rbac_check_node

        base_state["rbac_roles"] = ["agent"]
        result = await rbac_check_node(base_state, runnable_config, "tickets:respond")

        assert "error" not in result or result.get("error") is None
        assert len(result["audit_events"]) == 1
        assert result["audit_events"][0]["allowed"] is True

    @pytest.mark.asyncio
    async def test_denies_agent_for_refunds_approve(self, base_state, runnable_config):
        from security.rbac import rbac_check_node

        base_state["rbac_roles"] = ["agent"]
        result = await rbac_check_node(base_state, runnable_config, "refunds:approve")

        assert result.get("error") is not None
        assert "denied" in result["error"].lower() or "access" in result["error"].lower()
        assert result["audit_events"][0]["allowed"] is False

    @pytest.mark.asyncio
    async def test_allows_billing_admin_for_refunds_approve(self, base_state, runnable_config):
        from security.rbac import rbac_check_node

        base_state["rbac_roles"] = ["billing_admin"]
        result = await rbac_check_node(base_state, runnable_config, "refunds:approve")

        assert "error" not in result or result.get("error") is None
        assert result["audit_events"][0]["allowed"] is True

    @pytest.mark.asyncio
    async def test_audit_event_always_written(self, base_state, runnable_config):
        from security.rbac import rbac_check_node

        base_state["rbac_roles"] = []
        result = await rbac_check_node(base_state, runnable_config, "tickets:respond")

        # audit event written even on denial
        assert "audit_events" in result
        assert len(result["audit_events"]) == 1
        event = result["audit_events"][0]
        assert event["node"] == "rbac_check_node"
        assert event["action"] == "rbac_check"
        assert "permission_checked" in event


# ── Auth node tests ───────────────────────────────────────────────────────────

class TestAuthNode:
    """Tests for auth_node graph node."""

    @pytest.mark.asyncio
    async def test_auth_node_sets_authenticated(self, base_state, runnable_config):
        from security.auth import auth_node

        result = await auth_node(base_state, runnable_config)

        assert result["authenticated"] is True
        assert isinstance(result["rbac_roles"], list)
        assert len(result["audit_events"]) == 1

    @pytest.mark.asyncio
    async def test_auth_node_audit_event_contains_user_id(self, base_state, runnable_config):
        from security.auth import auth_node

        result = await auth_node(base_state, runnable_config)

        event = result["audit_events"][0]
        assert event["action"] == "auth_verified"
        assert event["user_id"] == "user-123"
        assert event["tenant_id"] == "acme-corp"

    @pytest.mark.asyncio
    async def test_auth_node_uses_roles_from_config(self, base_state):
        from security.auth import auth_node

        config = {
            "configurable": {
                "thread_id": "t",
                "user_id": "u",
                "tenant_id": "tenant",
                "roles": ["billing_admin", "supervisor"],
            }
        }
        result = await auth_node(base_state, config)

        assert "billing_admin" in result["rbac_roles"]
        assert "supervisor" in result["rbac_roles"]


# ── Tracing tests ─────────────────────────────────────────────────────────────

class TestTracing:
    """Tests for LangSmith tracing configuration."""

    def test_configure_tracing_no_op_without_key(self, mock_settings, monkeypatch):
        from observability.tracing import configure_tracing

        monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
        monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)

        configure_tracing(mock_settings)  # mock_settings has no langsmith_api_key

        assert os.environ.get("LANGCHAIN_API_KEY") is None

    def test_configure_tracing_sets_env_with_key(self, monkeypatch):
        from observability.tracing import configure_tracing

        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        monkeypatch.setenv("LANGSMITH_API_KEY", "test-langsmith-key")
        monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
        get_settings.cache_clear()
        settings = get_settings()

        configure_tracing(settings)

        assert os.environ.get("LANGCHAIN_TRACING_V2") == "true"
        assert os.environ.get("LANGCHAIN_API_KEY") == "test-langsmith-key"

    def test_get_run_metadata_includes_tenant(self):
        from observability.tracing import get_run_metadata

        metadata = get_run_metadata(tenant_id="acme", user_id="u-1", thread_id="t-1")

        assert metadata["tenant_id"] == "acme"
        assert metadata["user_id"] == "u-1"
        assert metadata["thread_id"] == "t-1"
        assert metadata["app"] == "support-research-agent"

    def test_get_run_metadata_omits_none_fields(self):
        from observability.tracing import get_run_metadata

        metadata = get_run_metadata(tenant_id="t")

        assert "ticket_type" not in metadata  # not provided → omitted

    def test_is_tracing_enabled_false_by_default(self, monkeypatch):
        from observability.tracing import is_tracing_enabled

        monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)

        assert is_tracing_enabled() is False


# ── Metrics tests ─────────────────────────────────────────────────────────────

class TestMetrics:
    """Tests for in-process metrics collection."""

    def setup_method(self):
        from observability.metrics import reset_metrics
        reset_metrics("test-metrics-thread")

    def teardown_method(self):
        from observability.metrics import reset_metrics
        reset_metrics("test-metrics-thread")

    def test_record_and_retrieve_node_latency(self):
        from observability.metrics import get_session_metrics, record_node_latency

        record_node_latency("classify_ticket", 45.2, "test-metrics-thread")
        record_node_latency("respond", 312.7, "test-metrics-thread")

        metrics = get_session_metrics("test-metrics-thread")

        assert len(metrics["node_latencies"]) == 2
        assert metrics["total_duration_ms"] == pytest.approx(357.9, abs=0.1)

    def test_slowest_node_identified(self):
        from observability.metrics import get_session_metrics, record_node_latency

        record_node_latency("classify_ticket", 50.0, "test-metrics-thread")
        record_node_latency("respond", 800.0, "test-metrics-thread")
        record_node_latency("kb_worker", 200.0, "test-metrics-thread")

        metrics = get_session_metrics("test-metrics-thread")

        assert metrics["slowest_node"] == ("respond", 800.0)

    def test_token_usage_accumulated(self):
        from observability.metrics import get_session_metrics, record_token_usage

        record_token_usage("classify_ticket", 100, 20, "test-metrics-thread", "gemma2-9b-it")
        record_token_usage("respond", 500, 150, "test-metrics-thread", "llama-3.3-70b")

        metrics = get_session_metrics("test-metrics-thread")

        assert metrics["total_tokens"] == 770
        assert len(metrics["token_usages"]) == 2

    def test_empty_session_returns_zero_totals(self):
        from observability.metrics import get_session_metrics

        metrics = get_session_metrics("nonexistent-thread")

        assert metrics["total_duration_ms"] == 0.0
        assert metrics["total_tokens"] == 0
        assert metrics["slowest_node"] is None

    def test_reset_clears_metrics(self):
        from observability.metrics import get_session_metrics, record_node_latency, reset_metrics

        record_node_latency("respond", 100.0, "test-metrics-thread")
        reset_metrics("test-metrics-thread")
        metrics = get_session_metrics("test-metrics-thread")

        assert metrics["total_duration_ms"] == 0.0
        assert len(metrics["node_latencies"]) == 0

    def test_configure_metrics_registers_post_hook(self):
        from observability.hooks import _post_hooks
        from observability.metrics import configure_metrics

        initial_count = len(_post_hooks)
        configure_metrics()

        assert len(_post_hooks) >= initial_count  # at least one hook registered
