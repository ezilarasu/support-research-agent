# LangGraph Notes — support-research-agent
*Personal reference generated after each concept. Append only.*

---

## Progress Tracker

| Module | # | Concept | Status | Date | Files | Tests |
|--------|---|---------|--------|------|-------|-------|
| 1 | 1 | StateGraph | ✅ | 2026-06-17 | core/settings.py, core/logging.py, core/graph_builder.py, state.py, graph.py, main.py | — |
| 1 | 2 | State | ✅ | 2026-06-17 | state.py (complete AgentState) | — |
| 1 | 3 | TypedDict | ✅ | 2026-06-17 | state.py (TypeAliases + __all__) | tests/unit/test_state.py (38 tests) |
| 1 | 4 | Nodes | ✅ | 2026-06-17 | agents/__init__.py, agents/nodes.py, graph.py (refactored), tests/conftest.py | tests/unit/test_nodes.py (24 tests) |
| 1 | 5 | Edges | ✅ | 2026-06-17 | pyproject.toml, llm_provider.py (with_retry fix), tests/integration/ | tests/integration/test_graph.py (13 tests) |
| 1 | 6 | Conditional Edges | ✅ | 2026-06-17 | state.py (RouterFn), agents/nodes.py (route_ticket + 4 stubs), graph.py (4-way routing) | test_nodes.py +14 tests, test_graph.py +6 tests |
| 1 | 7 | Compile | ✅ | 2026-06-17 | graph.py (build_graph params + get_graph_info), core/graph_builder.py (build() docs) | tests/unit/test_compile.py (26 tests) |
| 1 | 8 | Invoke | ✅ | 2026-06-17 | main.py (_build_config, _build_initial_state, run) | tests/unit/test_invoke.py (42 tests) |
| 2 | 9 | Reducers | ✅ | 2026-06-21 | state.py (6 reducer fields confirmed) | tests/unit/test_reducers.py (27 tests) |
| 2 | 10 | Annotated State | ✅ | 2026-06-21 | state.py (all Annotated fields) | tests/unit/test_reducers.py::TestAnnotatedStateDeclaration |
| 2 | 11 | operator.add | ✅ | 2026-06-21 | state.py (5 operator.add fields) | tests/unit/test_reducers.py::TestOperatorAddBehaviour |
| 2 | 12 | add_messages | ✅ | 2026-06-21 | state.py (messages field) | tests/unit/test_reducers.py::TestAddMessagesReducer |
| 2 | 13 | State Merging | ✅ | 2026-06-21 | agents/nodes.py (partial returns) | tests/unit/test_reducers.py::TestStateMerging |
| 2 | 14 | Reducer Design Patterns | ✅ | 2026-06-21 | tests/unit/test_reducers.py (test-local `_dedup_by_source` custom reducer; state.py defines none) | tests/unit/test_reducers.py::TestCustomReducers |
| 3 | 15 | Router Pattern | ✅ | 2026-06-25 | agents/nodes.py (route_ticket, ROUTE_KEYS) | tests/unit/test_module3.py::TestRouterPattern |
| 3 | 16 | ToolNode | ✅ | 2026-06-25 | tools/__init__.py, tools/search_tool.py, tools/kb_tool.py | tests/unit/test_module3.py::TestToolContracts |
| 3 | 17 | create_react_agent | ✅ | 2026-06-25 | agents/base.py (BaseAgent contract) | tests/unit/test_module3.py::TestBaseAgentABC |
| 3 | 18 | Dynamic Routing | ✅ | 2026-06-25 | agents/base.py, agents/registry.py | tests/unit/test_module3.py::TestAgentRegistry |
| 3 | 19 | Hooks | ✅ | 2026-06-25 | observability/hooks.py, observability/__init__.py | tests/unit/test_module3.py::TestHooks |
| 3 | 20 | RunnableConfig | ✅ | 2026-06-25 | agents/nodes.py (_get_thread_id) | tests/unit/test_module3.py::TestRunnableConfig |
| 3 | 21 | Runtime Configuration | ✅ | 2026-06-25 | observability/hooks.py (_latency_hook gated) | tests/unit/test_module3.py::TestRuntimeConfiguration |
| 4 | 22 | Graph Memory | ✅ | 2026-06-25 | core/factories.py, core/settings.py (checkpointer_backend) | tests/unit/test_persistence.py::TestGraphMemory |
| 4 | 23 | Checkpointing | ✅ | 2026-06-25 | core/factories.py (get_checkpointer) | tests/unit/test_persistence.py::TestCheckpointerFactory |
| 4 | 24 | InMemorySaver | ✅ | 2026-06-25 | tests/conftest.py (memory_checkpointer fixture) | tests/unit/test_persistence.py::TestInMemorySaver |
| 4 | 25 | SQLite Checkpointer | ✅ | 2026-06-25 | core/factories.py (sqlite3.connect direct) | tests/unit/test_persistence.py::TestPersistentThreads |
| 4 | 26 | Postgres Checkpointer | ✅ | 2026-06-25 | core/factories.py (LSP: both extend BaseCheckpointSaver) | — |
| 4 | 27 | Thread IDs | ✅ | 2026-06-25 | main.py (uuid4, _build_config) | tests/unit/test_persistence.py::TestThreadAndMetadata |
| 4 | 28 | User IDs | ✅ | 2026-06-25 | state.py (TicketMetadata) | tests/unit/test_persistence.py::TestThreadAndMetadata |
| 4 | 29 | Metadata | ✅ | 2026-06-25 | state.py (TicketMetadata) | tests/unit/test_persistence.py::TestThreadAndMetadata |
| 4 | 30 | Persistent Threads | ✅ | 2026-06-25 | main.py (session_id reused across loop) | tests/unit/test_persistence.py::TestPersistentThreads |
| 4 | 31 | Checkpoint Storage | ✅ | 2026-06-25 | graph.get_state_history() pattern | tests/unit/test_persistence.py::TestCheckpointStorage |
| 4 | 32 | State Recovery | ✅ | 2026-06-25 | idempotency pattern: check-before-act | tests/unit/test_persistence.py::TestIdempotencyAndDurableExecution |
| 4 | 33 | Durable Execution | ✅ | 2026-06-25 | side-effect isolation in dedicated nodes | tests/unit/test_persistence.py::TestIdempotencyAndDurableExecution |
| 4 | 34 | Time Travel | ✅ | 2026-06-25 | checkpoint_id in config["configurable"] | tests/unit/test_persistence.py::TestTimeTravel |
| 5 | 35 | Long-Term Memory | ✅ | 2026-06-25 | memory/ package created | tests/unit/test_memory.py::TestLongTermMemoryStore |
| 5 | 36 | Memory Store | ✅ | 2026-06-25 | memory/store.py, get_memory_store() factory | tests/unit/test_memory.py::TestMemoryStoreFactory |
| 5 | 37 | Semantic Memory | ✅ | 2026-06-25 | _embed_fn in get_memory_store(), graceful fallback | tests/unit/test_memory.py |
| 5 | 38 | Memory Retrieval | ✅ | 2026-06-25 | memory_retrieval_node in agents/nodes.py | tests/unit/test_memory.py::TestMemoryRetrievalNode |
| 5 | 39 | Memory Summarization | ✅ | 2026-06-25 | _summarise_for_memory() helper | tests/unit/test_memory.py::TestSummariseForMemory |
| 5 | 40 | Memory Compression | ✅ | 2026-06-25 | memory_write_node in agents/nodes.py | tests/unit/test_memory.py::TestMemoryWriteNode |
| 5 | 41 | Config Driven Workflows | ✅ | 2026-06-25 | memory_top_k, memory_summarize_after_turns in settings | tests/unit/test_memory.py::TestConfigDrivenMemory |
| 6 | 42 | graph.stream() | ✅ | 2026-06-25 | — (sync variant, educational context) | tests/unit/test_streaming.py::TestStreamVsInvoke |
| 6 | 43 | graph.astream() | ✅ | 2026-06-25 | main.py (_stream_response — two-pass) | tests/unit/test_streaming.py::TestStreamVsInvoke |
| 6 | 44 | Stream Modes | ✅ | 2026-06-25 | main.py (updates + messages) | tests/unit/test_streaming.py::TestStreamModes |
| 6 | 45 | values mode | ✅ | 2026-06-25 | — (educational; updates preferred) | tests/unit/test_streaming.py::TestStreamModes |
| 6 | 46 | updates mode | ✅ | 2026-06-25 | main.py Pass 1: collects ticket_type/agent | tests/unit/test_streaming.py::TestStreamModes |
| 6 | 47 | messages mode | ✅ | 2026-06-25 | main.py Pass 2: token-by-token display | tests/unit/test_streaming.py::TestMessagesMode |
| 6 | 48 | debug mode | ✅ | 2026-06-25 | graph.py build_graph(debug=settings.graph_debug) | tests/unit/test_streaming.py::TestStreamingArchitecture |
| 6 | 49 | Token Streaming | ✅ | 2026-06-25 | agents/nodes.py respond_node (llm.astream) | tests/unit/test_streaming.py::TestTokenStreaming |
| 6 | 50 | Event Streaming | ✅ | 2026-06-25 | — (astream_events educational) | tests/unit/test_streaming.py::TestEventStreaming |
| 6 | 51 | Streaming UIs | ✅ | 2026-06-25 | main.py _print_node_progress, _stream_response | tests/unit/test_invoke.py::TestPrintNodeProgress |
| 7 | 52 | Interrupt | ✅ | 2026-06-25 | agents/nodes.py (billing_node sets pending_approval) | tests/unit/test_hitl.py::TestBillingNodeInterrupt |
| 7 | 53 | interrupt_before | ✅ | 2026-06-25 | graph.py (default interrupt_before=["approval_check"]) | tests/unit/test_hitl.py::TestInterruptBefore |
| 7 | 55 | Resume | ✅ | 2026-06-25 | main.py (_handle_interrupt: update_state + ainvoke(None)) | tests/unit/test_hitl.py |
| 7 | 60 | Approval Workflows | ✅ | 2026-06-25 | workflows/approval.py (_extract_refund_amount, check_approval_required, build_approval_payload) | tests/unit/test_hitl.py::TestApprovalHelpers |
| 7 | 62 | Escalation Flows | ✅ | 2026-06-25 | workflows/escalation.py (EscalationLevel, escalate_node) | tests/unit/test_workflows.py |
| 7 | 65 | Workflow State Machines | ✅ | 2026-06-25 | state.py (TicketStatus str-enum) | tests/unit/test_workflows.py::TestTicketStatus |
| 7 | 69 | SLA Workflows | ✅ | 2026-06-25 | workflows/sla.py (compute_sla_deadline, is_sla_breached) | tests/unit/test_workflows.py::TestSLAFunctions |
| 7 | 70 | Audit Trails | ✅ | 2026-06-25 | observability/audit.py (create_audit_event, append-only audit_events) | tests/unit/test_workflows.py::TestAuditHelpers |
| 7 | 71 | Compliance Pipelines | ✅ | 2026-06-25 | security/compliance.py (detect_pii, compliance_gate_node) | tests/unit/test_workflows.py::TestPIIDetection |
| 8 | 73 | Subgraphs | ✅ | 2026-06-25 | rag/graph.py (get_rag_graph — CRAG built as a standalone subgraph; not yet invoked by the app, see Known limitations in README) | tests/rag/test_rag_pipeline.py |
| 8 | 74 | Supervisor Pattern | ✅ | 2026-06-25 | agents/supervisor.py (supervisor_node + registry dispatch) | tests/unit/test_multi_agent.py::TestSupervisorNode |
| 8 | 75 | Worker Agents | ✅ | 2026-06-25 | agents/{billing,technical,general,research}_agent.py (BaseAgent ABC) | tests/unit/test_multi_agent.py::TestBaseAgentContract |
| 8 | 76 | Specialist Agents | ✅ | 2026-06-25 | BillingAgent, TechnicalAgent, GeneralAgent, ResearchAgent | tests/unit/test_multi_agent.py |
| 8 | 82 | Agent Coordination | ✅ | 2026-06-25 | state.agent_results (operator.add reducer fan-in) | tests/unit/test_multi_agent.py::TestResearchCollectNode |
| 8 | 84 | Send API | ✅ | 2026-06-25 | agents/supervisor.py (route_after_supervisor returns list[Send]) | tests/unit/test_multi_agent.py::TestRouteAfterSupervisor |
| 8 | 85 | Fan-Out | ✅ | 2026-06-25 | kb_worker_node + web_worker_node (parallel via Send) | tests/unit/test_multi_agent.py::TestKbWorkerNode, TestWebWorkerNode |
| 8 | 86 | Fan-In | ✅ | 2026-06-25 | research_collect_node + operator.add on agent_results | tests/unit/test_multi_agent.py::TestResearchCollectNode |
| 9 | 89 | RAG Subgraphs | ✅ | 2026-06-26 | rag/__init__.py, rag/graph.py (CRAG), rag/protocols.py | tests/rag/test_rag_pipeline.py |
| 9 | 90 | Retrieval Nodes | ✅ | 2026-06-26 | rag/retriever.py (make_retriever_node DI factory) | tests/rag/test_rag_pipeline.py::TestRetrieverNode |
| 9 | 91 | Query Rewriting | ✅ | 2026-06-26 | rag/graph.py (rewrite_query_node) | tests/rag/test_rag_pipeline.py::TestCRAGNodes |
| 9 | 93 | Retrieval Graders | ✅ | 2026-06-26 | rag/grader.py (grade_document, grade_retrieval) | tests/rag/test_rag_pipeline.py::TestGraders |
| 9 | 94 | Hallucination Graders | ✅ | 2026-06-26 | rag/grader.py (check_hallucination) | tests/rag/test_rag_pipeline.py::TestGraders |
| 9 | 95 | CRAG | ✅ | 2026-06-26 | rag/graph.py (full CRAG loop: retrieve→grade→rewrite/generate→hallucination_check) | tests/rag/test_rag_pipeline.py::TestCRAGNodes |
| 9 | 99 | Document Ingestion | ✅ | 2026-06-26 | rag/ingest.py (load_kb_documents, build_index), rag/chunking/ (FixedSizeChunker, SemanticChunker) | tests/rag/test_rag_pipeline.py |
| 9 | 100 | Vector Stores | ✅ | 2026-06-26 | rag/stores/faiss_store.py (FAISSVectorStore, VectorStoreProtocol) | tests/rag/test_rag_pipeline.py::TestFAISSVectorStore |
| 9 | 101 | Embeddings | ✅ | 2026-06-26 | llm_provider.get_embeddings() — never instantiate directly | tests/rag/ |
| 9 | 102 | Reranking | ✅ | 2026-06-26 | rag/retriever.py (_rerank with CrossEncoder) | tests/rag/test_rag_pipeline.py::TestRetrieverNode |
| 9 | 106 | RAG Evaluation | ✅ | 2026-06-26 | rag/evaluator.py (recall, faithfulness, relevance, run_eval_suite) | tests/rag/test_rag_pipeline.py::TestEvaluator |
| 10 | 109 | Authentication | ✅ | 2026-06-26 | security/auth.py (validate_jwt, build_auth_context, create_jwt, auth_node) | tests/security/test_security.py::TestJWT + TestAuthNode |
| 10 | 110 | Authorization | ✅ | 2026-06-26 | core/exceptions.py (custom hierarchy: AuthenticationError, AuthorizationError, ProviderError, ValidationError) | tests/security/test_security.py::TestExceptionHierarchy |
| 10 | 111 | RBAC | ✅ | 2026-06-26 | security/rbac.py (ROLE_PERMISSIONS, has_permission, require_permission, rbac_check_node) | tests/security/test_security.py::TestRBAC + TestRBACNode |
| 10 | 112 | Multi-Tenant Graphs | ✅ | 2026-06-26 | tenant_id flows through all logs, audit events, and LangSmith metadata | — |
| 10 | 113 | Governance | ✅ | 2026-06-26 | observability/audit.py (create_audit_event), state.py (audit_events: operator.add) | tests/unit/test_workflows.py |
| 10 | 114 | Auditability | ✅ | 2026-06-26 | audit_events reducer (append-only by construction) | — |
| 10 | 115 | Compliance | ✅ | 2026-06-26 | security/compliance.py (detect_pii, compliance_gate_node) | tests/unit/test_workflows.py |
| 10 | 116 | LangSmith Tracing | ✅ | 2026-06-26 | observability/tracing.py (configure_tracing, get_run_metadata, is_tracing_enabled) | tests/security/test_security.py::TestTracing |
| 10 | 117 | Graph Observability | ✅ | 2026-06-26 | observability/metrics.py (record_node_latency, get_session_metrics, configure_metrics) | tests/security/test_security.py::TestMetrics |
| 10 | 118 | Graph Testing | ✅ | 2026-06-26 | 61 new tests: test_security.py (46) + test_api.py (15). Full suite: 541 tests | tests/security/ |
| 10 | 119 | Graph Debugging | ✅ | 2026-06-26 | GRAPH_DEBUG setting gates debug mode in build_graph, FastAPI /docs, /metrics endpoint | — |
| 10 | 120 | Enterprise Deployment | ✅ | 2026-06-26 | api/routes.py (create_app factory, JWT Bearer dep, SSE streaming, /health, /tickets, /auth/token) | tests/security/test_api.py (15 tests) |

---

## Concept 1: StateGraph  [Module 1]
**Date:** 2026-06-17 | **Complexity:** Beginner | **Layer:** Graph

### One-line summary
A `StateGraph` is a directed graph where nodes are Python async functions and edges are transition rules, all sharing a typed state dict.

### Production standard applied
`core/settings.py` created first — all config via pydantic-settings, never `os.getenv()` in app code.
`core/graph_builder.py` introduced — fluent builder centralises all `add_node`/`add_edge` calls.

### When to reach for it
- Multi-step LLM workflow with branching or loops
- You need persistence, HITL, streaming, or time travel
- Multiple agents need shared state

### The canonical pattern
```python
from langgraph.graph import END, StateGraph
from core.graph_builder import GraphBuilder
from state import AgentState

graph = (
    GraphBuilder(AgentState)
    .node("classify", classify_node)
    .node("respond", respond_node)
    .edge("classify", "respond")
    .edge("respond", END)
    .entry("classify")
    .build()                        # no checkpointer yet — Module 4
)

# async nodes → must use ainvoke, not invoke
result = await graph.ainvoke(initial_state, config=config)
```

### The most common mistake
Using sync `graph.invoke()` when any node is `async def` — raises `TypeError: No synchronous function provided`. Fix: use `await graph.ainvoke()` inside `asyncio.run()`.

### Where it lives in our app
File: `graph.py` → function: `build_graph()`
Builder: `core/graph_builder.py` → class: `GraphBuilder`
Entry: `main.py` → function: `run()`

---

## Concept 2: State  [Module 1]
**Date:** 2026-06-17 | **Complexity:** Beginner | **Layer:** Infrastructure

### One-line summary
State is the shared `TypedDict` whiteboard every node reads from and writes to — the single source of truth for a graph execution.

### Production standard applied
`state.py` written before `graph.py` — it is the contract everything else depends on.
Nested `TypedDict` sub-objects (`TicketMetadata`, `RetrievedDocument`) instead of raw `dict`.
Fields for all 10 modules defined upfront with comments marking when each is introduced.

### When to reach for it
- Every `StateGraph` must have a typed state schema
- Add a field whenever a node introduces new data
- Use nested TypedDict for any structured sub-object

### The canonical pattern
```python
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # reducer = append+dedup
    ticket_type: str | None                               # scalar = replace
    retrieved_docs: Annotated[list[RetrievedDocument], operator.add]  # reducer = append

# Nodes return PARTIAL dicts — only what they changed
async def my_node(state: AgentState, config: RunnableConfig) -> dict:
    return {"ticket_type": "billing"}   # ✅ partial
    # NOT return state                  # ❌ never full state
```

### The most common mistake
Returning the full `AgentState` from a node instead of a partial dict. LangGraph merges partial dicts — returning full state can silently overwrite other nodes' results.

### Where it lives in our app
File: `state.py` → class: `AgentState`, `TicketMetadata`, `RetrievedDocument`

---

## Concept 3: TypedDict  [Module 1]
**Date:** 2026-06-17 | **Complexity:** Beginner | **Layer:** Infrastructure

### One-line summary
`TypedDict` gives a plain Python `dict` a fixed schema that mypy and IDEs check at analysis time — zero runtime overhead, full type safety.

### Production standard applied
`TypeAlias` definitions added to `state.py` for all repeated literal types.
`__all__` declared in `state.py` — explicit public API contract.
`tests/unit/test_state.py` — 38 tests guarding the schema contract.

### When to reach for it
- Every LangGraph state schema
- Any dict with a fixed known shape passed between functions
- Structured sub-objects inside state (never raw `dict`)

### The canonical pattern
```python
from typing import Literal, TypeAlias, TypedDict

TicketType: TypeAlias = Literal["billing", "technical", "general", "research"]

class AgentState(TypedDict):
    ticket_type: TicketType | None    # TypeAlias catches routing typos
    escalation_level: int             # not str — wrong type caught by mypy

__all__ = ["AgentState", "TicketType", ...]
```

### The most common mistake
Using `dict` for structured sub-objects inside state (`pending_approval: dict | None`). Define a TypedDict instead — gives autocomplete and mypy checking all the way into nested fields.

### Where it lives in our app
File: `state.py` → `TicketType`, `AgentName`, `GradeLabel`, `ApprovalDecision`, `Priority`
Tests: `tests/unit/test_state.py` → `TestTypeAliases`, `TestPublicAPI`

---

## Concept 4: Nodes  [Module 1]
**Date:** 2026-06-17 | **Complexity:** Beginner | **Layer:** Graph

### One-line summary
A node is an async function `(state, config) -> partial_dict` that does exactly one unit of work and writes only the fields it changed.

### Production standard applied
Node template: `@with_retry` + typed args + `logger.debug/info` + `_prefixed` pure helpers + partial dict return.
Separation: node functions in `agents/nodes.py`, topology registration in `graph.py`.
`tests/conftest.py` created with shared fixtures (`base_state`, `runnable_config`, `mock_llm`).

### When to reach for it
- Every step in the graph is a node
- One node = one responsibility — never combine two concerns
- Node depends on LLM → `@with_retry` is mandatory

### The canonical pattern
```python
@with_retry
async def my_node(state: AgentState, config: RunnableConfig) -> dict:
    """Reads: ...\nWrites: ..."""
    messages = state.get("messages", [])      # extract at top
    thread_id = _get_thread_id(config)        # safe config access
    logger.debug("my_node: thread=%s", thread_id)
    result = _pure_helper(messages[-1].content)
    logger.info("my_node: thread=%s result=%s", thread_id, result)
    return {"field_i_own": result}            # partial dict only
```

### The most common mistake
`state["field"] = value` — direct mutation. Always `return {"field": value}`.
Especially dangerous in parallel fan-out (Module 8) where mutations from two nodes can race and corrupt state silently.

### Where it lives in our app
File: `agents/nodes.py` → `classify_ticket_node()`, `respond_node()`
Tests: `tests/unit/test_nodes.py` → `TestClassifyTicketNode`, `TestRespondNode`
Fixtures: `tests/conftest.py` → `base_state`, `runnable_config`, `mock_llm`

---

## Concept 5: Edges  [Module 1]
**Date:** 2026-06-17 | **Complexity:** Beginner | **Layer:** Graph

### One-line summary
An edge is a directed connection between two nodes; unconditional edges always route from A to B; every terminal node needs `.edge(node, END)` or the graph hangs silently.

### Production standard applied
`pyproject.toml` created — pytest `asyncio_mode = "auto"`, ruff, mypy config.
`tests/integration/test_graph.py` — topology tests verify edges without LLM calls.
`with_retry` fixed with `functools.update_wrapper` — preserves signature for LangGraph's `inspect.signature` check.

### When to reach for it
- Next node is always the same (no branching)
- Every terminal node must connect to END
- When you want topology tests to guard against wiring regressions

### The canonical pattern
```python
GraphBuilder(AgentState)
    .node("classify_ticket", classify_ticket_node)
    .node("respond", respond_node)
    .edge("classify_ticket", "respond")   # unconditional
    .edge("respond", END)                 # REQUIRED — missing = silent hang
    .entry("classify_ticket")
    .build()
```

### The most common mistake
Forgetting `.edge(last_node, END)`. The graph compiles without error, the last node runs, but execution never terminates — `ainvoke` hangs forever. Always add an END edge and test with `test_no_unexpected_nodes`.

### Production bug caught
`@with_retry` stripped function signature → LangGraph called nodes with only `(state,)`, skipping `config`. Fixed with `functools.update_wrapper(wrapper, func)` in `llm_provider.py`.

### Where it lives in our app
File: `graph.py` → `build_graph()` — all edges declared here
Tests: `tests/integration/test_graph.py` → `TestGraphEdges`, `TestGraphInvocation`

---

## Concept 6: Conditional Edges  [Module 1]
**Date:** 2026-06-17 | **Complexity:** Beginner | **Layer:** Graph

### One-line summary
A conditional edge calls a pure sync router function on the current state and routes to the matching destination in a mapping dict — the graph's branching mechanism.

### Production standard applied
`RouterFn: TypeAlias = Callable[[AgentState], str]` added to `state.py`.
`ROUTE_KEYS` defined in `agents/nodes.py`, imported into `graph.py` — single source of truth for routing mapping.
`TestConditionalRouting` — parametrised integration test for all 4 routes.

### When to reach for it
- Different state values should produce different execution paths
- Multiple specialist agents need separate branches
- Retry/fallback loops (CRAG in Module 9)

### The canonical pattern
```python
# Router — SYNC, pure, always has a fallback
def route_ticket(state: AgentState) -> str:
    ticket_type = state.get("ticket_type")
    if ticket_type not in ROUTE_KEYS:
        return "general"          # fallback — never raise
    return ticket_type

# Registration — ROUTE_KEYS shared between router and graph.py
.conditional_edge("classify_ticket", route_ticket, ROUTE_KEYS)
```

### The most common mistake
`async def route_ticket(...)`. LangGraph calls routers synchronously — an async router returns a coroutine object instead of a string, silently routing everything wrong with no error raised.

## Concept 7: Compile  [Module 1]
**Date:** 2026-06-17 | **Complexity:** Beginner | **Layer:** Graph

### One-line summary
`compile()` turns a mutable `StateGraph` blueprint into an immutable, runnable `CompiledStateGraph` — attaching the Pregel engine, state merging, checkpointer, streaming, and interrupt support in one call.

### Production standard applied
`build_graph(checkpointer, interrupt_before, debug)` — explicit params, not `**kwargs`.
`get_graph_info()` — development utility for Mermaid visualisation.
`test_compile.py` — 26 tests covering immutability, interface, kwargs passthrough.

### When to reach for it
- Always: `compile()` once at startup, reuse for all requests
- With checkpointer: whenever persistence is needed (Module 4)
- With `interrupt_before`: HITL pause points (Module 7)

### The canonical pattern
```python
def build_graph(checkpointer=None, interrupt_before=None, debug=False):
    compile_kwargs = {}
    if checkpointer: compile_kwargs["checkpointer"] = checkpointer
    if interrupt_before: compile_kwargs["interrupt_before"] = interrupt_before
    if debug: compile_kwargs["debug"] = debug
    return GraphBuilder(AgentState).node(...).build(**compile_kwargs)

# Tests inject InMemorySaver — no SQLite file created
graph = build_graph(checkpointer=InMemorySaver())
```

### The most common mistake
Adding a node to the `StateGraph` after calling `compile()`. No error is raised — the compiled graph silently ignores the change. Always modify topology, then call `build()` again.

### Production discoveries
- `interrupt_before` without checkpointer: compiles OK, fails only at resume time
- `draw_mermaid()` prepends YAML frontmatter (`---`) in LangGraph >= 0.2

### Where it lives in our app
File: `graph.py` → `build_graph()`, `get_graph_info()`
File: `core/graph_builder.py` → `GraphBuilder.build()`
Tests: `tests/unit/test_compile.py`

---

### Where it lives in our app
File: `agents/nodes.py` → `route_ticket()`, `ROUTE_KEYS`
File: `graph.py` → `.conditional_edge("classify_ticket", _route_ticket, ROUTE_KEYS)`
Tests: `tests/unit/test_nodes.py` → `TestRouteTicket`
Tests: `tests/integration/test_graph.py` → `TestConditionalRouting`

---

## Concept 8: Invoke  [Module 1]
**Date:** 2026-06-17 | **Complexity:** Beginner | **Layer:** Delivery

### One-line summary
`await graph.ainvoke(state, config)` runs the compiled graph to completion and returns the final `AgentState` dict — the blocking async entry point for all graph execution.

### Production standard applied
Compile-once pattern: `build_graph()` called once at session start, reused across all messages.
`session_id = uuid4()` generated once per process, passed as `thread_id` in every config.
`_build_config()` enforces correct `configurable` nesting — the #1 ainvoke mistake caught in tests.

### When to reach for it
- One response needed before proceeding (API endpoints, CLI, batch jobs, integration tests)
- Simple workflows where blocking on the complete result is acceptable
- Any time you don't need token-level streaming (Module 6 upgrades this to `astream`)

### The canonical pattern
```python
# Compile once at startup
graph = build_graph(debug=settings.graph_debug)
session_id = str(uuid.uuid4())

# Per message
config = {
    "configurable": {"thread_id": session_id},   # ← MUST be under configurable
    "recursion_limit": settings.graph_recursion_limit,
}
result = await graph.ainvoke(initial_state, config=config)
response = result.get("final_response") or "No response generated."
```

### The most common mistake
`config = {"thread_id": "abc"}` — thread_id at the top level is silently ignored by LangGraph; the checkpointer never sees it. Always nest under `config["configurable"]["thread_id"]`.

### Where it lives in our app
File: `main.py` → `run()`, `_build_config()`, `_build_initial_state()`, `_display_result()`
Tests: `tests/unit/test_invoke.py` → `TestBuildConfig`, `TestBuildInitialState`, `TestRunSessionLoop`, `TestGraphCompiledOnce`

---

## Concept 9: Reducers  [Module 2]
**Date:** 2026-06-21 | **Complexity:** Intermediate | **Layer:** Infrastructure (state schema)

### One-line summary
A reducer is a function declared on a state field that tells LangGraph how to merge a node's return value — append vs replace — preventing silent data loss when multiple nodes write to the same field.

### Production standard applied
All 6 list fields in AgentState that multiple nodes write to use `Annotated[list[T], reducer]`. Declared upfront in Module 1 so no field ever lacks a reducer. Custom reducers live at module level (not inside functions) to avoid `NameError` from lazy annotation evaluation with `from __future__ import annotations`.

### When to reach for it
- Any list field that two or more nodes write to in the same graph run
- Audit trails, accumulated results, multi-query retrieval docs, fan-out workers
- Chat history where checkpoint replay could otherwise create duplicates

### The canonical pattern
```python
# state.py — declare reducer inline with the field
audit_events: Annotated[list[dict], operator.add]
# Node — return ONLY new items, never read-then-write
return {"audit_events": [{"action": "refund_requested"}]}
```

### The most common mistake
Returning `state["field"] + [new_item]` from a node — the reducer then applies `operator.add(state["field"], state["field"] + [new_item])`, doubling every pre-existing entry on every checkpoint replay.

### Where it lives in our app
File: `state.py` → all 6 `Annotated` fields
Tests: `tests/unit/test_reducers.py` → all 5 test classes, 27 tests

---

## Concept 12: add_messages  [Module 2]
**Date:** 2026-06-21 | **Complexity:** Intermediate | **Layer:** Infrastructure (state schema)

### One-line summary
`add_messages` is `operator.add` plus deduplication by `message.id` — required for chat history because HITL checkpoint resume replays prior messages, and without deduplication the same message appears twice.

### The canonical pattern
```python
messages: Annotated[list[BaseMessage], add_messages]
# Dedup in action:
add_messages([AIMessage(content="v1", id="ai-1")],
             [AIMessage(content="v2", id="ai-1")])
# → [AIMessage("v2")]  — one message, latest version
```

### The most common mistake
Using `operator.add` for messages: it appends correctly on fresh runs but duplicates messages on every HITL resume or checkpoint replay.

### Where it lives in our app
File: `state.py` → `messages` field only (all other list fields use `operator.add`)
Tests: `tests/unit/test_reducers.py` → `TestAddMessagesReducer`

---

## Concept 22: Graph Memory  [Module 4]
**Date:** 2026-06-25 | **Complexity:** Beginner | **Layer:** Infrastructure

### One-line summary
Without a checkpointer every invocation starts blank; with one, the same `thread_id` accumulates state across calls.

### Production standard applied
Factory Pattern — `get_checkpointer()` in `core/factories.py` hides backend selection behind a stable interface.

### When to reach for it
- Multi-turn conversation — customer sends multiple messages in one session
- Resume after crash — graph must survive `kill -9`
- Time travel / debugging — replay from any historical checkpoint

### The canonical pattern
```python
# Seed on first call; subsequent calls pass {} to resume from checkpoint
graph.invoke({"messages": [], "count": 0}, config=config)
graph.invoke({}, config=config)   # count provided by checkpoint, not input
```

### The most common mistake
Passing `{"count": 0}` on every invocation resets the non-reducer `count` field, defeating the checkpointer.

### Where it lives in our app
File: `core/factories.py` → `get_checkpointer()` | `main.py` → wired into `build_graph()`
Tests: `tests/unit/test_persistence.py::TestGraphMemory`

---

## Concept 23: Checkpointing  [Module 4]
**Date:** 2026-06-25 | **Complexity:** Intermediate | **Layer:** Infrastructure

### One-line summary
`get_checkpointer()` resolves `InMemorySaver` (tests) or `SqliteSaver` (dev/prod) from `CHECKPOINTER_BACKEND` in settings.

### Production standard applied
Factory Pattern (no `lru_cache`) — `SqliteSaver` holds an open DB connection; caching it causes "database is locked" in test suites.

### When to reach for it
- Any time the graph needs to resume across invocations or process restarts

### The canonical pattern
```python
# core/factories.py — NOT lru_cached
def get_checkpointer():
    settings = get_settings()
    if settings.checkpointer_backend == "memory":
        return MemorySaver()
    if settings.checkpointer_backend == "sqlite":
        import sqlite3
        conn = sqlite3.connect(settings.sqlite_path, check_same_thread=False)
        return SqliteSaver(conn)
```

### The most common mistake
Using `SqliteSaver.from_conn_string(path)` directly — it returns a context manager, not an instance. Pass to `graph.compile()` and it raises: `Expected BaseCheckpointSaver, received _GeneratorContextManager`.

### Where it lives in our app
File: `core/factories.py` → `get_checkpointer()`
Tests: `tests/unit/test_persistence.py::TestCheckpointerFactory`

---

## Concept 34: Time Travel  [Module 4]
**Date:** 2026-06-25 | **Complexity:** Advanced | **Layer:** Infrastructure

### One-line summary
Pass `checkpoint_id` in `config["configurable"]` to resume graph execution from any past snapshot, not just the latest.

### Production standard applied
Typed config pattern — `checkpoint_id` lives under `"configurable"`, same key namespace as `thread_id`.

### When to reach for it
- Debugging: replay from just before a node failed
- Testing: fork from a known state, try different inputs
- Audit: replay any historical conversation verbatim

### The canonical pattern
```python
history = list(graph.get_state_history(config))
snapshot = next(s for s in history if s.values.get("count") == 1)

time_travel_config = {
    "configurable": {
        "thread_id": "my-thread",
        "checkpoint_id": snapshot.config["configurable"]["checkpoint_id"],
    }
}
result = graph.invoke({}, config=time_travel_config)  # {} = let checkpoint fill state
```

### The most common mistake
Passing state values in the time-travel invocation (e.g. `{"count": 0}`) — they override the historical checkpoint, defeating time travel.

### Where it lives in our app
Tests: `tests/unit/test_persistence.py::TestTimeTravel`

---

## Concept 42–44: graph.stream() / graph.astream() / Stream Modes  [Module 6]
**Date:** 2026-06-25 | **Complexity:** Intermediate | **Layer:** Delivery

### One-line summary
`astream()` is an async generator that yields graph execution events chunk-by-chunk; `stream_mode` controls what each chunk contains.

### Production standard applied
All streaming + print calls live in `main.py` only (delivery layer). Nodes never print — they return state dicts.

### When to reach for it
- Any user-facing CLI or API that should show progress or tokens as they arrive
- Debugging: `stream_mode="updates"` shows which nodes ran and what they wrote
- Token streaming to users: `stream_mode="messages"` gives AIMessageChunk word-by-word

### The canonical pattern
```python
# Pass 1: progress + metadata collection
async for event in graph.astream(initial_state, config=config, stream_mode="updates"):
    for node_name, node_output in event.items():
        print(f"  [{node_name}]", end="\r", flush=True)
        if "ticket_type" in node_output:
            ticket_type = node_output["ticket_type"]

# Pass 2: token streaming (resume from checkpoint — pass {} not initial_state)
async for chunk, metadata in graph.astream({}, config=config, stream_mode="messages"):
    if metadata.get("langgraph_node") == "respond" and chunk.content:
        print(chunk.content, end="", flush=True)
```

### The most common mistake
Re-passing `initial_state` in Pass 2 — this starts the graph from scratch and re-runs all nodes. Pass `{}` so LangGraph resumes from the checkpoint written by Pass 1.

### Where it lives in our app
File: `main.py` → `_stream_response()`
Tests: `tests/unit/test_streaming.py` → `TestStreamVsInvoke`, `TestStreamModes`

---

## Concept 45–46: values mode / updates mode  [Module 6]
**Date:** 2026-06-25 | **Complexity:** Beginner | **Layer:** Delivery

### One-line summary
`"values"` yields the full state after every node; `"updates"` yields only the fields each node changed — much lighter for production use.

### Production standard applied
Use `"updates"` in production — `"values"` duplicates the entire state on every node, which is expensive for large state objects. Gate `"values"` behind `settings.graph_debug`.

### When to reach for it
- `"updates"`: production logging, UI progress indicators, capturing routing metadata
- `"values"`: debugging only — need to see full state at each step

### The canonical pattern
```python
# ✅ Production: updates mode
async for event in graph.astream(state, config=config, stream_mode="updates"):
    for node_name, partial_state in event.items():
        logger.debug("node=%s delta=%s", node_name, list(partial_state.keys()))
```

### The most common mistake
Using `"values"` in a production streaming loop — each event carries the entire state including full message history and retrieved docs. One long conversation can easily send megabytes per invocation.

### Where it lives in our app
File: `main.py` → Pass 1 of `_stream_response()`
Tests: `tests/unit/test_streaming.py::TestStreamModes`

---

## Concept 47: messages mode  [Module 6]
**Date:** 2026-06-25 | **Complexity:** Intermediate | **Layer:** Delivery

### One-line summary
`stream_mode="messages"` yields `(AIMessageChunk, metadata)` tuples as the LLM generates tokens — the only mode that gives true word-by-word output.

### Production standard applied
Token streaming only works when the node calls `llm.astream()` internally. The `streaming_enabled` flag in state switches `respond_node` between `llm.astream()` (CLI) and `llm.ainvoke()` (tests/batch).

### When to reach for it
- Any user-facing UI where response latency > 1s would feel slow
- CLI apps, WebSocket endpoints, SSE streams
- NOT for batch processing or automated pipelines (use ainvoke)

### The canonical pattern
```python
# In respond_node: branch on streaming_enabled
if streaming_enabled:
    chunks = []
    async for chunk in llm.astream(prompt, config=config):
        chunks.append(chunk.content)
    response_text = "".join(chunks)
else:
    response = await llm.ainvoke(prompt, config=config)
    response_text = response.content

# In main.py: display chunks as they arrive
async for chunk, metadata in graph.astream({}, config=config, stream_mode="messages"):
    if metadata.get("langgraph_node") == "respond" and chunk.content:
        print(chunk.content, end="", flush=True)
```

### The most common mistake
Forgetting to filter by `langgraph_node`. Memory nodes, classify nodes, and tool calls all emit messages too — without the filter you'll see intermediate LLM outputs mixed into the user-facing response.

### Where it lives in our app
File: `agents/nodes.py` → `respond_node` (llm.astream branch)
File: `main.py` → Pass 2 of `_stream_response()`
Tests: `tests/unit/test_streaming.py::TestMessagesMode`, `TestTokenStreaming`

---

## Concept 49: Token Streaming  [Module 6]
**Date:** 2026-06-25 | **Complexity:** Intermediate | **Layer:** Graph / Delivery

### One-line summary
`AIMessageChunk` objects arrive one token at a time from `llm.astream()`; join their `.content` strings to reconstruct the full response.

### Production standard applied
Two-pass streaming architecture: Pass 1 (`"updates"`) for metadata collection and progress display; Pass 2 (`"messages"`) for token delivery. This separates concerns cleanly — metadata never blocks token display.

### When to reach for it
- All user-facing LLM calls in CLI or streaming API contexts
- The `streaming_enabled` flag makes it easy to disable in tests without code changes

### The canonical pattern
```python
# Collecting chunks in respond_node (graph layer)
chunks = []
async for chunk in llm.astream(prompt, config=config):
    chunks.append(chunk.content)  # each chunk.content is "" or a few tokens
response_text = "".join(chunks)   # full response after streaming completes

# Displaying chunks in main.py (delivery layer only)
async for chunk, metadata in graph.astream({}, config=config, stream_mode="messages"):
    if metadata.get("langgraph_node") == "respond" and chunk.content:
        print(chunk.content, end="", flush=True)
```

### The most common mistake
Printing chunks inside nodes — violates clean architecture. The delivery layer owns all output. Nodes return state dicts; main.py decides how to display them.

### Where it lives in our app
File: `agents/nodes.py` → `respond_node` lines ~120-130
File: `main.py` → `_stream_response()` Pass 2
Tests: `tests/unit/test_streaming.py::TestTokenStreaming`

---

## Concept 51: Streaming UIs  [Module 6]
**Date:** 2026-06-25 | **Complexity:** Beginner | **Layer:** Delivery

### One-line summary
All user-facing output (tokens, progress labels, error messages) lives exclusively in `main.py` — never inside graph nodes or services.

### Production standard applied
Clean Architecture delivery layer rule: `main.py` is the ONLY file allowed to call `print()`. This keeps nodes pure, testable, and UI-agnostic — swap CLI for FastAPI/WebSocket without touching any node code.

### When to reach for it
- Every time you're tempted to add `print()` to a node — move it to `_stream_response()` instead
- When adding a new delivery channel (FastAPI, Slack bot): only `main.py` / `routes.py` changes

### The canonical pattern
```python
# ✅ _print_node_progress in main.py — node name → human label
_NODE_LABELS = {
    "classify_ticket": "Classifying",
    "respond":         "Generating response",
}
_SILENT_NODES = {"memory_retrieval", "memory_write"}  # fast, no progress needed
_INTERNAL_NODES = {"__start__"}                        # LangGraph internal

def _print_node_progress(node_name: str) -> None:
    if node_name in _INTERNAL_NODES or node_name in _SILENT_NODES:
        return
    label = _NODE_LABELS.get(node_name, node_name)
    print(f"  [{label}]", end="\r", flush=True)
```

### The most common mistake
Adding `print()` calls to nodes for "quick debugging" — they stay there forever and break when the app gains a non-CLI delivery channel (API, tests that capture stdout, etc.).

### Where it lives in our app
File: `main.py` → `_print_node_progress()`, `_stream_response()`, `_print_banner()`
Tests: `tests/unit/test_invoke.py::TestPrintNodeProgress`

---

## Concept 52: Interrupt  [Module 7]
**Date:** 2026-06-25 | **Complexity:** Advanced | **Layer:** Graph

### One-line summary
LangGraph's `interrupt()` function pauses execution mid-node and waits for human input — but it's broken on this project's Python 3.10 runtime, so HITL is done with a state-flag + `interrupt_before` pattern instead.

### Production standard applied
Root-cause first, workaround second: `interrupt()` depends on `var_child_runnable_config` being copied into the thread that runs the node. Python's `loop.run_in_executor()` only copies context vars starting in 3.11, so on 3.10 the var is always `None` and `interrupt()` raises. Rather than pin a newer Python, the app models "needs approval" as **data** (`state["pending_approval"]`), not as a control-flow exception.

### When to reach for it
- ✅ Any Python ≥3.11 runtime — `interrupt()` is the idiomatic primitive there
- ❌ On this project (Python 3.10) — use `billing_node` setting `pending_approval` + `interrupt_before=["approval_check"]` instead
- Detection must happen in a plain node (sync-safe), not inside the interrupt call itself

### The canonical pattern
```python
# agents/nodes.py — billing_node detects the need for approval as a state write,
# NOT by calling interrupt() directly (would raise RuntimeError on Python 3.10)
async def billing_node(state: AgentState, config: RunnableConfig) -> dict:
    amount = _extract_refund_amount(state["messages"][-1].content)
    if amount is not None and amount >= settings.approval_threshold:
        return {"pending_approval": build_approval_payload(amount, state)}
    return {"final_response": _auto_handle(amount)}
```

### The most common mistake
Calling `interrupt()` inside an `async def` node and assuming it "just pauses" — on Python 3.10 it raises `RuntimeError` because the config context var never propagates through `run_in_executor`. Check the Python version before reaching for `interrupt()`; on 3.10, model the pause as a state field instead.

### Where it lives in our app
File: `agents/nodes.py` → `billing_node()` (sets `pending_approval`)
Tests: `tests/unit/test_hitl.py::TestBillingNodeInterrupt`

---

## Concept 53: interrupt_before  [Module 7]
**Date:** 2026-06-25 | **Complexity:** Intermediate | **Layer:** Graph

### One-line summary
`graph.compile(interrupt_before=[...])` is a static, compile-time list of node names the graph will always pause before — the Python-3.10-safe way to build HITL gates.

### Production standard applied
`GraphBuilder.build()` defaults `interrupt_before` to `["approval_check"]` so every caller gets the HITL gate unless explicitly overridden — the safe default is "always pause," not "never pause."

### When to reach for it
- ✅ You know in advance which node(s) must always pause for review (e.g. `approval_check`)
- ❌ You need to pause *conditionally* based on runtime state — `interrupt_before` can't see state; combine it with a router that only enters that node when needed (state-flag pattern from Concept 52)

### The canonical pattern
```python
# graph.py
def build_graph(*, interrupt_before: list[str] | None = None, **kwargs):
    if interrupt_before is None:
        interrupt_before = ["approval_check"]   # safe default: always gate refunds
    compile_kwargs: dict = {}
    if interrupt_before:
        compile_kwargs["interrupt_before"] = interrupt_before
    return builder.build(checkpointer=get_checkpointer(), **compile_kwargs)
```

### The most common mistake
Treating `interrupt_before` as dynamic — it's baked in at `compile()` time. If a node only needs to pause sometimes, route to it conditionally (via `route_after_billing`) rather than trying to skip the interrupt at runtime.

### Where it lives in our app
File: `graph.py` → `build_graph()` (`interrupt_before=["approval_check"]` default)
Tests: `tests/unit/test_hitl.py::TestInterruptBefore`

---

## Concept 55: Resume  [Module 7]
**Date:** 2026-06-25 | **Complexity:** Intermediate | **Layer:** Delivery

### One-line summary
Resuming a paused graph means writing the human's decision into state with `update_state()`, then calling `ainvoke(None, config)` — passing `None` re-enters the same run instead of starting a new one.

### Production standard applied
This project's equivalent of `Command(resume=...)` (LangGraph's Concept 56 in the curriculum folds into this one — there's no separate `Command` object here, just typed state fields) is: `graph.update_state(thread, {"approval_decision": decision})` followed by `graph.ainvoke(None, config=thread)`.

### When to reach for it
- ✅ Whenever a graph is paused at `approval_check` and a human has made a decision
- ❌ Never call `ainvoke(state_dict, config)` to resume — that restarts the graph from `__start__` with a fresh state, discarding everything before the pause

### The canonical pattern
```python
# main.py — resuming after a HITL pause
graph.update_state(thread, {"approval_decision": decision})   # merge decision into paused state
result = await graph.ainvoke(None, config=thread)              # None = "continue this run"
```

### The most common mistake
Passing a state dict instead of `None` to `ainvoke` when resuming — LangGraph treats a non-`None` first argument as a brand-new invocation from the entry node, silently re-running `classify_ticket` etc. and losing the paused context.

### Where it lives in our app
File: `main.py` → `_handle_interrupt()` (`update_state` + `ainvoke(None, ...)`)
Tests: `tests/unit/test_hitl.py`

---

## Concept 60: Approval Workflows  [Module 7]
**Date:** 2026-06-25 | **Complexity:** Intermediate | **Layer:** Service

### One-line summary
A single-step HITL gate for refunds ≥ a configurable threshold: `billing_node` builds a typed approval payload, the graph pauses, a human approves/rejects/escalates, `approval_check_node` records the outcome.

### Production standard applied
Approval payload is a typed structure built by pure helpers (`_extract_refund_amount`, `check_approval_required`, `build_approval_payload`), so the delivery layer can render it without reaching into business logic. Only one approval step exists — multi-step approval chains (curriculum Concepts 61/67) are not implemented, this is a single-gate design.

### When to reach for it
- ✅ Any action with real-world cost (refunds, account deletion, compliance actions) above a risk threshold
- ❌ Low-risk, reversible actions — HITL adds latency and human load for no real safety gain

### The canonical pattern
```python
# workflows/approval.py
def check_approval_required(amount: float, threshold: float = 500.0) -> bool:
    return amount >= threshold

def build_approval_payload(amount: float, state: AgentState) -> dict:
    return {
        "action": "refund",
        "amount": amount,
        "ticket_type": "billing",
        "thread_id": state.get("thread_id"),
        "reason": f"Refund of ${amount:.2f} exceeds auto-approval threshold",
    }
```

### The most common mistake
Hardcoding the approval threshold inline in `billing_node` instead of reading it from `Settings` — makes it impossible to tune per-tenant or per-environment without a code change.

### Where it lives in our app
File: `workflows/approval.py` → `_extract_refund_amount()`, `check_approval_required()`, `build_approval_payload()`; `agents/nodes.py` → `billing_node()`, `approval_check_node()`
Tests: `tests/unit/test_hitl.py::TestApprovalHelpers`

---

## Concept 62: Escalation Flows  [Module 7]
**Date:** 2026-06-25 | **Complexity:** Intermediate | **Layer:** Service

### One-line summary
Tickets escalate through three levels (L1 → L2 → L3) via `escalate_node`, which bumps `escalation_level`, writes an audit event, and adapts the response tone per level. Routing to it (a form of agent handoff) is done by the pure `should_escalate()` router, not a `Command(goto=...)` object.

### Production standard applied
`EscalationLevel` is a class of named integer constants (not a bare `int`), and `escalate_node` caps at L3 — escalation is monotonic and bounded, never silently overflows. Handoff logic stays data-driven (`RouterFn`: `(AgentState) -> str`) rather than imperative, keeping it unit-testable without a compiled graph.

### When to reach for it
- ✅ Triggered from `should_escalate`: an explicit human "escalated" decision, an unrecoverable node error, or an SLA breach
- ❌ Don't escalate automatically on every retry — reserve it for genuinely stuck or high-risk tickets

### The canonical pattern
```python
# workflows/escalation.py
class EscalationLevel:
    L1_FRONTLINE = 0
    L2_SENIOR = 1
    L3_ENGINEERING = 2

def should_escalate(state: AgentState) -> str:
    if state.get("approval_decision") == "escalated" or state.get("error"):
        return "escalate"
    return "respond"

async def escalate_node(state: AgentState, config: RunnableConfig) -> dict:
    new_level = min(state["escalation_level"] + 1, EscalationLevel.L3_ENGINEERING)
    message = _build_escalation_message(new_level, state)
    return {
        "escalation_level": new_level,
        "audit_events": [create_audit_event("escalate", "escalated", state["thread_id"])],
        "final_response": message,
    }
```

### The most common mistake
Letting `escalation_level` increment past L3 — without the `min()` clamp, a ticket that keeps failing could report a nonsensical escalation tier that no downstream code (or human) knows how to handle.

### Where it lives in our app
File: `workflows/escalation.py` → `EscalationLevel`, `should_escalate()`, `escalate_node()`
Tests: `tests/unit/test_workflows.py::TestShouldEscalate, TestEscalateNode`

---

## Concept 65: Workflow State Machines  [Module 7]
**Date:** 2026-06-25 | **Complexity:** Beginner | **Layer:** Service

### One-line summary
`TicketStatus` is a `str`-subclassed `Enum` modeling the ticket lifecycle (`pending → in_progress → awaiting_approval → approved|rejected|escalated → resolved`) so state transitions are named, not magic strings.

### Production standard applied
Subclassing `str` alongside `Enum` means the value serializes as a plain JSON string (`"pending"`) in SQLite checkpoints — no custom encoder needed, and it still compares safely with `==`.

### When to reach for it
- ✅ Any field with a fixed, known set of lifecycle stages that other code branches on
- ❌ Don't use a plain `Literal` if you also want enum methods/iteration (`list(TicketStatus)`); don't use a bare class of string constants if you want type-checker exhaustiveness

### The canonical pattern
```python
# state.py
class TicketStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    ESCALATED = "escalated"
    RESOLVED = "resolved"
```

### The most common mistake
Defining the enum but never actually threading `state["ticket_status"]` through the graph nodes — it's currently a typed field with no writer wired into the main topology yet; don't assume it reflects the live ticket state until a node sets it.

### Where it lives in our app
File: `state.py` → `TicketStatus`
Tests: `tests/unit/test_workflows.py::TestTicketStatus`

---

## Concept 69: SLA Workflows  [Module 7]
**Date:** 2026-06-25 | **Complexity:** Intermediate | **Layer:** Service

### One-line summary
`sla_tracking_node` computes a per-priority deadline (critical=1h, high=4h, medium=24h, low=72h) and flags breaches, using pure helper functions that are trivial to unit test.

### Production standard applied
All the actual math (`compute_sla_deadline`, `is_sla_breached`, `hours_until_deadline`) is pure and synchronous — no I/O, no LLM calls — so the node itself is a thin wrapper that just calls helpers and writes an audit event.

### When to reach for it
- ✅ Any workflow with contractual or policy response-time guarantees
- ❌ Don't treat this as live monitoring — the breach check is point-in-time (evaluated only when the node runs), not a background timer

### The canonical pattern
```python
# workflows/sla.py — pure, unit-testable helpers
def compute_sla_deadline(created_at: str, priority: str) -> str:
    hours = {"critical": 1, "high": 4, "medium": 24, "low": 72}[priority]
    return (datetime.fromisoformat(created_at) + timedelta(hours=hours)).isoformat()

def is_sla_breached(deadline_str: str) -> bool:
    return datetime.now(timezone.utc) > datetime.fromisoformat(deadline_str)
```

### The most common mistake
Assuming an SLA breach auto-escalates the ticket — currently `sla_tracking_node` only logs and writes an audit event; it does not route into `escalate_node`. Wire that connection explicitly if the policy requires it.

### Where it lives in our app
File: `workflows/sla.py` → `compute_sla_deadline()`, `is_sla_breached()`, `hours_until_deadline()`, `sla_tracking_node()`
Tests: `tests/unit/test_workflows.py::TestSLAFunctions`

---

## Concept 70: Audit Trails  [Module 7]
**Date:** 2026-06-25 | **Complexity:** Beginner | **Layer:** Service

### One-line summary
`audit_events` is an append-only log (`Annotated[list[dict], operator.add]`) — every sensitive action (approval, escalation, PII detection) adds an entry; nothing is ever overwritten or deleted.

### Production standard applied
Reducer choice **is** the governance guarantee here: because the field uses `operator.add`, a node physically cannot erase history even by returning the wrong dict — the worst it can do is fail to append.

### When to reach for it
- ✅ Anything a compliance officer, auditor, or incident responder might need to reconstruct later
- ❌ Don't put high-volume debug info here — use `logger.debug()` for that; audit events are for decisions, not diagnostics

### The canonical pattern
```python
# observability/audit.py
class AuditEvent(TypedDict, total=False):
    node: str
    action: str
    thread_id: str
    timestamp: str

def create_audit_event(node: str, action: str, thread_id: str, **extra) -> AuditEvent:
    return {"node": node, "action": action, "thread_id": thread_id,
            "timestamp": datetime.now(timezone.utc).isoformat(), **extra}

# Nodes always APPEND, never replace:
return {"audit_events": [create_audit_event("billing", "refund_approved", thread_id, amount=amount)]}
```

### The most common mistake
Writing `return {"audit_events": [...]}` thinking it "sets" the log — with the `operator.add` reducer this is correct (it appends the one new event), but the mistake shows up if someone changes the reducer to a plain replace-on-write later, silently deleting all prior history. Never remove `operator.add` from this field.

### Where it lives in our app
File: `observability/audit.py` → `AuditEvent`, `create_audit_event()`, `format_audit_trail()`, `audit_trail_node()`
Tests: `tests/unit/test_workflows.py::TestAuditHelpers`

---

## Concept 71: Compliance Pipelines  [Module 7]
**Date:** 2026-06-25 | **Complexity:** Intermediate | **Layer:** Service

### One-line summary
`compliance_gate_node` regex-scans incoming ticket text for PII (email, phone, SSN, credit card, IP address) and flags it in the audit trail — it detects and logs, it does not block.

### Production standard applied
"Detect-and-flag, not detect-and-block" is a deliberate design choice: blocking requires understanding intent (a user *should* be able to give a callback number), so the pipeline defers the actual decision to human review via the audit trail, rather than guessing.

### When to reach for it
- ✅ Any user-facing text entering a system that has downstream compliance obligations (logging, LLM context, third-party APIs)
- ❌ Don't rely on regex PII detection alone for hard compliance guarantees (e.g. HIPAA/PCI) — pair it with a proper DLP tool; this is a fast, local, first-pass flag

### The canonical pattern
```python
# security/compliance.py
_PII_PATTERNS = {
    "email": r"[\w.+-]+@[\w-]+\.[\w.-]+",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b(?:\d[ -]*?){13,16}\b",
    # ... phone, ip_address
}

def detect_pii(text: str) -> list[str]:
    return [kind for kind, pattern in _PII_PATTERNS.items() if re.search(pattern, text)]
```

### The most common mistake
Treating a PII hit as an error to raise — `compliance_gate_node` intentionally returns a partial state update (`{"audit_events": [...]}`) rather than raising, so the ticket keeps flowing while the flag is preserved for review.

### Where it lives in our app
File: `security/compliance.py` → `detect_pii()`, `_redact_pii()`, `compliance_gate_node()`
Tests: `tests/unit/test_workflows.py::TestPIIDetection`

---

## Concept 73: Subgraphs  [Module 8]
**Date:** 2026-06-25 | **Complexity:** Advanced | **Layer:** Graph

### One-line summary
A compiled `StateGraph` can itself be treated as a black-box unit inside a larger graph — this app's CRAG pipeline (Module 9) is built this way. Status: `get_rag_graph()` is implemented and tested but not yet called by the running app — `kb_worker_node` currently calls `make_retriever_node()` directly.

### Production standard applied
Subgraph boundaries are explicit: the caller only needs to know the input it must provide (`query`) and the outputs it gets back (`retrieved_docs`, `final_response`) — internal CRAG nodes (`rewrite_query`, `grade_retrieval`, etc.) are fully encapsulated.

### When to reach for it
- ✅ A self-contained multi-step pipeline (retrieval + grading + generation) that's reused from more than one call site
- ❌ Don't build a subgraph for a single linear sequence of 2-3 nodes — a plain node chain is simpler

### The canonical pattern
```python
# rag/graph.py — subgraph is just a cached, compiled StateGraph
@lru_cache(maxsize=1)
def get_rag_graph():
    return (GraphBuilder(AgentState)
            .node("retrieve", get_retriever_node())
            .node("grade_retrieval", retrieval_grade_node)
            .node("generate", generate_node)
            .node("hallucination_check", hallucination_check_node)
            .conditional_edge("grade_retrieval", route_by_grade, {...})
            .entry("retrieve")
            .build())

# Intended call shape (not wired into kb_worker_node yet):
kb_result = await get_rag_graph().ainvoke({"query": query}, config=config)
```

### The most common mistake
Assuming a subgraph gets its own checkpointer — it shares the parent graph's checkpoint context, so double-compiling it with a different checkpointer configuration silently does nothing useful.

### Where it lives in our app
File: `rag/graph.py` → `get_rag_graph()`
Tests: `tests/rag/test_rag_pipeline.py`

---

## Concept 74: Supervisor Pattern  [Module 8]
**Date:** 2026-06-25 | **Complexity:** Advanced | **Layer:** Graph

### One-line summary
`supervisor_node` doesn't know about `BillingAgent` or `ResearchAgent` by name — it resolves `get_agent(state["ticket_type"])` from a registry and calls `.process()`, so adding a new specialist never touches the supervisor.

### Production standard applied
Open/Closed Principle in action: `agents/registry.py` maps names to factories via `@register_agent(name)`; the supervisor is closed for modification but open for extension by anyone adding a new `@register_agent("legal")` elsewhere.

### When to reach for it
- ✅ More than 2-3 specialist behaviors that share a common interface and need central dispatch
- ❌ A single linear flow with no branching specialization — plain conditional edges are simpler

### The canonical pattern
```python
# agents/supervisor.py
async def supervisor_node(state: AgentState, config: RunnableConfig) -> dict:
    agent = get_agent(state["ticket_type"])       # resolved from registry, not imported by name
    return await agent.process(state, config)
```

### The most common mistake
Forgetting to import a new agent module before `build_graph()` runs — `@register_agent` only fires when the module is imported, so a specialist that's never imported is invisible to the registry and `get_agent()` raises `KeyError` at runtime, not at import time.

### Where it lives in our app
File: `agents/supervisor.py` → `supervisor_node()`; `agents/registry.py` → `register_agent()`, `get_agent()`
Tests: `tests/unit/test_multi_agent.py::TestSupervisorNode`

---

## Concept 75: Worker Agents  [Module 8]
**Date:** 2026-06-25 | **Complexity:** Intermediate | **Layer:** Graph

### One-line summary
`BaseAgent` is the abstract contract (`process()`, `get_tools()`) every agent implements, letting the supervisor treat any registered agent interchangeably (Liskov substitution) regardless of what it actually does.

### Production standard applied
Interface Segregation — `BaseAgent` only demands the two methods every caller actually needs (`process`, `get_tools`); it does not force agents to implement retrieval, memory, or search methods they may not use.

### When to reach for it
- ✅ Whenever the supervisor needs to call into a specialist without a per-type `if/elif` chain
- ❌ Don't add a new abstract method to `BaseAgent` unless every current and near-future agent needs it — that's how fat interfaces creep in

### The canonical pattern
```python
# agents/base.py
class BaseAgent(ABC):
    @abstractmethod
    async def process(self, state: AgentState, config: RunnableConfig) -> dict: ...
    @abstractmethod
    def get_tools(self) -> list: ...
```

### The most common mistake
Making `process()` synchronous "because this agent doesn't need to await anything" — every agent must be `async def process(...)` to satisfy the shared contract, even a trivial one that just returns a dict; the supervisor always `await`s it.

### Where it lives in our app
File: `agents/base.py` → `BaseAgent`
Tests: `tests/unit/test_multi_agent.py::TestBaseAgentContract`

---

## Concept 76: Specialist Agents  [Module 8]
**Date:** 2026-06-25 | **Complexity:** Intermediate | **Layer:** Graph

### One-line summary
`BillingAgent`, `TechnicalAgent`, `GeneralAgent`, and `ResearchAgent` each extend `BaseAgent` with domain-specific logic only — none of them do their own expensive I/O.

### Production standard applied
Single Responsibility per agent: `BillingAgent` only knows about refund detection, `ResearchAgent` only knows how to extract a search query and signal fan-out — neither does retrieval or web search itself, keeping each agent trivially unit-testable with a fake state.

### When to reach for it
- ✅ Distinct domains of ticket handling that need different logic but a uniform calling convention
- ❌ Don't create a new agent subclass for a one-line behavioral difference — a config flag on an existing agent is simpler

### The canonical pattern
```python
# agents/research_agent.py
class ResearchAgent(BaseAgent):
    async def process(self, state, config):
        query = _extract_query(state["messages"][-1].content)
        return {"query": query, "active_agent": "research"}  # signals fan-out to supervisor router
    def get_tools(self):
        return []   # research agent delegates tool use to kb_worker/web_worker
```

### The most common mistake
Putting expensive I/O (LLM calls, retrieval) directly inside `agent.process()` for research-style agents — `ResearchAgent` deliberately only extracts the query and signals fan-out; the actual work happens in `kb_worker_node`/`web_worker_node`, which are dispatched in parallel via `Send` (Concept 84).

### Where it lives in our app
File: `agents/billing_agent.py`, `agents/technical_agent.py`, `agents/general_agent.py`, `agents/research_agent.py`
Tests: `tests/unit/test_multi_agent.py::TestBillingAgent, TestTechnicalAgent, TestGeneralAgent, TestResearchAgent`

---

## Concept 82: Agent Coordination  [Module 8]
**Date:** 2026-06-25 | **Complexity:** Intermediate | **Layer:** Graph

### One-line summary
Agents never call each other directly — they coordinate exclusively through shared `AgentState`, most visibly via `agent_results: Annotated[list[dict], operator.add]`, which both parallel workers append to independently.

### Production standard applied
The `operator.add` reducer on `agent_results` is what makes fan-out coordination safe: LangGraph merges each worker's partial-dict return by concatenating lists, so `kb_worker_node` and `web_worker_node` never need to know about each other or coordinate write order.

### When to reach for it
- ✅ Any time two or more nodes/agents produce results that must be combined without one overwriting the other
- ❌ Don't use `operator.add` for scalar fields two nodes might both want to "win" on (e.g. `ticket_type`) — that needs a plain replace or a custom reducer with defined precedence

### The canonical pattern
```python
# state.py
agent_results: Annotated[list[dict], operator.add]

# agents/supervisor.py — both workers append independently; LangGraph merges automatically
async def kb_worker_node(state, config) -> dict:
    return {"agent_results": [{"source": "kb", "docs": docs}]}

async def web_worker_node(state, config) -> dict:
    return {"agent_results": [{"source": "web", "results": results}]}
```

### The most common mistake
Assuming ordering between `agent_results` entries — fan-out workers run concurrently with no guaranteed completion order, so `research_collect_node` must treat the list as unordered and identify entries by their `source` field, not by position.

### Where it lives in our app
File: `state.py` → `agent_results` field; `agents/supervisor.py` → `kb_worker_node()`, `web_worker_node()`
Tests: `tests/unit/test_multi_agent.py::TestKbWorkerNode, TestWebWorkerNode`

---

## Concept 84: Send API  [Module 8]
**Date:** 2026-06-25 | **Complexity:** Advanced | **Layer:** Graph

### One-line summary
`route_after_supervisor` returns a `list[Send(...)]` instead of a single node name, telling LangGraph to dispatch to `kb_worker` and `web_worker` with per-branch state overrides in one call.

### Production standard applied
`Send("kb_worker", {"query": query})` lets a router pass a targeted state override without needing separate conditional edges for each destination — the routing decision and the payload travel together.

### When to reach for it
- ✅ Dispatching to N destinations decided at runtime (not statically known at compile time)
- ❌ Don't reach for `Send` when a single static conditional edge (`add_conditional_edges`) would do — it's for the *multiple simultaneous destinations* case specifically

### The canonical pattern
```python
# agents/supervisor.py
def route_after_supervisor(state: AgentState) -> str | list[Send]:
    if state.get("active_agent") == "research":
        query = state["query"]
        return [Send("kb_worker", {"query": query}), Send("web_worker", {"query": query})]
    return "respond"
```

### The most common mistake
Passing a node name to `Send` that doesn't exist in the compiled graph — `Send` destinations are resolved at runtime, so a typo (`"kb_worrker"`) fails with an opaque error deep in LangGraph's scheduler rather than at `build_graph()` time. Double-check names match `builder.node(...)` calls exactly.

### Where it lives in our app
File: `agents/supervisor.py` → `route_after_supervisor()`
Tests: `tests/unit/test_multi_agent.py::TestRouteAfterSupervisor`

---

## Concept 85: Fan-Out  [Module 8]
**Date:** 2026-06-25 | **Complexity:** Advanced | **Layer:** Graph

### One-line summary
`kb_worker` and `web_worker` execute concurrently once `Send` schedules them — LangGraph runs both branches before merging state and continuing.

### Production standard applied
`@with_retry` is mandatory on every fanned-out worker (per the node standard) precisely because parallel branches can't easily be re-run individually after a partial fan-out failure — each worker must be independently resilient to transient errors.

### When to reach for it
- ✅ Independent sub-tasks that can run concurrently and don't depend on each other's output
- ⚠️ On Groq's free tier, concurrent fan-out to multiple LLM-calling workers can trip rate limits — this app avoids the issue because only `web_worker` calls an external API (Tavily); `kb_worker` uses local retrieval. If both workers start calling Groq concurrently (e.g. after wiring `kb_worker` into full CRAG), add jitter/staggering.

### The canonical pattern
```python
# agents/supervisor.py
@with_retry
async def kb_worker_node(state: AgentState, config: RunnableConfig) -> dict:
    docs = await make_retriever_node()(state, config)
    return {"agent_results": [{"source": "kb", **docs}]}

@with_retry
async def web_worker_node(state: AgentState, config: RunnableConfig) -> dict:
    results = await search_tool.ainvoke(state["query"])
    return {"agent_results": [{"source": "web", "results": results}]}
```

### The most common mistake
Assuming "parallel" means "async" — fan-out is LangGraph's scheduling behavior (it runs both branches before the next shared node), not a guarantee about how each individual node is implemented. A synchronous, blocking node dropped into a fan-out branch still blocks the event loop during its turn.

### Where it lives in our app
File: `agents/supervisor.py` → `kb_worker_node()`, `web_worker_node()`
Tests: `tests/unit/test_multi_agent.py::TestKbWorkerNode, TestWebWorkerNode`

---

## Concept 86: Fan-In  [Module 8]
**Date:** 2026-06-25 | **Complexity:** Advanced | **Layer:** Graph

### One-line summary
`research_collect_node` is the "reduce" step: it only runs once both fanned-out workers finish, and reads their combined output from `agent_results` (already merged by the `operator.add` reducer).

### Production standard applied
LangGraph guarantees all fan-out branches complete and their state updates merge *before* the next shared node runs — `research_collect_node` never has to poll or wait manually, it just reads the fully-merged state.

### When to reach for it
- ✅ Whenever fan-out (Concept 85) is used — every fan-out needs a corresponding fan-in to aggregate, or the parallel results are stranded in state with nothing consuming them
- ❌ Don't add a fan-in node if you only fanned out for side effects (e.g. two independent audit writes with no combined result to produce)

### The canonical pattern
```python
# agents/supervisor.py
async def research_collect_node(state: AgentState, config: RunnableConfig) -> dict:
    results = state.get("agent_results", [])   # already contains both kb + web entries
    summary = _summarize_results(results)
    return {"final_response": summary,
            "audit_events": [create_audit_event("research_collect", "fan_in_complete", state["thread_id"], count=len(results))]}
```

### The most common mistake
Calling this "reduce" like Python's `functools.reduce` — it's really just aggregation over an already-concatenated list, not a fold with an accumulator function. Don't over-engineer it into a generic reducer abstraction the app doesn't need.

### Where it lives in our app
File: `agents/supervisor.py` → `research_collect_node()`
Tests: `tests/unit/test_multi_agent.py::TestResearchCollectNode`

---

## Concept 89: RAG Subgraphs  [Module 9]
**Date:** 2026-06-26 | **Complexity:** Advanced | **Layer:** Graph

### One-line summary
`get_rag_graph()` compiles the full CRAG (Corrective RAG) pipeline as its own `StateGraph`, exposed through `rag/__init__.py`'s public API so callers never import internal CRAG node functions directly.

### Production standard applied
Module boundary rule: `rag/__init__.py` declares `__all__ = ["get_rag_graph", "get_retriever_node", "build_index"]` — everything else in `rag/` (`grader.py`, `graph.py` internals) is implementation detail callers must not reach into.

### When to reach for it
- ✅ Any RAG usage in the app — always go through `get_rag_graph()` or `get_retriever_node()`, never construct `FAISSVectorStore` or call graders directly from agent code
- ❌ Don't import `from rag.graph import generate_node` from outside the `rag` package — that breaks the encapsulation the subgraph boundary is meant to provide

### The canonical pattern
```python
# rag/__init__.py — the only sanctioned entry points
from rag.graph import get_rag_graph
from rag.retriever import get_retriever_node
from rag.ingest import build_index
__all__ = ["get_rag_graph", "get_retriever_node", "build_index"]
```

### The most common mistake
Reaching into `rag.graph` or `rag.grader` submodules from `agents/` code "just this once" — it works today but locks the internal RAG node signatures in place; any refactor of CRAG internals now has to hunt down every improper import.

### Where it lives in our app
File: `rag/__init__.py`, `rag/graph.py` → `get_rag_graph()`
Tests: `tests/rag/test_rag_pipeline.py`

---

## Concept 90: Retrieval Nodes  [Module 9]
**Date:** 2026-06-26 | **Complexity:** Intermediate | **Layer:** Service

### One-line summary
`make_retriever_node()` is a DI factory returning an async node that does two-stage retrieval: FAISS similarity search (top 20) then cross-encoder reranking (top 5) — with store/reranker injectable for tests.

### Production standard applied
Dependency Injection via factory: production calls `make_retriever_node()` with no args and gets real FAISS + real reranker; tests call it with mock store/reranker directly, no monkeypatching needed.

### When to reach for it
- ✅ Any node that wraps an external or heavy dependency (vector store, model) that tests need to swap out
- ❌ Don't hardcode `FAISSVectorStore(...)` inside the node body — that's exactly the hard-dependency anti-pattern DI avoids

### The canonical pattern
```python
# rag/retriever.py
def make_retriever_node(store=None, reranker=None):
    _store = store or get_vector_store()
    _reranker = reranker or get_reranker()

    async def retriever_node(state: AgentState, config: RunnableConfig) -> dict:
        docs = _store.similarity_search(state["query"], k=settings.rag_top_k_retrieve)
        ranked = _rerank(state["query"], docs, _reranker)[: settings.rag_top_k_rerank]
        return {"retrieved_docs": ranked}
    return retriever_node
```

### The most common mistake
Assuming retrieval failure raises — if no index is built yet, `FAISSVectorStore` returns an empty list rather than raising, so the node degrades gracefully to `retrieved_docs: []` (which CRAG's grader then correctly marks "irrelevant"). Don't wrap this call in a try/except expecting an exception that won't come.

### Where it lives in our app
File: `rag/retriever.py` → `make_retriever_node()`
Tests: `tests/rag/test_rag_pipeline.py::TestRetrieverNode`

---

## Concept 91: Query Rewriting  [Module 9]
**Date:** 2026-06-26 | **Complexity:** Intermediate | **Layer:** Graph

### One-line summary
When `grade_retrieval` marks results "irrelevant," `rewrite_query_node` asks the LLM to rephrase the query, then loops back to `retrieve` — the "corrective" half of Corrective RAG.

### Production standard applied
The rewrite is isolated in its own node (not folded into `retrieve` or `generate`) so the CRAG loop edge (`rewrite_query → retrieve`) is a plain graph edge, not a hidden retry loop buried in a single function.

### When to reach for it
- ✅ Retrieval quality varies with query phrasing (common with short or ambiguous user text)
- ⚠️ There is currently no max-iteration guard — a query that never improves could loop `rewrite → retrieve → grade` indefinitely. Treat this as a known gap, not an implemented safeguard.

### The canonical pattern
```python
# rag/graph.py
async def rewrite_query_node(state: AgentState, config: RunnableConfig) -> dict:
    llm = get_fast_llm()
    prompt = f"Rewrite this search query to be more specific and retrievable: {state['query']}"
    try:
        response = await llm.ainvoke(prompt, config=config)
        return {"query": response.content.strip()}
    except ProviderError:
        logger.warning("query rewrite failed, keeping original query=%r", state["query"])
        return {"query": state["query"]}   # fall back, don't crash the loop
```

### The most common mistake
Not capping CRAG's rewrite→retrieve loop — before extending this pipeline, add a loop counter to `state` and route to a fallback (e.g. "answer with best-effort context") after N rewrites, or a bad query can burn the `graph_recursion_limit` and hard-fail instead of degrading gracefully.

### Where it lives in our app
File: `rag/graph.py` → `rewrite_query_node()`, `route_by_grade()`
Tests: `tests/rag/test_rag_pipeline.py::TestCRAGNodes`

---

## Concept 93: Retrieval Graders  [Module 9]
**Date:** 2026-06-26 | **Complexity:** Intermediate | **Layer:** Service

### One-line summary
`grade_document` and `grade_retrieval` are pure functions — `(query, docs) -> "relevant" | "irrelevant" | "partial"` — called from `retrieval_grade_node`, not nodes themselves.

### Production standard applied
Keeping graders as pure functions (no `state`, no `config`) makes them trivially unit-testable with plain strings and mock LLM responses — the fixed bug below would have been caught immediately by a grader-level unit test.

### When to reach for it
- ✅ Any LLM-based classification step that a node needs but that doesn't itself need graph context
- ❌ Don't make a grader a full node if nothing else in the graph calls it standalone — keep it a helper the node imports

### The canonical pattern
```python
# rag/grader.py
def grade_document(query: str, doc_content: str, llm) -> str:
    response = llm.invoke(f"Is this document relevant to '{query}'? Answer: relevant/irrelevant/partial\n\n{doc_content}")
    text = response.content.strip().lower()
    if "irrelevant" in text:      # MUST check before "relevant" — see mistake below
        return "irrelevant"
    if "partial" in text:
        return "partial"
    return "relevant"
```

### The most common mistake
Checking `"relevant" in text` before `"irrelevant" in text` — since `"irrelevant"` contains `"relevant"` as a substring, that ordering misclassifies every irrelevant document as relevant. This was a real bug caught during Module 9 development; the fix is checking the more specific string first.

### Where it lives in our app
File: `rag/grader.py` → `grade_document()`, `grade_retrieval()`; `rag/graph.py` → `retrieval_grade_node()`
Tests: `tests/rag/test_rag_pipeline.py::TestGraders`

---

## Concept 94: Hallucination Graders  [Module 9]
**Date:** 2026-06-26 | **Complexity:** Intermediate | **Layer:** Service

### One-line summary
`check_hallucination(answer, docs, llm)` is a pure function returning `True` if the generated answer is grounded in the retrieved documents, `False` if it appears fabricated — called by `hallucination_check_node` as the last CRAG step.

### Production standard applied
Same pure-function-behind-a-node pattern as the retrieval grader (Concept 93) — grounding checks are business logic, not graph orchestration, so they're unit-tested with plain strings, no compiled graph required.

### When to reach for it
- ✅ Any generation step where factual grounding matters (support answers referencing a KB, not free-form chat)
- ⚠️ Currently a dead-end check: if `hallucination_check` is `False`, the graph still ends — there's no regeneration loop back to `generate`. That's Self-RAG (a curriculum concept this app doesn't implement).

### The canonical pattern
```python
# rag/grader.py
def check_hallucination(answer: str, docs: list[RetrievedDocument], llm) -> bool:
    context = "\n".join(d["content"] for d in docs)
    response = llm.invoke(f"Is this answer fully supported by the context? yes/no\nContext: {context}\nAnswer: {answer}")
    return response.content.strip().lower().startswith("yes")
```

### The most common mistake
Assuming `hallucination_check: False` triggers a retry — it doesn't in this app. If you need "regenerate on hallucination," that's an explicit extension (loop `hallucination_check → generate` with a counter), not something CRAG gives you for free.

### Where it lives in our app
File: `rag/grader.py` → `check_hallucination()`; `rag/graph.py` → `hallucination_check_node()`
Tests: `tests/rag/test_rag_pipeline.py::TestGraders`

---

## Concept 95: CRAG (Corrective RAG)  [Module 9]
**Date:** 2026-06-26 | **Complexity:** Advanced | **Layer:** Graph

### One-line summary
The full pipeline: `retrieve → grade_retrieval → [generate | rewrite_query → retrieve (loop)] → generate → hallucination_check → END` — retrieval quality is corrected before generation, and generation is checked for grounding after.

### Production standard applied
`GraphBuilder` assembles CRAG the same way the main app graph is built — conditional edges (`route_by_grade`) drive the corrective loop, keeping the topology declarative and readable in one place (`get_rag_graph()`).

### When to reach for it
- ✅ Knowledge-base Q&A where retrieval quality is variable and a cheap corrective step (query rewrite) meaningfully improves results
- ❌ Don't reach for full CRAG if your corpus is small/high-quality enough that plain retrieve→generate rarely misses — the extra LLM calls (grading, rewriting) add latency and token cost

### The canonical pattern
```python
# rag/graph.py
def get_rag_graph():
    return (GraphBuilder(AgentState)
        .node("retrieve", get_retriever_node())
        .node("grade_retrieval", retrieval_grade_node)
        .node("rewrite_query", rewrite_query_node)
        .node("generate", generate_node)
        .node("hallucination_check", hallucination_check_node)
        .entry("retrieve")
        .edge("retrieve", "grade_retrieval")
        .conditional_edge("grade_retrieval", route_by_grade, {"generate": "generate", "rewrite_query": "rewrite_query"})
        .edge("rewrite_query", "retrieve")
        .edge("generate", "hallucination_check")
        .build())
```

### The most common mistake
Confusing CRAG's corrective loop (fixes *retrieval*, before generation) with Self-RAG's regeneration loop (fixes *generation*, after the fact) — this app only implements the former. See Concept 94's gotcha.

### Where it lives in our app
File: `rag/graph.py` → `get_rag_graph()`
Tests: `tests/rag/test_rag_pipeline.py::TestCRAGNodes`

---

## Concept 99: Document Ingestion Pipeline  [Module 9]
**Date:** 2026-06-26 | **Complexity:** Intermediate | **Layer:** Service

### One-line summary
`build_index()` loads KB markdown files, splits them via a `ChunkingStrategy`, embeds them, and persists a FAISS index to disk — run offline as `python -m rag.ingest`, not on the request path.

### Production standard applied
Strategy pattern for chunking: `ChunkingStrategy` is a `Protocol` with `FixedSizeChunker` (default, fast) as the concrete implementation in use; swapping to `SemanticChunker` (requires `langchain_experimental`) needs zero changes to `build_index()` itself.

### When to reach for it
- ✅ Whenever the KB source documents change — re-run ingestion to rebuild the index
- ❌ Never call `build_index()` from inside a graph node — it's a batch/offline operation, not part of request latency

### The canonical pattern
```python
# rag/ingest.py
def build_index(data_dir: str = "data/kb", chunker: ChunkingStrategy | None = None) -> None:
    docs = load_kb_documents(data_dir)
    chunks = (chunker or FixedSizeChunker()).split(docs)
    store = FAISSVectorStore(get_embeddings())
    store.add_documents(chunks)
    store.save(settings.rag_index_path)

if __name__ == "__main__":
    build_index()   # run as: python -m rag.ingest
```

### The most common mistake
Forgetting to run ingestion before the first graph invocation — `get_vector_store()` will happily return a store with no index loaded, and retrieval silently degrades to empty results + "irrelevant" grade rather than erroring loudly. Always run `python -m rag.ingest` after adding/editing KB docs.

### Where it lives in our app
File: `rag/ingest.py` → `load_kb_documents()`, `build_index()`; `rag/chunking/strategies.py` → `FixedSizeChunker`, `SemanticChunker`
Tests: `tests/rag/test_rag_pipeline.py::TestBuildIndex, TestLoadKBDocuments`

---

## Concept 100: Vector Stores  [Module 9]
**Date:** 2026-06-26 | **Complexity:** Intermediate | **Layer:** Infrastructure

### One-line summary
`FAISSVectorStore` implements `VectorStoreProtocol` (`similarity_search`, `add_documents`, `save`/`load`) — any code depending on the protocol works unchanged if FAISS is later swapped for `ChromaVectorStore`.

### Production standard applied
Liskov Substitution via a structural `Protocol`, not an ABC — `FAISSVectorStore` doesn't need to inherit from anything, it just needs to match the method signatures, which keeps the infrastructure layer decoupled from the protocol definition.

### When to reach for it
- ✅ Any place that needs "a vector store" without caring which backend — always type-hint against `VectorStoreProtocol`, never `FAISSVectorStore` directly, outside the factory itself
- ❌ Don't instantiate `FAISSVectorStore(...)` directly in node/agent code — go through `get_vector_store()` (the factory pattern)

### The canonical pattern
```python
# rag/protocols.py
class VectorStoreProtocol(Protocol):
    def similarity_search(self, query: str, k: int) -> list[Document]: ...
    def add_documents(self, docs: list[Document]) -> None: ...
    def save(self, path: str) -> None: ...

# core/factories.py
@lru_cache(maxsize=1)
def get_vector_store() -> VectorStoreProtocol:
    return FAISSVectorStore.load(get_settings().rag_index_path, get_embeddings())
```

### The most common mistake
Calling `similarity_search()` before any documents have been added or loaded — `FAISSVectorStore` returns an empty list rather than raising (a deliberate design choice so retrieval nodes degrade gracefully), so a "no results ever" bug can silently be "forgot to run ingestion," not a retrieval logic bug.

### Where it lives in our app
File: `rag/stores/faiss_store.py` → `FAISSVectorStore`; `rag/protocols.py` → `VectorStoreProtocol`; `core/factories.py` → `get_vector_store()`
Tests: `tests/rag/test_rag_pipeline.py::TestFAISSVectorStore`

---

## Concept 101: Embeddings  [Module 9]
**Date:** 2026-06-26 | **Complexity:** Beginner | **Layer:** Infrastructure

### One-line summary
`get_embeddings()` returns a cached `HuggingFaceEmbeddings` instance (`sentence-transformers/all-MiniLM-L6-v2`, local, free) — the only sanctioned way to get an embedder anywhere in the app.

### Production standard applied
Provider Adapter Pattern (the project's single most important rule): embeddings, like every model, are obtained exclusively through `llm_provider.py` — never `HuggingFaceEmbeddings(...)` instantiated directly in `rag/` or `memory/` code.

### When to reach for it
- ✅ RAG document embedding, long-term memory semantic search — anywhere text needs to become a vector
- ❌ Never construct the HF embeddings class directly, even "just for a quick test" — it bypasses the `lru_cache` singleton and settings-driven model selection

### The canonical pattern
```python
# llm_provider.py
@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=get_settings().hf_embeddings_model)
```

### The most common mistake
Instantiating a *different* embeddings model ad hoc for a one-off script — FAISS indexes are tied to the embedding dimensionality/model used at build time; querying with a mismatched embedder returns garbage similarity scores with no error.

### Where it lives in our app
File: `llm_provider.py` → `get_embeddings()`
Tests: `tests/rag/test_rag_pipeline.py` (mocked in fixtures)

---

## Concept 102: Reranking  [Module 9]
**Date:** 2026-06-26 | **Complexity:** Intermediate | **Layer:** Service

### One-line summary
Stage 2 of retrieval: a cross-encoder (`ms-marco-MiniLM`) re-scores the top-20 FAISS hits jointly with the query, cutting to the top 5 — more accurate than the bi-encoder similarity search alone, at extra latency cost.

### Production standard applied
Two-stage retrieval (`rag_top_k_retrieve=20` cheap bi-encoder pass, `rag_top_k_rerank=5` expensive cross-encoder pass) is a standard production RAG cost/accuracy tradeoff, tuned entirely through `Settings` fields rather than hardcoded constants.

### When to reach for it
- ✅ Whenever bi-encoder similarity search alone returns too many marginally-relevant results — cross-encoders score query+doc jointly and rank much more accurately
- ❌ Don't rerank a huge candidate set (e.g. k=200) — cross-encoders are O(n) model calls per document; sweet spot here is reranking ~20 down to ~5

### The canonical pattern
```python
# rag/retriever.py
def _rerank(query: str, docs: list[Document], reranker, top_k: int) -> list[Document]:
    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:top_k]]
```

### The most common mistake
Reranking with a large `k` "to be safe" — cross-encoder inference is noticeably slower per-document than the initial FAISS search; doubling `rag_top_k_retrieve` without reason doubles reranking latency for marginal recall gain.

### Where it lives in our app
File: `rag/retriever.py` → `_rerank()`; `llm_provider.py` → `get_reranker()`
Tests: `tests/rag/test_rag_pipeline.py::TestRetrieverNode`

---

## Concept 106: RAG Evaluation  [Module 9]
**Date:** 2026-06-26 | **Complexity:** Intermediate | **Layer:** Service

### One-line summary
`rag/evaluator.py` offers three offline metrics — retrieval recall, faithfulness (reuses `check_hallucination`), and answer relevance — run against a fixed test set via `run_eval_suite()`, never on the live request path.

### Production standard applied
Evaluation logic is pure and offline by construction: it takes a golden test set and a graph, and produces scores — no evaluation code lives inside the request-serving nodes, so scoring a new prompt/model doesn't touch production latency.

### When to reach for it
- ✅ Before/after any change to prompts, chunking strategy, embedding model, or reranker — regression-test RAG quality the same way you'd run a test suite
- ❌ Don't call these functions inside `generate_node` or any live node — they're for CI/offline runs (`tests/rag/`), not production

### The canonical pattern
```python
# rag/evaluator.py
def evaluate_retrieval_recall(retrieved: list[RetrievedDocument], expected_sources: set[str]) -> float:
    found = {d["source"] for d in retrieved} & expected_sources
    return len(found) / len(expected_sources) if expected_sources else 1.0

def evaluate_faithfulness(answer: str, docs: list[RetrievedDocument], llm) -> bool:
    return check_hallucination(answer, docs, llm)   # reuses the grader, doesn't reimplement it
```

### The most common mistake
Reimplementing hallucination detection inside the evaluator instead of calling `check_hallucination()` — that would let the production grader and the evaluation metric drift apart, so a regression in one wouldn't be caught by the other.

### Where it lives in our app
File: `rag/evaluator.py` → `evaluate_retrieval_recall()`, `evaluate_faithfulness()`, `evaluate_answer_relevance()`, `run_eval_suite()`
Tests: `tests/rag/test_rag_pipeline.py::TestEvaluator`

---

## Concept 109: Authentication  [Module 10]
**Date:** 2026-06-26 | **Complexity:** Intermediate | **Layer:** Security / Infrastructure

### One-line summary
`validate_jwt()` decodes and verifies a bearer token (`sub`, `tenant_id`, `roles`, `exp`); `build_auth_context()` wraps the result (or a permissive dev-mode default when no `JWT_SECRET` is configured) into a typed `AuthContext`.

### Production standard applied
Authentication is a pre-graph guard, not a graph node — real auth happens in the FastAPI dependency (`get_current_user`) before the graph is ever invoked, keeping "is this caller who they claim to be" out of business logic entirely.

### When to reach for it
- ✅ Every request entering `api/routes.py` — `get_current_user` runs as a FastAPI `Depends()`
- ❌ Don't validate JWTs inside graph nodes — by the time a node runs, the caller is already assumed authenticated; nodes should trust `config["configurable"]`, not re-derive identity

### The canonical pattern
```python
# security/auth.py
def build_auth_context(token: str | None, settings: Settings) -> AuthContext:
    if settings.jwt_secret is None:
        return {"user_id": "dev-user", "tenant_id": settings.default_tenant_id, "roles": ["agent", "billing_admin"]}
    payload = validate_jwt(token, settings.jwt_secret.get_secret_value(), settings.jwt_algorithm)
    return {"user_id": payload["sub"], "tenant_id": payload["tenant_id"], "roles": payload["roles"]}
```

### The most common mistake
Assuming dev-mode's permissive default (`no JWT_SECRET → open access`) is also safe for staging/prod — it's explicitly a local-dev convenience. Always set `JWT_SECRET` outside local development, and treat its absence as a `ConfigurationError` in any deployed environment.

### Where it lives in our app
File: `security/auth.py` → `validate_jwt()`, `build_auth_context()`, `create_jwt()`, `auth_node()`; `api/routes.py` → `get_current_user()`
Tests: `tests/security/test_security.py::TestJWT, TestAuthNode`

---

## Concept 110: Authorization  [Module 10]
**Date:** 2026-06-26 | **Complexity:** Intermediate | **Layer:** Service

### One-line summary
`core/exceptions.py`'s custom hierarchy (`SupportAgentError` base → `AuthenticationError`, `AuthorizationError`, `ProviderError`, `RetrievalError`, `ValidationError`, `ConfigurationError`) is what makes "authorized or not" a typed, catchable outcome instead of a string comparison.

### Production standard applied
Three-layer error handling: `@with_retry` for transient failures, catch specific `SupportAgentError` subclasses for expected business failures (e.g. `AuthorizationError` → return `{"error": ...}` for the router to act on), never a bare `except Exception`.

### When to reach for it
- ✅ Any place a permission/authorization decision needs to short-circuit normal flow with context (who, what action, what role was required)
- ❌ Don't raise a bare `Exception("not authorized")` — it loses the structured fields (`user_id`, `action`, `required_role`) that a caller or error-handling node needs to act intelligently

### The canonical pattern
```python
# core/exceptions.py
class SupportAgentError(Exception):
    """Base exception for all application errors. Never raise this directly."""

class AuthorizationError(SupportAgentError):
    def __init__(self, user_id: str, action: str, required_role: str):
        super().__init__(f"User '{user_id}' lacks role '{required_role}' for action '{action}'")
```

### The most common mistake
Catching `SupportAgentError` (the base class) everywhere instead of the specific subclass — that swallows `AuthorizationError` and `RetrievalError` identically, losing the ability to route each failure type differently (e.g. authorization failures should surface to the user; retrieval failures should trigger a fallback).

### Where it lives in our app
File: `core/exceptions.py` → `SupportAgentError`, `AuthorizationError`, and siblings
Tests: `tests/security/test_security.py::TestExceptionHierarchy`

---

## Concept 111: RBAC  [Module 10]
**Date:** 2026-06-26 | **Complexity:** Intermediate | **Layer:** Security / Service

### One-line summary
`ROLE_PERMISSIONS` is a policy-as-code dict mapping role → set of permitted actions; `has_permission()`/`require_permission()` check it, and `require_permission()` raises `AuthorizationError` (Concept 110) with full context on denial.

### Production standard applied
Policy lives in a plain Python dict, not a database table — every permission change is a git diff someone reviews, which is the whole governance benefit (see Concept 113) of keeping RBAC as code.

### When to reach for it
- ✅ Any action with real consequence gated by role — e.g. billing refund approval requires `refunds:approve`, held only by `billing_admin`
- ❌ Don't scatter role checks as ad hoc `if "admin" in roles` throughout node code — always go through `has_permission()`/`require_permission()` so the policy stays in one reviewable place

### The canonical pattern
```python
# security/rbac.py
ROLE_PERMISSIONS: dict[str, set[str]] = {
    "agent": {"tickets:read", "tickets:respond"},
    "billing_admin": {"tickets:read", "tickets:respond", "refunds:approve"},
    "readonly": {"tickets:read"},
}

def require_permission(roles: list[str], action: str, user_id: str) -> None:
    if not has_permission(roles, action):
        raise AuthorizationError(user_id, action, required_role=action)
```

### The most common mistake
Editing `ROLE_PERMISSIONS` without a code review — because it's policy-as-code (not a database table), every permission change is a diff someone approves; treating it as "just a dict I can tweak" defeats that purpose.

### Where it lives in our app
File: `security/rbac.py` → `ROLE_PERMISSIONS`, `has_permission()`, `get_permissions()`, `require_permission()`, `rbac_check_node()`
Tests: `tests/security/test_security.py::TestRBAC, TestRBACNode`

---

## Concept 112: Multi-Tenant Graphs  [Module 10]
**Date:** 2026-06-26 | **Complexity:** Intermediate | **Layer:** Security / Observability

### One-line summary
`tenant_id` flows through every log line, audit event, and LangSmith trace tag — it's carried in `config["configurable"]["tenant_id"]` from the auth layer, not re-derived per node.

### Production standard applied
Structured logging rule (`logger.info("node=%s thread=%s ...", ...)`) is extended to always include `tenant_id` where relevant — every audit event and trace is filterable per tenant without needing separate log streams per customer.

### When to reach for it
- ✅ Any SaaS-style deployment serving multiple customers from one graph/checkpointer
- ⚠️ Current gap: `tenant_id` is *tagged* everywhere (logs, audit, tracing) but there's no enforcement that a request's `tenant_id` actually matches the ticket/thread being accessed — cross-tenant data isolation is not yet code-enforced, only labeled

### The canonical pattern
```python
# observability/tracing.py
def get_run_metadata(tenant_id: str, user_id: str, thread_id: str) -> dict:
    return {"tags": [f"tenant:{tenant_id}"], "metadata": {"tenant_id": tenant_id, "user_id": user_id, "thread_id": thread_id}}
```

### The most common mistake
Treating "tenant_id is in the metadata" as equivalent to "tenant isolation is enforced" — it enables *observability* per tenant, not *access control* per tenant. Don't assume this alone stops User A's request from reading User B's thread; that check doesn't exist yet.

### Where it lives in our app
File: `state.py` → `TicketMetadata.tenant_id`; `observability/tracing.py` → `get_run_metadata()`
Tests: — (no dedicated test; covered incidentally via `tests/security/test_security.py`)

---

## Concept 113: Governance  [Module 10]
**Date:** 2026-06-26 | **Complexity:** Beginner | **Layer:** Service

### One-line summary
Governance here means policy is code: `ROLE_PERMISSIONS` (Concept 111) is a git-tracked, reviewable dict — permission changes go through the same code-review process as any other logic change, not a runtime admin panel.

### Production standard applied
Policy-as-code over database-backed policy: anyone can `git blame` a permission change and see who approved it and why, which is a stronger governance guarantee than an audit log of admin-panel clicks.

### When to reach for it
- ✅ Any regulated or high-trust workflow where "who can do what" needs to be provably reviewed, not just recorded after the fact
- ❌ Don't treat this as a substitute for a real compliance/legal review process — it's the technical foundation a real process builds on

### The canonical pattern
```python
# Governance = policy as reviewable code, not runtime-mutable state:
require_permission(roles, "refunds:approve", user_id)   # every permitted action traces back to a committed dict entry
```

### The most common mistake
Adding a runtime permission-bypass flag ("just for this one deployment") — that reintroduces exactly the un-reviewed, un-audited policy mutation that keeping RBAC as code was meant to prevent.

### Where it lives in our app
File: `security/rbac.py` → `ROLE_PERMISSIONS`
Tests: `tests/security/test_security.py::TestRBAC`

---

## Concept 114: Auditability  [Module 10]
**Date:** 2026-06-26 | **Complexity:** Beginner | **Layer:** Service

### One-line summary
`audit_events` (`Annotated[list[dict], operator.add]`, introduced in Module 7's Concept 70) is the mechanism that makes the enterprise layer auditable — every sensitive action across billing, escalation, compliance, and RBAC denial appends here, and the reducer guarantees nothing is ever overwritten.

### Production standard applied
Same append-only reducer guarantee as Concept 70, now load-bearing for enterprise auditability requirements: the checkpointer persists `audit_events` after every node, so the full trail survives a crash or replay.

### When to reach for it
- ✅ Anything a compliance officer, auditor, or incident responder might need to reconstruct later — extend to new sensitive actions by calling `create_audit_event()`, never by adding a separate logging mechanism
- ❌ Don't build a second audit mechanism (e.g. a direct DB write) alongside `audit_events` — that creates two sources of truth that can drift

### The canonical pattern
```python
# Every sensitive node appends, never replaces:
return {"audit_events": [create_audit_event("rbac_check", "permission_denied", thread_id, action=action, required_role=role)]}
```

### The most common mistake
Assuming a node needs a new audit mechanism because it's "enterprise" code — it should call the exact same `create_audit_event()` helper from `observability/audit.py` that billing and escalation already use. One audit trail, one format, everywhere.

### Where it lives in our app
File: `observability/audit.py` → `create_audit_event()`; `state.py` → `audit_events` reducer
Tests: `tests/unit/test_workflows.py`, `tests/security/test_security.py`

---

## Concept 115: Compliance  [Module 10]
**Date:** 2026-06-26 | **Complexity:** Intermediate | **Layer:** Security / Service

### One-line summary
Same `detect_pii()` / `compliance_gate_node()` from Concept 71 (Module 7) — Module 10 doesn't reimplement PII detection, it exercises the same `security/compliance.py` module alongside RBAC and tracing in the enterprise test suite.

### Production standard applied
Reuse over duplication: one PII-detection gate, called from one place, is easier to keep correct than two similar-but-slightly-different implementations drifting apart over time.

### When to reach for it
- ✅ Wherever user-supplied text is logged, sent to an LLM, or persisted — flag PII at the boundary, once
- ❌ Don't add a second PII-detection pass elsewhere in the app "to be extra safe" — one gate, one place, one set of patterns to maintain

### The canonical pattern
```python
# Same as Concept 71 — compliance_gate_node is reused, not reimplemented
pii_types = detect_pii(ticket_text)
if pii_types:
    audit_events.append(create_audit_event("compliance", "pii_detected", thread_id, pii_types=pii_types))
```

### The most common mistake
See Concept 71 — treating a PII hit as something to block on rather than flag. That decision boundary hasn't changed moving from Module 7 to Module 10.

### Where it lives in our app
File: `security/compliance.py` → `detect_pii()`, `compliance_gate_node()`
Tests: `tests/unit/test_workflows.py::TestPIIDetection`

---

## Concept 116: LangSmith Tracing  [Module 10]
**Date:** 2026-06-26 | **Complexity:** Beginner | **Layer:** Observability

### One-line summary
`configure_tracing()` sets the `LANGCHAIN_TRACING_V2`/`LANGCHAIN_API_KEY`/`LANGCHAIN_PROJECT` env vars once at startup; without a `LANGSMITH_API_KEY` it's a complete no-op with zero performance impact.

### Production standard applied
Opt-in observability: tracing must be explicitly configured before any graph/LLM call happens (called once in `main.py` startup) — nothing traces "by accident," and forgetting to set the API key degrades to silently-off rather than erroring.

### When to reach for it
- ✅ Any environment where you want to inspect LLM calls, node timings, and prompts after the fact — set `LANGSMITH_API_KEY` and call `configure_tracing()` before building the graph
- ❌ Don't call `configure_tracing()` after the graph is already compiled/invoked — traces from calls made before configuration are simply lost, not error

### The canonical pattern
```python
# observability/tracing.py
def configure_tracing(settings: Settings) -> None:
    if settings.langsmith_api_key is None:
        return   # no-op — tracing stays off, no error
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key.get_secret_value()
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project

def get_run_metadata(tenant_id: str, user_id: str, ticket_type: str, thread_id: str) -> dict:
    return {"tags": [f"tenant:{tenant_id}", f"ticket_type:{ticket_type}"], "metadata": {"thread_id": thread_id}}
```

### The most common mistake
Calling `configure_tracing()` inside a node or per-request — it sets process-wide environment variables, so it belongs in app startup (`main.py`) exactly once, not scattered through request-handling code.

### Where it lives in our app
File: `observability/tracing.py` → `configure_tracing()`, `get_run_metadata()`, `is_tracing_enabled()`
Tests: `tests/security/test_security.py::TestTracing`

---

## Concept 117: Graph Observability  [Module 10]
**Date:** 2026-06-26 | **Complexity:** Intermediate | **Layer:** Observability

### One-line summary
`observability/metrics.py` collects per-node latency and token usage in an in-process dict keyed by `thread_id`, populated via the Observer-pattern post-hook (Module 3's `node_with_hooks`) so node code never calls metrics functions directly.

### Production standard applied
Same Observer pattern from Module 3 reused for a new concern: `configure_metrics()` registers `record_node_latency` as a post-hook, so instrumentation is added without touching a single node's business logic.

### When to reach for it
- ✅ Local dev / small-deployment visibility into per-node timing and token spend
- ⚠️ In-process storage means metrics are lost on process restart — for real production monitoring, swap the internal `_store` for a Prometheus push gateway or similar; this is explicitly a stopgap, not a scaling solution

### The canonical pattern
```python
# observability/metrics.py
def record_node_latency(node: str, duration_ms: float, thread_id: str) -> None:
    _store.setdefault(thread_id, {}).setdefault("latencies", []).append({"node": node, "ms": duration_ms})

def get_session_metrics(thread_id: str) -> dict[str, Any]:
    return _store.get(thread_id, {})
```

### The most common mistake
Reading `get_session_metrics()` expecting durable, cross-restart history — it's an in-memory dict for the life of the process, useful for a single debugging session or a `/metrics/{thread_id}` debug endpoint, not a substitute for real observability infrastructure.

### Where it lives in our app
File: `observability/metrics.py` → `record_node_latency()`, `record_token_usage()`, `get_session_metrics()`, `configure_metrics()`
Tests: `tests/security/test_security.py::TestMetrics`

---

## Concept 118: Graph Testing  [Module 10]
**Date:** 2026-06-26 | **Complexity:** Intermediate | **Layer:** Testing (all layers)

### One-line summary
The security/enterprise layer landed with 61 new tests (`test_security.py`: 46, `test_api.py`: 15), following the same unit/integration/e2e pyramid used throughout the app — bringing the full suite to 541 tests at the time (currently 563 passing, `-m "not e2e"`).

### Production standard applied
Every module ships tests alongside the code, not after — `security/`, `observability/`, and `api/` all got dedicated test files in the same session the features were built, matching the project's "tests are not an afterthought" rule.

### When to reach for it
- ✅ Any new node, agent, or service function — write the unit test in the same commit
- ✅ Any new FastAPI route — cover it in `tests/security/test_api.py` style (auth required, RBAC enforced, debug-gated)

### The canonical pattern
```python
# tests/security/test_api.py — pattern for FastAPI route tests
class TestTicketsEndpoint:
    def test_requires_bearer_token(self, client):
        response = client.post("/tickets", json={"message": "help"})
        assert response.status_code == 401

    def test_rbac_denies_refund_without_role(self, client, readonly_token):
        response = client.post("/tickets", json={"message": "refund $1000"}, headers=readonly_token)
        assert response.status_code == 403
```

### The most common mistake
Testing a FastAPI route with a real (or improperly mocked) LLM call — every route test in `test_api.py` patches the LLM/embeddings/checkpointer layer so the test suite stays fast and deterministic; only `tests/e2e/` (marked `@pytest.mark.e2e`) makes real calls.

### Where it lives in our app
File: `tests/security/test_security.py`, `tests/security/test_api.py`
Tests: (this concept *is* the tests)

---

## Concept 119: Graph Debugging  [Module 10]
**Date:** 2026-06-26 | **Complexity:** Beginner | **Layer:** Delivery / Infrastructure

### One-line summary
`settings.graph_debug` gates LangGraph's `debug=True` compile flag and the FastAPI `/metrics/{thread_id}` debug endpoint — verbose internals are opt-in via config, never on by default.

### Production standard applied
Config-driven behavior (`Settings.graph_debug: bool = Field(default=False)`) instead of a code-level `if DEBUG:` constant — toggling debug output is an environment change, not a code change or redeploy.

### When to reach for it
- ✅ Local development, or temporarily in a specific environment when diagnosing a stuck/misbehaving graph run
- ❌ Never leave `graph_debug=True` on in production by default — it exposes internal state (`/metrics/{thread_id}`) and verbose LangGraph output that shouldn't be public

### The canonical pattern
```python
# graph.py
def build_graph(**kwargs):
    settings = get_settings()
    return builder.build(checkpointer=get_checkpointer(), debug=settings.graph_debug, **kwargs)

# api/routes.py
@app.get("/metrics/{thread_id}")
async def metrics(thread_id: str, settings: Settings = Depends(get_settings)):
    if not settings.graph_debug:
        raise HTTPException(404)   # hidden entirely unless debug mode is on
    return get_session_metrics(thread_id)
```

### The most common mistake
Gating a debug endpoint with a `403 Forbidden` instead of `404 Not Found` — a `403` confirms the endpoint exists to anyone probing it; returning `404` when `graph_debug` is off means the endpoint appears not to exist at all.

### Where it lives in our app
File: `core/settings.py` → `graph_debug` field; `graph.py` → `build_graph()`; `api/routes.py` → `/metrics/{thread_id}`
Tests: — (implicitly covered by `tests/security/test_api.py` debug-gate tests)

---

## Concept 120: Enterprise Deployment  [Module 10]
**Date:** 2026-06-26 | **Complexity:** Intermediate | **Layer:** Delivery

### One-line summary
`api/routes.py`'s `create_app()` factory wraps the same compiled graph used by the CLI (`main.py`) behind FastAPI — JWT Bearer auth, SSE token streaming on `/tickets`, `/health`, and a dev-only `/auth/token` issuer.

### Production standard applied
Delivery-layer swap without touching business logic: `create_app()` calls the exact same `build_graph()`/`get_agent()`/node functions the CLI does — FastAPI is just a second delivery channel (Clean Architecture's outermost layer), proving the layered design pays off.

### When to reach for it
- ✅ Any time the app needs a non-CLI consumer (web frontend, another service) — add a route, don't fork the graph
- ❌ `/auth/token` is explicitly dev-only (issues JWTs without real credential verification) — never expose it in a deployed environment; it exists purely to make local testing of the Bearer-auth flow easy

### The canonical pattern
```python
# api/routes.py
def create_app() -> FastAPI:
    app = FastAPI()

    @app.post("/tickets")
    async def create_ticket(req: TicketRequest, user: AuthContext = Depends(get_current_user)):
        graph = build_graph()
        initial_state = _build_initial_state(req, user)
        return StreamingResponse(_stream_tokens(graph, initial_state, config), media_type="text/event-stream")

    return app
```

### The most common mistake
Building a second, parallel graph-construction path for the API instead of reusing `build_graph()` from `graph.py` — that would let the CLI and API drift out of sync (different node sets, different interrupt config) with no test catching it.

### Where it lives in our app
File: `api/routes.py` → `create_app()`, `get_current_user()`, `_stream_tokens()`
Tests: `tests/security/test_api.py` (15 tests)


---

## Backfilled entries (concepts documented after the fact)
*Added 2026-11-05 to close gaps between the Progress Tracker and the per-concept entries. Each entry was written against the code as it exists today; where the tracker's description and the code disagree, the entry says so.*

---

## Concept 10: Annotated State  [Module 2]
**Date:** 2026-11-05 | **Complexity:** Beginner | **Layer:** Graph

### One-line summary
`Annotated[T, reducer]` attaches a merge function to a state field so LangGraph knows how to combine a node's partial update with the existing value.

### Production standard applied
State Design Standard: every list that nodes append to is `Annotated[list[T], reducer]`; scalars that one node replaces stay plain.

### When to reach for it
- A field is written by more than one node, or by parallel branches
- A list must accumulate across steps instead of being overwritten
- A plain field would silently lose data (last write wins)

### The canonical pattern
```python
# state.py
messages: Annotated[list[BaseMessage], add_messages]
agent_results: Annotated[list[dict], operator.add]
```

### The most common mistake
Declaring a list field without `Annotated`: each node's return replaces the list, and earlier entries vanish with no error.

### Where it lives in our app
File: `state.py` → `AgentState` (`messages`, `retrieved_memories`, `agent_results`, `retrieved_docs`, `rbac_roles`, `audit_events`)
Tests: `tests/unit/test_reducers.py::TestAnnotatedStateDeclaration`

---

## Concept 11: operator.add  [Module 2]
**Date:** 2026-11-05 | **Complexity:** Beginner | **Layer:** Graph

### One-line summary
`operator.add` as a reducer concatenates lists, so each node's returned list is appended to the field.

### Production standard applied
Show the data-loss bug first (`TestTheDataLossBugWithoutReducer`), then fix with the reducer. Append-only audit trail relies on it.

### When to reach for it
- Fan-in from parallel workers (`agent_results`)
- Append-only logs (`audit_events`)
- Accumulating retrieval results (`retrieved_docs`)

### The canonical pattern
```python
# state.py
agent_results: Annotated[list[dict], operator.add]
audit_events: Annotated[list[dict], operator.add]
```

### The most common mistake
Returning the whole accumulated list from a node: with `operator.add` it is appended again, duplicating every earlier entry. Return only the new items.

### Where it lives in our app
File: `state.py` → `AgentState` (5 `operator.add` fields: `retrieved_memories`, `agent_results`, `retrieved_docs`, `rbac_roles`, `audit_events`)
Tests: `tests/unit/test_reducers.py::TestOperatorAddReducer`, `::TestOperatorAddBehaviour`

---

## Concept 13: State Merging  [Module 2]
**Date:** 2026-11-05 | **Complexity:** Beginner | **Layer:** Graph

### One-line summary
A node returns a partial dict; LangGraph merges each key into state (via the field's reducer if it has one, else by replacing it).

### Production standard applied
Node Rule 5: return only the fields the node changes, never the full state.

### When to reach for it
- Always, in every node
- When asserting in tests that a node doesn't touch unrelated fields

### The canonical pattern
```python
# agents/nodes.py (memory_write_node)
return {"audit_events": [{"node": "memory_write_node", "action": "memory_written", ...}]}
```

### The most common mistake
Returning `{**state, ...}`: with reducers on list fields this re-appends everything already in state.

### Where it lives in our app
File: `agents/nodes.py` (all nodes return partial dicts)
Tests: `tests/unit/test_reducers.py::TestStateMerging`

---

## Concept 14: Reducer Design Patterns  [Module 2]
**Date:** 2026-11-05 | **Complexity:** Intermediate | **Layer:** Graph

### One-line summary
Any callable `(existing, new) -> merged` can be a reducer, enabling dedup, bounded lists, or priority merges beyond plain append.

### Production standard applied
Strategy-pattern preview: merge logic is a swappable function attached via `Annotated`.

### When to reach for it
- Multi-query retrieval needs duplicate documents dropped
- A list must stay bounded
- Merge needs a rule other than concatenation

### The canonical pattern
```python
# tests/unit/test_reducers.py (illustrative reducer)
def _dedup_by_source(existing: list, new: list) -> list: ...

class DocState(TypedDict):
    docs: Annotated[list[dict], _dedup_by_source]
```

### The most common mistake
Writing a reducer with side effects or non-determinism; reducers run on every merge, including during checkpoint replay.

### Where it lives in our app
**Discrepancy:** the tracker lists `state.py (custom reducer pattern)`, but `state.py` defines no custom reducer. The only custom reducer is the test-local `_dedup_by_source` in `tests/unit/test_reducers.py:698`; production state uses `add_messages` and `operator.add` only.
Tests: `tests/unit/test_reducers.py::TestCustomReducers`

---

## Concept 15: Router Pattern  [Module 3]
**Date:** 2026-11-05 | **Complexity:** Beginner | **Layer:** Graph

### One-line summary
A router is a pure, synchronous function `(AgentState) -> str` that picks the next node name for a conditional edge.

### Production standard applied
`RouterFn` TypeAlias; routers do no I/O and never use an LLM.

### When to reach for it
- Branching on a value already in state (`ticket_type`)
- Any decision that is a plain lookup

### The canonical pattern
```python
# agents/nodes.py
ROUTE_KEYS: dict[str, str] = {
    "billing": "billing_node",
    "technical": "supervisor_node",
    "general": "supervisor_node",
    "research": "supervisor_node",
}

def route_ticket(state: AgentState) -> str:   # sync, pure
    ...  # unknown ticket_type falls back to "general"
```

### The most common mistake
Making the router `async def`: LangGraph calls it synchronously, so you get a coroutine object instead of a node name.

### Where it lives in our app
File: `agents/nodes.py` → `route_ticket`, `ROUTE_KEYS`
Tests: `tests/unit/test_module3.py::TestRouterPattern`

---

## Concept 16: ToolNode  [Module 3]
**Date:** 2026-11-05 | **Complexity:** Intermediate | **Layer:** Service

### One-line summary
Tools are classes with a `name`, `description`, and a Pydantic `args_schema`, so the LLM-facing contract is typed and validated.

### Production standard applied
Tool-as-class with `args_schema` (Pydantic); sync `_run` and async `_arun` both provided.

### When to reach for it
- An agent needs a callable capability (KB search, web search)
- Inputs must be validated before the call runs

### The canonical pattern
```python
# tools/kb_tool.py
class KnowledgeBaseTool(BaseTool):
    name: str = "knowledge_base_search"
    args_schema: type[BaseModel] = KBToolInput
    def _run(self, query: str, top_k: int = 5) -> str: ...
    async def _arun(self, query: str, top_k: int = 5) -> str: ...
```

### The most common mistake
A vague `description`: the model decides when to call the tool from that text alone.

### Where it lives in our app
File: `tools/kb_tool.py` → `KnowledgeBaseTool`; `tools/search_tool.py` → `SearchTool`; `tools/__init__.py` → `get_research_tools()`, `get_technical_tools()`
Note: a prebuilt `ToolNode` is not instantiated anywhere in the graph; this entry covers the tool contracts only.
Tests: `tests/unit/test_module3.py::TestToolContracts`

---

## Concept 17: create_react_agent  [Module 3]
**Date:** 2026-11-05 | **Complexity:** Intermediate | **Layer:** Graph

### One-line summary
`create_react_agent` is LangGraph's prebuilt tool-calling loop; this project instead hand-rolls agents behind a `BaseAgent` ABC to keep control of state and routing.

### Production standard applied
Open/Closed: new agents extend `BaseAgent` without modifying the supervisor.

### When to reach for it
- Prebuilt: a plain "LLM + tools" loop with no custom state handling
- Hand-rolled (this app): agents that must set `pending_approval`, write `agent_results`, or integrate HITL

### The canonical pattern
```python
# agents/base.py
class BaseAgent(ABC):
    @property
    @abstractmethod
    def agent_name(self) -> str: ...
    @abstractmethod
    def get_tools(self) -> list: ...
    @abstractmethod
    async def process(self, state: AgentState, config: RunnableConfig) -> dict: ...
```

### The most common mistake
Reaching for the prebuilt agent and then fighting it to return custom state fields.

### Where it lives in our app
File: `agents/base.py` → `BaseAgent`. `create_react_agent` itself is not used anywhere in the repo.
Tests: `tests/unit/test_module3.py::TestBaseAgentABC`

---

## Concept 18: Dynamic Routing  [Module 3]
**Date:** 2026-11-05 | **Complexity:** Intermediate | **Layer:** Graph

### One-line summary
Agents register themselves by name in a registry, and the supervisor resolves them at runtime instead of hard-coding a branch per agent.

### Production standard applied
Registry pattern (`@register_agent`) plus `BaseAgent` ABC.

### When to reach for it
- The set of specialist agents will grow
- Agent selection depends on runtime state

### The canonical pattern
```python
# agents/billing_agent.py
@register_agent("billing")
def create_billing_agent() -> BillingAgent: ...

# agents/registry.py
def get_agent(name: str) -> BaseAgent: ...   # KeyError if unregistered
def list_agents() -> list[str]: ...
```

### The most common mistake
Forgetting to import the agent module: the decorator never runs, so `get_agent("technical")` raises `KeyError`. `graph.py` imports the agents for this reason.

### Where it lives in our app
File: `agents/registry.py` → `register_agent`, `get_agent`, `list_agents`, `clear_registry`
Tests: `tests/unit/test_module3.py::TestAgentRegistry`

---

## Concept 19: Hooks  [Module 3]
**Date:** 2026-11-05 | **Complexity:** Intermediate | **Layer:** Infrastructure

### One-line summary
Observer-pattern pre/post hooks wrap a node so logging and metrics stay out of node business logic.

### Production standard applied
Observer pattern in `observability/hooks.py`; hook failures are caught per hook so they can never crash a node.

### When to reach for it
- Cross-cutting concerns (latency, audit logging) across many nodes

### The canonical pattern
```python
# observability/hooks.py
register_pre_hook(_audit_hook)
register_post_hook(_latency_hook)
wrapped = node_with_hooks("classify", classify_ticket_node)
```

### The most common mistake
Defining hooks but never wrapping nodes or calling `configure_hooks()`.

### Where it lives in our app
File: `observability/hooks.py` → `node_with_hooks`, `register_pre_hook`, `register_post_hook`, `configure_hooks`
**Discrepancy:** `graph.py` and `main.py` never call `node_with_hooks()` or `configure_hooks()`; hooks are implemented and tested but not wired into the running graph. (`observability/metrics.py` also expects the post-hook to feed `record_node_latency`.)
Tests: `tests/unit/test_module3.py::TestHooks`

---

## Concept 20: RunnableConfig  [Module 3]
**Date:** 2026-11-05 | **Complexity:** Beginner | **Layer:** Graph

### One-line summary
`RunnableConfig` carries per-invocation values (`thread_id`, `user_id`, tenant) under `config["configurable"]`, read safely with `.get()`.

### Production standard applied
Never `KeyError`: helpers use `.get()` chains with a default.

### When to reach for it
- Any node needing the thread or user identity

### The canonical pattern
```python
# agents/nodes.py
def _get_thread_id(config: RunnableConfig) -> str:
    return config.get("configurable", {}).get("thread_id", "unknown")
```

### The most common mistake
`config["configurable"]["thread_id"]`: raises `KeyError` whenever the caller omits it (e.g. in tests).

### Where it lives in our app
File: `agents/nodes.py` → `_get_thread_id`, `_get_user_id` (falls back to `"anonymous"`)
Tests: `tests/unit/test_module3.py::TestRunnableConfig`

---

## Concept 21: Runtime Configuration  [Module 3]
**Date:** 2026-11-05 | **Complexity:** Beginner | **Layer:** Infrastructure

### One-line summary
Behaviour is switched by typed `Settings` fields at runtime rather than by editing code.

### Production standard applied
Config-driven behaviour; `get_settings()` is the only config entry point (no `os.getenv`).

### When to reach for it
- A feature needs an on/off or tuning switch (debug logging, backends, top-k)

### The canonical pattern
```python
# observability/hooks.py
def _latency_hook(node_name, result, config):
    settings = get_settings()
    if not settings.graph_debug:
        return
    ...
```

### The most common mistake
Reading settings once at import time; tests then can't change them (hence `get_settings.cache_clear()` in `conftest.py`).

### Where it lives in our app
File: `observability/hooks.py` → `_latency_hook`; `core/settings.py` → `graph_debug`
Tests: `tests/unit/test_module3.py::TestRuntimeConfiguration`

---

## Concept 24: InMemorySaver  [Module 4]
**Date:** 2026-11-05 | **Complexity:** Beginner | **Layer:** Infrastructure

### One-line summary
`MemorySaver` keeps checkpoints in a Python dict: same API as the SQLite saver, no disk I/O, gone on exit.

### Production standard applied
A fresh instance per test via a fixture, so thread state never leaks between tests.

### When to reach for it
- Unit/integration tests; throwaway local runs

### The canonical pattern
```python
# tests/conftest.py
@pytest.fixture
def memory_checkpointer():
    return MemorySaver()
```

### The most common mistake
Sharing one `MemorySaver` across tests, so thread A's state shows up in thread B's assertions.

### Where it lives in our app
File: `tests/conftest.py` → `memory_checkpointer`; `core/factories.py` → `get_checkpointer()` (`"memory"` branch)
Tests: `tests/unit/test_persistence.py::TestInMemorySaver`

---

## Concept 25: SQLite Checkpointer  [Module 4]
**Date:** 2026-11-05 | **Complexity:** Intermediate | **Layer:** Infrastructure

### One-line summary
`SqliteSaver` persists checkpoints to a local file (`memory.db`), so a session can resume after the process exits.

### Production standard applied
Factory pattern: callers use `get_checkpointer()`; not `lru_cache`d because it owns an open connection.

### When to reach for it
- Local dev and single-user CLI sessions that must survive restarts

### The canonical pattern
```python
# core/factories.py
conn = sqlite3.connect(settings.sqlite_path, check_same_thread=False)
return SqliteSaver(conn)
```

### The most common mistake
Caching the checkpointer: tests share one connection and hit "database is locked".

### Where it lives in our app
File: `core/factories.py` → `get_checkpointer()`
**Discrepancy:** `main.py` does not call `get_checkpointer()`; it opens `AsyncSqliteSaver.from_conn_string(settings.sqlite_path)` directly (needed because the CLI uses async streaming). The factory's sync `SqliteSaver` path is exercised by tests.
Tests: `tests/unit/test_persistence.py::TestCheckpointerFactory`, `::TestPersistentThreads`

---

## Concept 26: Postgres Checkpointer  [Module 4]
**Date:** 2026-11-05 | **Complexity:** Intermediate | **Layer:** Infrastructure

### One-line summary
`PostgresSaver` would give multi-process shared checkpoints through the same `BaseCheckpointSaver` contract; **it is not implemented in this repo**.

### Production standard applied
Liskov substitution: `build_graph(checkpointer=...)` accepts any `BaseCheckpointSaver`.

### When to reach for it
- Multiple API replicas must resume the same `thread_id`

### The canonical pattern
```python
# core/settings.py (current reality)
checkpointer_backend: Literal["memory", "sqlite"] = Field(default="sqlite")
# core/factories.py raises ValueError for any other value
```

### The most common mistake
Believing the docstring "swap SQLite → Postgres in one line": there is no `postgres_url` setting and no Postgres branch.

### Where it lives in our app
No implementation. Seam only: `core/factories.py` (docstring), `core/graph_builder.py` (`build()` docstring). Tracker row 26 has no test, consistent with this.
Full write-up: `docs/LangGraphFoundations.md` → Concept 26

---

## Concept 27: Thread IDs  [Module 4]
**Date:** 2026-11-05 | **Complexity:** Beginner | **Layer:** Delivery

### One-line summary
A `thread_id` identifies one conversation; the checkpointer keys all saved state by it.

### Production standard applied
`uuid4()` per session; thread id used as a correlation id in log lines.

### When to reach for it
- Every invocation of a graph with a checkpointer

### The canonical pattern
```python
# main.py
session_id = str(uuid.uuid4())
config = _build_config(session_id, settings)   # config["configurable"]["thread_id"]
```

### The most common mistake
Generating a new `thread_id` per message: each turn starts with empty history.

### Where it lives in our app
File: `main.py` → `run()`, `_build_config()`
Tests: `tests/unit/test_persistence.py::TestThreadAndMetadata`

---

## Concept 28: User IDs  [Module 4]
**Date:** 2026-11-05 | **Complexity:** Beginner | **Layer:** Graph

### One-line summary
`user_id` travels in `config["configurable"]` and scopes per-user data such as long-term memory.

### Production standard applied
Typed `TicketMetadata` (includes `user_id`, `tenant_id`); safe extraction with a default.

### When to reach for it
- Scoping memory, audit events, and authorization to a person

### The canonical pattern
```python
# agents/nodes.py
return config.get("configurable", {}).get("user_id", "anonymous")
```

### The most common mistake
Treating `"anonymous"` as a real user: all unauthenticated callers then share one memory namespace.

### Where it lives in our app
File: `state.py` → `TicketMetadata`; `agents/nodes.py` → `_get_user_id`
Tests: `tests/unit/test_persistence.py::TestThreadAndMetadata`

---

## Concept 29: Metadata  [Module 4]
**Date:** 2026-11-05 | **Complexity:** Beginner | **Layer:** Graph

### One-line summary
Structured ticket metadata is a `TypedDict`, not a free-form dict.

### Production standard applied
No raw `dict[str, Any]` in state: fields are enumerated and typed.

### When to reach for it
- Any nested structure carried in state

### The canonical pattern
```python
# state.py
class TicketMetadata(TypedDict):
    ticket_id: str
    tenant_id: str
    user_id: str
    priority: Priority
    created_at: str      # ISO 8601
    sla_deadline: str    # ISO 8601
```

### The most common mistake
Misspelling a key in a plain dict; a `TypedDict` lets mypy catch it.

### Where it lives in our app
File: `state.py` → `TicketMetadata`, `AgentState.ticket_metadata`
Tests: `tests/unit/test_persistence.py::TestThreadAndMetadata`

---

## Concept 30: Persistent Threads  [Module 4]
**Date:** 2026-11-05 | **Complexity:** Intermediate | **Layer:** Delivery

### One-line summary
Reusing one `thread_id` across turns (and across restarts with SQLite) lets the graph continue the same conversation.

### Production standard applied
One session id per process, reused for every message in the loop.

### When to reach for it
- Multi-turn chat; resumable sessions

### The canonical pattern
```python
# main.py
async with AsyncSqliteSaver.from_conn_string(settings.sqlite_path) as checkpointer:
    graph = build_graph(checkpointer=checkpointer, debug=settings.graph_debug)
    ...  # same session_id for every ticket in the loop
```

### The most common mistake
Expecting a restart to resume automatically: the CLI mints a fresh `uuid4()` each run, so resuming an old thread requires supplying its id.

### Where it lives in our app
File: `main.py` → `run()`
Tests: `tests/unit/test_persistence.py::TestPersistentThreads`

---

## Concept 31: Checkpoint Storage  [Module 4]
**Date:** 2026-11-05 | **Complexity:** Intermediate | **Layer:** Graph

### One-line summary
`graph.get_state_history(config)` iterates every saved checkpoint for a thread, most recent first.

### Production standard applied
Typed iteration over `StateSnapshot`s; used for debugging and time travel.

### When to reach for it
- Inspecting how state evolved; choosing a checkpoint to replay from

### The canonical pattern
```python
history = list(graph.get_state_history(config))   # newest first
```

### The most common mistake
Assuming the list is oldest-first.

### Where it lives in our app
Pattern demonstrated in tests only; no production code calls `get_state_history`.
Tests: `tests/unit/test_persistence.py::TestCheckpointStorage`

---

## Concept 32: State Recovery  [Module 4]
**Date:** 2026-11-05 | **Complexity:** Advanced | **Layer:** Graph

### One-line summary
After a crash, execution resumes from the last checkpoint, so nodes may re-run and must be safe to repeat.

### Production standard applied
Idempotency: check-before-act inside nodes.

### When to reach for it
- Any node with a side effect that could be replayed

### The canonical pattern
```python
# idea: skip the side effect if its result is already recorded in state
if state.get("final_response"):
    return {}
```

### The most common mistake
Assuming a node runs exactly once; replays run it again.

### Where it lives in our app
Demonstrated in `tests/unit/test_persistence.py::TestIdempotencyAndDurableExecution`; the snippet above is illustrative of the pattern the tests exercise, not a quote from a production node.

---

## Concept 33: Durable Execution  [Module 4]
**Date:** 2026-11-05 | **Complexity:** Advanced | **Layer:** Graph

### One-line summary
Isolating side effects in dedicated nodes bounds what a replay can repeat.

### Production standard applied
Side-effect isolation: persistence/notification lives in its own node (e.g. `memory_write_node`, audit nodes), not inside reasoning nodes.

### When to reach for it
- Writes to external systems; anything non-idempotent

### The canonical pattern
```python
# separate nodes, each with a single responsibility
.node("approval_check", approval_check_node)
.node("memory_write",   memory_write_node)
```

### The most common mistake
Mixing an LLM call and an external write in one node: a retry repeats both.

### Where it lives in our app
File: `graph.py` (node layout); `agents/nodes.py` → `memory_write_node`
Tests: `tests/unit/test_persistence.py::TestIdempotencyAndDurableExecution`

---

## Concept 35: Long-Term Memory  [Module 5]
**Date:** 2026-11-05 | **Complexity:** Intermediate | **Layer:** Service

### One-line summary
Cross-thread, per-user memory that outlives any single conversation (as opposed to the thread-scoped checkpointer).

### Production standard applied
Interface Segregation: `MessageStoreProtocol` (short-term) and `SemanticSearchProtocol` (long-term) are separate small protocols.

### When to reach for it
- Facts about a user that should persist across sessions

### The canonical pattern
```python
# memory/protocols.py
class SemanticSearchProtocol(Protocol):
    async def astore(...): ...
    async def asearch(...): ...
```

### The most common mistake
Confusing checkpointer state (per thread) with long-term memory (per user).

### Where it lives in our app
File: `memory/protocols.py`, `memory/store.py`
Tests: `tests/unit/test_memory.py::TestLongTermMemoryStore`

---

## Concept 36: Memory Store  [Module 5]
**Date:** 2026-11-05 | **Complexity:** Intermediate | **Layer:** Service

### One-line summary
`LongTermMemoryStore` wraps LangGraph's `InMemoryStore` behind our protocol, created by the `get_memory_store()` factory.

### Production standard applied
Factory with `lru_cache` (safe: a plain in-process dict, unlike the SQLite connection).

### When to reach for it
- Nodes need memory without knowing the backing store

### The canonical pattern
```python
# memory/store.py
@lru_cache(maxsize=1)
def get_memory_store() -> LongTermMemoryStore: ...
```

### The most common mistake
Expecting persistence: `InMemoryStore` is **process-local and lost on exit**. The docstring suggests `AsyncPostgresStore` for production; it is not implemented. (`about.MD` mentions ChromaDB for memory; the code does not use it.)

### Where it lives in our app
File: `memory/store.py` → `LongTermMemoryStore`, `get_memory_store()`
Tests: `tests/unit/test_memory.py::TestMemoryStoreFactory`

---

## Concept 37: Semantic Memory  [Module 5]
**Date:** 2026-11-05 | **Complexity:** Intermediate | **Layer:** Service

### One-line summary
Memories are embedded so retrieval matches by meaning rather than exact key.

### Production standard applied
Dependency injection of embeddings with graceful fallback to an exact-key store if embeddings can't load.

### When to reach for it
- Free-text memories queried with natural language

### The canonical pattern
```python
# memory/store.py
store = InMemoryStore(index={"embed": _embed_fn, "dims": 384})
# except Exception: store = InMemoryStore()   # no semantic search
```

### The most common mistake
Changing the embedding model without updating `dims`: vectors no longer match the index (384 matches `all-MiniLM-L6-v2`).

### Where it lives in our app
File: `memory/store.py` → `get_memory_store()` (`_embed_fn`)
Tests: `tests/unit/test_memory.py::TestMemoryStoreFactory`

---

## Concept 38: Memory Retrieval  [Module 5]
**Date:** 2026-11-05 | **Complexity:** Intermediate | **Layer:** Graph

### One-line summary
`memory_retrieval_node` runs first and loads the user's relevant memories into `retrieved_memories`.

### Production standard applied
Failures degrade gracefully (empty list) and never block the ticket; no `@with_retry` because it doesn't call an external LLM.

### When to reach for it
- Personalising responses with prior context

### The canonical pattern
```python
# agents/nodes.py
memories = await store.asearch(user_id=user_id, query=query)
return {"retrieved_memories": [m["text"] for m in memories if m.get("text")]}
```

### The most common mistake
Letting a memory-store error propagate and failing the whole ticket.

### Where it lives in our app
File: `agents/nodes.py` → `memory_retrieval_node`
Note: the CLAUDE.md curriculum names a `make_memory_retrieval_node()` DI factory; the code uses a plain node function.
Tests: `tests/unit/test_memory.py::TestMemoryRetrievalNode`

---

## Concept 39: Memory Summarization  [Module 5]
**Date:** 2026-11-05 | **Complexity:** Intermediate | **Layer:** Graph

### One-line summary
Each turn is compressed into one sentence by a fast LLM before it is stored.

### Production standard applied
`_summarise_for_memory()` is isolated from node logic so the prompt/strategy can change independently; uses `get_fast_llm()`.

### When to reach for it
- Storing conversation facts cheaply

### The canonical pattern
```python
# agents/nodes.py
llm = get_fast_llm()
response = await llm.ainvoke(prompt)   # "ONE sentence ... Maximum 50 words."
return response.content.strip()
```

### The most common mistake
Storing raw transcripts: memory bloats and retrieval gets noisy.

### Where it lives in our app
File: `agents/nodes.py` → `_summarise_for_memory`
Tests: `tests/unit/test_memory.py::TestSummariseForMemory`

---

## Concept 40: Memory Compression  [Module 5]
**Date:** 2026-11-05 | **Complexity:** Intermediate | **Layer:** Graph

### One-line summary
`memory_write_node` summarises the turn and writes it under the user's namespace, keyed by ticket type and thread prefix.

### Production standard applied
`@with_retry` (LLM call); failures are logged and return `{}` rather than blocking delivery.

### When to reach for it
- Persisting a durable record after a response is produced

### The canonical pattern
```python
# agents/nodes.py
memory_key = f"{ticket_type}_{thread_id[:8]}"
await store.astore(user_id=user_id, key=memory_key, text=summary, metadata={...})
```

### The most common mistake
Assuming compression strategies exist: the curriculum describes rolling-window/hierarchical `CompressionStrategy` implementations; the code has a single summarise-and-store path.

### Where it lives in our app
File: `agents/nodes.py` → `memory_write_node`
Tests: `tests/unit/test_memory.py::TestMemoryWriteNode`

---

## Concept 41: Config Driven Workflows  [Module 5]
**Date:** 2026-11-05 | **Complexity:** Beginner | **Layer:** Infrastructure

### One-line summary
Memory behaviour is tuned from `Settings` (`memory_top_k`, `memory_summarize_after_turns`).

### Production standard applied
Config-driven behaviour with typed settings.

### When to reach for it
- Tuning retrieval depth without code changes

### The canonical pattern
```python
# memory/store.py (asearch)
k = limit if limit is not None else settings.memory_top_k
```

### The most common mistake
Adding a setting that nothing reads. `memory_summarize_after_turns` is declared in `core/settings.py` but unused: `memory_write_node` calls `get_settings()` and discards the result, and its comment says it "always writes after every turn".

### Where it lives in our app
File: `core/settings.py` → `memory_top_k`, `memory_summarize_after_turns`; `memory/store.py` → `asearch` (only `memory_top_k` is honoured)
Tests: `tests/unit/test_memory.py::TestConfigDrivenMemory`

---

## Concept 48: debug mode  [Module 6]
**Date:** 2026-11-05 | **Complexity:** Beginner | **Layer:** Graph

### One-line summary
`compile(debug=True)` makes LangGraph print verbose per-step output; here it is gated behind `settings.graph_debug`.

### Production standard applied
Debug behaviour is config-gated, never hard-coded on.

### When to reach for it
- Diagnosing routing or state-merge problems locally

### The canonical pattern
```python
# graph.py (build_graph)
if debug:
    compile_kwargs["debug"] = debug
# main.py
graph = build_graph(checkpointer=checkpointer, debug=settings.graph_debug)
```

### The most common mistake
Leaving debug on in deployed environments: it is noisy and may log state contents.

### Where it lives in our app
File: `graph.py` → `build_graph(debug=...)`; `core/settings.py` → `graph_debug`; `main.py` → `run()`
Note: this is a thin wiring of a LangGraph feature, not a separate module.
Tests: `tests/unit/test_streaming.py::TestStreamingArchitecture`

---

## Concept 50: Event Streaming  [Module 6]
**Date:** 2026-11-05 | **Complexity:** Intermediate | **Layer:** Delivery

### One-line summary
`graph.astream_events(..., version="v2")` yields fine-grained typed events (node start/end, model tokens, etc.).

### Production standard applied
None in production code: **educational-only**. The app streams with `astream(stream_mode=...)` (`updates` and `messages`), not `astream_events`.

### When to reach for it
- You need per-LLM-call or per-tool events that `updates`/`messages` don't expose

### The canonical pattern
```python
# tests/unit/test_streaming.py
async for event in graph.astream_events(state, config=config, version="v2"):
    ...   # dicts with "event", "name", "data"
```

### The most common mistake
Omitting `version="v2"` and getting the legacy event schema.

### Where it lives in our app
Not used in production code; demonstrated in `tests/unit/test_streaming.py::TestEventStreaming` only.
