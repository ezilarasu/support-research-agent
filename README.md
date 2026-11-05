# support-research-agent

A **learning project**: a multi-agent customer-support and research assistant built with
[LangGraph](https://langchain-ai.github.io/langgraph/), where each LangGraph concept is
implemented as tested code in a working app and documented alongside it.

- **Stack:** LangGraph · Groq (`langchain-groq`) · HuggingFace (embeddings, zero-shot classifier,
  cross-encoder reranker) · FAISS · SQLite checkpointer · Tavily · FastAPI · pydantic-settings
- **Status (v0.1.0):** 91 of 120 curriculum concepts implemented; 541 tests passing
  (`pytest --ignore=tests/e2e`, with `GROQ_API_KEY` set). See [Roadmap](#roadmap).

> This is a teaching/reference codebase, not a deployed product. Several pieces are
> deliberately simple — see [Known limitations](#known-limitations).

## What the app does

A support ticket goes in; a grounded answer comes out.

```
ticket ─► memory_retrieval ─► classify_ticket (HF zero-shot, no LLM tokens)
                                   │
          ┌────────────────────────┼───────────────────────────┐
       billing                technical / general           research
          │                        │                           │
     billing_node            supervisor_node            supervisor_node
          │                  (registry dispatch)        ── Send API fan-out ──┐
   refund ≥ $500?                  │                     kb_worker   web_worker
   └► approval_check (HITL)        │                          └─ research_collect ─┘
          └────────────────────────┴───────────┬───────────────┘
                                            respond (token streaming)
                                               │
                                          memory_write ─► END
```

Topology lives in [`graph.py`](graph.py); node implementations live in `agents/`, `rag/`,
`security/`, `workflows/`.

## Quick start

Requires Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env          # set GROQ_API_KEY (required); the rest are optional
python -m rag.ingest          # build the FAISS index from data/kb/ (first run downloads HF models)
python main.py                # interactive CLI; Ctrl+C to exit
```

HTTP API (FastAPI; JWT required unless `JWT_SECRET` is unset):

```bash
uvicorn api.routes:create_app --factory --reload
# GET /health · POST /tickets (SSE) · GET /metrics/{thread_id} · POST /auth/token
```

Tests and lint:

```bash
pytest --ignore=tests/e2e
ruff check .
```

> `Settings` requires `GROQ_API_KEY` at construction time, and some integration tests build
> the real graph, so the suite needs it set (any non-empty value works offline).

## Configuration

All config goes through `core/settings.py` (pydantic-settings; secrets are `SecretStr`).
See [`.env.example`](.env.example). Key switches:

| Variable | Purpose |
|----------|---------|
| `LLM_PROVIDER` | `groq` (default) · `huggingface` · `hf_local` — resolved in `llm_provider.py` only |
| `CHECKPOINTER_BACKEND` | `sqlite` (default, persisted to `SQLITE_PATH`) or `memory` |
| `GRAPH_DEBUG` | enables LangGraph debug mode |
| `JWT_SECRET` | unset = open-access dev mode; set = JWT required on the API |
| `LANGCHAIN_TRACING_V2` / `LANGSMITH_API_KEY` | LangSmith tracing |

## Project layout

| Path | Layer | Contents |
|------|-------|----------|
| `core/` | Infrastructure | settings, exceptions, factories, `GraphBuilder`, logging |
| `llm_provider.py` | Infrastructure | provider adapter: `get_llm`, `get_router_llm`, `get_embeddings`, `get_classifier`, `get_reranker`, `with_retry` |
| `state.py` | — | `AgentState` TypedDict, reducers, type aliases |
| `memory/` | Service | long-term semantic memory (protocols + store) |
| `rag/` | Service | CRAG subgraph, graders, retriever/reranker, ingestion, chunking, FAISS store, evaluator |
| `security/` | Service | JWT auth, RBAC, PII compliance gate |
| `tools/` | Service | Tavily search and KB tools (Pydantic `args_schema`) |
| `workflows/` | Service | approval (HITL), escalation, SLA |
| `observability/` | Service | hooks, audit events, metrics, LangSmith tracing |
| `agents/` | Graph | `BaseAgent` ABC, registry, supervisor, specialist agents, shared nodes |
| `graph.py` | Graph | main graph topology |
| `main.py`, `api/` | Delivery | CLI and FastAPI |
| `tests/` | — | `unit/`, `integration/`, `rag/`, `security/`, `e2e/` |
| `data/kb/` | — | sample knowledge-base documents |

Engineering conventions (clean layers, SOLID, node template, error handling, logging, testing)
are specified in [`CLAUDE.md`](CLAUDE.md), which also defines the curriculum for this project.

## Documentation

| Doc | What it is |
|-----|-----------|
| [`docs/LangGraphFoundations.md`](docs/LangGraphFoundations.md) | Teaching write-up per concept (what / why / when / pitfalls / production pattern / tests), Modules 1–10 |
| [`LANGGRAPH_NOTES.md`](LANGGRAPH_NOTES.md) | Progress tracker (concept → files → tests) plus condensed per-concept reference notes |
| [`about.MD`](about.MD) | Product-level overview: architecture, agents, feature set |
| [`CHANGELOG.md`](CHANGELOG.md) | Release history |

To find the code for a concept, use the progress tracker at the top of `LANGGRAPH_NOTES.md`:
each row names the implementing file and the test that exercises it.

## Roadmap

Not yet implemented (concept numbers from the curriculum in `CLAUDE.md`):

54, 56–59 (interrupt_after, Command family, agent handoffs) · 61, 63, 64, 66–68, 72
(review/compliance workflows, feedback loops, approval/escalation chains, workflow versioning) ·
77–81, 83 (worker graphs, shared-state mapping, supervisor library, hierarchical agents,
handoff via `Command(goto)`) · 87–88 (parallel staggering, map-reduce) ·
92, 96–98 (multi-query, Self-RAG, Adaptive RAG, Agentic RAG) · 103–105, 107–108 (hybrid search,
metadata filtering, secure retrieval, RAG tracing, tuning).

## Known limitations

- **Long-term memory is in-process.** `memory/store.py` wraps LangGraph's `InMemoryStore`, so
  memories are lost on restart. Conversation state *is* persisted by the SQLite checkpointer.
- **Postgres checkpointer is documented but not implemented.** `checkpointer_backend` accepts only
  `memory` and `sqlite`; `main.py` also opens `AsyncSqliteSaver` directly rather than using
  `core.factories.get_checkpointer()`.
- **CRAG subgraph is not wired in.** `rag.get_rag_graph()` is built and tested, but the research
  path's `kb_worker_node` calls the retriever directly.
- **Observer hooks are not wired in.** `node_with_hooks()` / `configure_hooks()` are implemented and
  tested but not applied to the running graph, so the metrics post-hook doesn't fire in the app.
- `memory_summarize_after_turns` is unused; `memory_write_node` always writes.
- `create_react_agent` / `ToolNode` are not used; Concepts 16–17 cover the tool classes and the
  `BaseAgent` ABC.
- `main.py` uses `print()` for CLI output; the rest of the app uses `logging`.
- Web research needs `TAVILY_API_KEY`; without it the web worker logs a warning and returns no results.
- `mypy` strict mode is configured but was not run for this release (mypy isn't installed in the venv).
- Groq free-tier rate limits apply; `with_retry` backs off with exponential wait.
