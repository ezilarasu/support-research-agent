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

## Try it

Start the CLI (`python main.py`) and paste tickets at the `You:` prompt. Routing uses a zero-shot
classifier, so a borderline ticket may land on a different label than listed here.

**Technical** — supervisor → technical agent, answered from `data/kb/technical_troubleshooting.md`
- `I keep getting 401 Unauthorized when calling your API, what should I check?`
- `My requests are failing with 429 Too Many Requests. How should I handle rate limits?`
- `Webhook deliveries to my server keep failing. How do I debug this?`
- `The Python SDK install fails with a dependency error.`

**General** — supervisor → general agent, answered from `data/kb/general_faq.md`
- `How do I reset my password?`
- `What's included in each plan and how do I upgrade?`
- `Where is my data stored and is it encrypted?`
- `What is your uptime SLA?`

**Billing, no approval needed**
- `I was charged twice for my invoice this month, can you help?`
- `I'd like a refund of $50 for an accidental charge.`

**Billing with human approval (HITL)** — refunds of **$500 or more** pause at `approval_check`
and the CLI prompts `Decision:`. Try approving once and rejecting once to see both branches.
- `Please refund $750 for my annual subscription, I was charged by mistake.`
- `I need a refund of $500 for a duplicate payment.`

**Research** — KB worker and web worker run in parallel (Send API fan-out). Without
`TAVILY_API_KEY` the web worker returns nothing, so you get the KB side only.
- `Research the best practices for designing API rate limiting and compare them with our rate limit policy.`
- `What are current industry standards for webhook retry and backoff strategies?`

**Follow-ups** — turns in one CLI run share a thread, so context carries over:
`How do I upgrade my plan?` then `And will I be charged immediately?`
(Long-term memory is in-process and resets on restart; see [Known limitations](#known-limitations).)

**Edge cases**
- `hello` or `it's broken` — low-confidence classification falls back to `general`.
- `refund me 600 dollars` — exercises refund-amount parsing without a `$` sign.

**Via the API** (no token needed when `JWT_SECRET` is unset; each request gets a new thread ID):

```bash
uvicorn api.routes:create_app --factory
curl -N -X POST localhost:8000/tickets -H 'Content-Type: application/json' \
  -d '{"ticket": "How do I reset my password?"}'
# add "stream": false to get JSON back: {"thread_id": ..., "response": ...}
```

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
  memories are lost on restart. Conversation state *is* persisted by the SQLite checkpointer, but the CLI starts a new
  thread on every launch and the API assigns a new thread ID per request, so there is no
  user-facing way to resume an earlier thread yet.
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
