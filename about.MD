What It Is

  A production-grade multi-agent customer support and research assistant built on LangGraph. It accepts support tickets via CLI (and eventually a FastAPI endpoint), intelligently routes them to
  specialist AI agents, retrieves knowledge, and generates grounded responses — all with full persistence, streaming, human oversight, and enterprise security.

  Think of it as the AI backend for a company's Tier-1 support desk.

  ---
  Architecture Overview

                          ┌─────────────────────────────────────┐
                          │         CLI / FastAPI (main.py)      │
                          └──────────────┬──────────────────────┘
                                         │ ticket text
                                         ▼
                          ┌─────────────────────────────────────┐
                          │     HuggingFace Zero-Shot Classifier │  ← no LLM tokens
                          │     (facebook/bart-large-mnli)       │  ← runs locally
                          └──────────────┬──────────────────────┘
                                         │ ticket_type + confidence
                                         ▼
                          ┌─────────────────────────────────────┐
                          │           LangGraph Router           │
                          └──┬──────────┬───────────┬───────────┘
                             │          │           │           │
                        billing    technical    general    research
                             │          │           │           │
                      ┌──────▼──┐ ┌────▼────┐ ┌───▼───┐ ┌────▼──────┐
                      │Billing  │ │Technical│ │General│ │Research   │
                      │Agent    │ │Agent    │ │Agent  │ │Agent      │
                      │(Groq)   │ │(Groq)   │ │(Groq) │ │(Groq+RAG) │
                      └────┬────┘ └────┬────┘ └───┬───┘ └────┬──────┘
                           └──────────┴───────────┴──────────┘
                                         │
                                         ▼
                          ┌─────────────────────────────────────┐
                          │          respond_node                │
                          │    (formats final response)          │
                          └─────────────────────────────────────┘

  ---
  Full Feature Set (10 Modules)

  ┌────────┬────────────────────────────────────────┬─────────┐
  │ Module │                Feature                 │ Status  │
  ├────────┼────────────────────────────────────────┼─────────┤
  │ 1      │ Graph skeleton, routing, CLI           │ ✅ Done │
  ├────────┼────────────────────────────────────────┼─────────┤
  │ 2      │ Reducer-safe state, chat history       │ ✅ Done │
  ├────────┼────────────────────────────────────────┼─────────┤
  │ 3      │ Tool use, hooks, runtime config        │ ✅ Done │
  ├────────┼────────────────────────────────────────┼─────────┤
  │ 4      │ SQLite persistence, resumable sessions │ ✅ Done │
  ├────────┼────────────────────────────────────────┼─────────┤
  │ 5      │ Long-term semantic memory (in-process) │ ✅ Done │
  ├────────┼────────────────────────────────────────┼─────────┤
  │ 6      │ Real-time token streaming (CLI + SSE)  │ ✅ Done │
  ├────────┼────────────────────────────────────────┼─────────┤
  │ 7      │ Human-in-the-loop approval flows       │ ✅ Done │
  ├────────┼────────────────────────────────────────┼─────────┤
  │ 8      │ Supervisor + worker multi-agent system │ ✅ Done │
  ├────────┼────────────────────────────────────────┼─────────┤
  │ 9      │ Full RAG pipeline (FAISS + CRAG)       │ ✅ Done │
  ├────────┼────────────────────────────────────────┼─────────┤
  │ 10     │ JWT auth, RBAC, LangSmith tracing      │ ✅ Done │
  └────────┴────────────────────────────────────────┴─────────┘

  ---
  The Four Specialist Agents

  1. Billing Agent

  - Handles: invoice disputes, refunds, double charges, subscription changes
  - Tools: order lookup, refund initiation, billing history
  - HITL: refunds over $500 require human approval before processing
  - LLM: gemma2-9b-it (fast, low-cost for structured tasks)

  2. Technical Agent

  - Handles: API errors, integration issues, authentication failures, rate limits
  - Tools: status page lookup, API docs search, error code lookup
  - RAG: searches internal technical knowledge base (FAISS)
  - LLM: llama-3.1-8b-instant (fastest, tool-calling optimised)

  3. General Agent

  - Handles: business hours, pricing questions, onboarding, policies
  - Tools: FAQ search, policy document retrieval
  - Memory: retrieves past interactions for this user/tenant
  - LLM: gemma2-9b-it

  4. Research Agent

  - Handles: compliance questions (GDPR, HIPAA), deep technical research, vendor comparisons
  - Tools: Tavily live web search, internal KB, academic sources
  - Pipeline: CRAG (Corrective RAG) — grades retrieved docs, falls back to web search if KB insufficient
  - LLM: llama-3.3-70b-versatile (most capable, for synthesis and reasoning)

  ---
  Data & Knowledge Base

  What the RAG system searches over:

  Internal KB documents (FAISS index — built in Module 9):
  data/kb/
  ├── billing_policies.md        ← refund rules, SLA commitments, dispute process
  ├── technical_docs/
  │   ├── api_reference.md       ← endpoints, auth, rate limits, error codes
  │   ├── integration_guides.md  ← webhooks, SDKs, common patterns
  │   └── troubleshooting.md     ← known issues, error resolution steps
  ├── compliance/
  │   ├── gdpr_policy.md         ← data retention, right to erasure, DPA
  │   ├── hipaa_guidelines.md    ← PHI handling, BAA requirements
  │   └── soc2_controls.md       ← security controls, audit procedures
  └── general/
      ├── faq.md                 ← common questions, business info
      └── onboarding.md          ← getting started, first steps

  Live web search (Tavily — used by Research Agent when KB is insufficient):
  - Regulatory updates (new GDPR guidance, compliance rulings)
  - External API documentation
  - Industry benchmarks and comparisons

  Long-term memory (LangGraph InMemoryStore, in-process — Module 5):
  - Stores past ticket resolutions per user/tenant
  - Semantically searched: "did this user report this issue before?"
  - Summarised after 10 turns to prevent context bloat

  ---
  State That Flows Through the Graph

  Every node reads and writes from AgentState — a 21-field TypedDict:

  messages              ← full chat history (add_messages reducer)
  ticket_type           ← billing | technical | general | research
  ticket_metadata       ← ticket_id, tenant_id, user_id, priority, SLA deadline
  classification_confidence  ← 0.0–1.0 from HF classifier
  thread_id             ← session UUID (checkpointer key)
  retrieved_memories    ← past interactions for this user
  retrieved_docs        ← documents from RAG retrieval
  retrieval_grade       ← relevant | irrelevant | partial
  hallucination_check   ← True = grounded, False = hallucination detected
  pending_approval      ← action waiting for human sign-off
  approval_decision     ← approved | rejected | escalated
  escalation_level      ← 0=L1, 1=L2, 2=L3
  authenticated         ← JWT verified
  rbac_roles            ← ["agent", "billing_approver", ...]
  audit_events          ← immutable log of every sensitive action
  final_response        ← the formatted reply to send to the user
  error                 ← set if anything fails

  ---
  What "Production-Grade" Means Here

  ┌───────────────────────────┬─────────────────────────────────────────────────────────────────────────────────────┐
  │          Concern          │                                   Implementation                                    │
  ├───────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
  │ No token waste on routing │ HuggingFace zero-shot runs locally — classification costs zero Groq tokens          │
  ├───────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
  │ Rate limit resilience     │ @with_retry (tenacity) on every LLM-calling node                                    │
  ├───────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
  │ Resumable sessions        │ SQLite checkpointer — kill the process, restart, same conversation                  │
  ├───────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
  │ Audit trail               │ Every sensitive action appended to audit_events (immutable reducer)                 │
  ├───────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
  │ Human oversight           │ interrupt_before pauses graph, waits for human Command(resume=...)                  │
  ├───────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
  │ Observability             │ LangSmith traces every node, token usage, latency per turn                          │
  ├───────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
  │ Multi-tenancy             │ tenant_id in every log and state field — tenant data never crosses                  │
  ├───────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
  │ Cost control              │ Cheap model for routing (gemma2-9b-it), expensive only for research (llama-3.3-70b) │
  └───────────────────────────┴─────────────────────────────────────────────────────────────────────────────────────┘

  ---
  What You Can Do With It After All 10 Modules

  # Send a billing ticket → get a response + refund initiated (with HITL approval)
  You: My order #8821 was charged twice. I want my money back.
  Agent [billing • billing • conf=94%]: I can see the duplicate charge...
    [HITL] Refund of $129.99 requires approval. Approve? (y/n): y
  Agent: Refund approved and submitted. Reference: REF-2026-4421.

  # Send a research ticket → get a GDPR-grounded answer from live web + KB
  You: Does GDPR Article 17 require us to delete backups within 30 days?
  Agent [research • research • conf=88%]: Based on GDPR guidance and recent
    DPA rulings... [grounded citations from KB + Tavily]

  # Resume a previous session (Module 4)
  SUPPORT_THREAD_ID=abc-123 .venv/bin/python main.py
  # → full conversation history restored from SQLite