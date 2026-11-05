# Module 1 — LangGraph Foundations
*Concepts 1–8 · Session 1 · ~2.5 hours*
*Unlocks: working skeleton app — CLI accepts a ticket, graph runs end-to-end*

---

## 📌 CONCEPT 1: StateGraph

### 🔍 WHAT

A `StateGraph` is LangGraph's core orchestration primitive — a directed graph where every node is a Python function and every edge is a transition rule. You define the shape of your application's shared memory (the *state*), then wire up functions that read from and write to that state as the graph executes.

**Mental model:** Think of it like a railway switchyard. Trains (data) arrive at a station (node), get processed, then are routed along tracks (edges) to the next station. The entire yard's layout — which tracks connect which stations — is defined once at build time. Once the train is moving, the layout is fixed; only the routing decisions change based on cargo.

**Where it lives:** `StateGraph` is the root object. Everything else — nodes, edges, checkpointers, streaming — attaches to it. It lives in the Graph Layer of the clean architecture.

---

### 🤔 WHY

**The problem it solves:** Before LangGraph, multi-step LLM workflows were tangled chains of function calls. Want to branch? Add an `if`. Want to loop? Add a `while`. Want to pause for human input? Refactor everything. The control flow was baked into Python's call stack — invisible, untestable, non-resumable.

**What breaks without it:**

```python
# ❌ BEFORE LangGraph — spaghetti workflow
def handle_ticket(ticket: str) -> str:
    ticket_type = classify(ticket)          # what if this fails?
    if ticket_type == "billing":
        result = billing_handler(ticket)    # what if we need to pause here for approval?
    elif ticket_type == "technical":
        result = tech_handler(ticket)       # can't resume if process crashes mid-way
    else:
        result = general_handler(ticket)
    return result                           # no memory, no audit trail, no time travel
```

Problems:
- No persistence — crash at line 4, lose everything
- No branching visibility — control flow is invisible to tooling
- No pause/resume — can't interrupt for human approval mid-flow
- No parallelism — sequential only, hardcoded
- No observability — can't see what happened inside

**In our app:** The support ticket system has 4 routing lanes, HITL approval, RAG retrieval, and streaming — none of that is expressible as a flat function chain. `StateGraph` lets us declare the topology once, then LangGraph handles execution, persistence, and resumption.

---

### 📍 WHEN TO USE

**Use StateGraph when:**
- ✅ Workflow has branching logic (different paths for different ticket types)
- ✅ You need persistence / resumability (crash recovery, HITL pause)
- ✅ Multiple agents need shared state
- ✅ Cycles are possible (retry loops, self-correction)
- ✅ You need streaming, checkpointing, or time travel

**Don't use StateGraph when:**
- ❌ Simple single-step LLM call — just call the LLM directly
- ❌ Pure data transformation pipeline with no branching
- ❌ You only need LangChain LCEL chains (`chain = prompt | llm | parser`)

---

### 🌍 WHERE IT FITS

```
                    ┌─────────────────────────────┐
                    │      StateGraph              │
                    │   (root of everything)       │
                    └─────────────┬───────────────┘
                                  │
              ┌───────────────────┼────────────────────┐
              ▼                   ▼                     ▼
         add_node()           add_edge()          compile()
        (Concept 4)          (Concept 5)          (Concept 7)
              │                   │                     │
              ▼                   ▼                     ▼
      Python async fn      src → dst rule      CompiledGraph
      reads/writes state   conditional edge    + checkpointer
                                               = runnable app
```

`StateGraph` is defined → compiled → invoked. You only define it once. After `compile()`, the graph is immutable.

---

### ⚖️ PROS & CONS

| Pros | Cons |
|------|------|
| Explicit, visualisable topology | More setup than a plain function |
| Built-in persistence and resumption | Requires thinking in graph terms upfront |
| Native streaming, HITL, time travel | State schema must be designed carefully (Module 2) |
| Parallel execution with Send API | Recursion limit must be managed |
| Full type safety with TypedDict state | LangGraph version updates occasionally break APIs |

---

### 🔀 MULTIPLE APPROACHES

**Approach 1: `StateGraph`** — use this always
- Any multi-step, stateful, branching LLM workflow
- Full LangGraph feature set — persistence, streaming, HITL, time travel
- Trade-off: slightly more upfront structure

**Approach 2: `MessageGraph`** — legacy, avoid
- Simple chatbot with only message history as state
- State is locked to `list[BaseMessage]` — can't add custom fields
- Superseded by `StateGraph` with `add_messages` reducer (Concept 12)

**Approach 3: `Graph`** — low-level, rarely needed
- Maximum flexibility, but loses type safety and LangGraph state management

---

### 🆚 DIFFERENCES & COMPARISONS

| | `StateGraph` | LangChain LCEL Chain |
|---|---|---|
| Execution model | Graph with explicit topology | Linear pipe (`\|` operator) |
| Branching | Native conditional edges | Requires `RunnableBranch` hack |
| Cycles / loops | Native | Not supported |
| Persistence | Built-in with checkpointer | None |
| HITL | Native `interrupt()` | Not supported |
| Streaming | Native per-node | Per-chain only |
| **Use for** | Multi-step agents | Single-step chains |

**Groq vs HuggingFace:** `StateGraph` is provider-agnostic — topology is identical regardless of provider.

---

### 💡 BEST PRACTICES

1. **Define state schema before the graph.** `state.py` is written before `graph.py`. Every node depends on the contract it defines.
2. **Use `GraphBuilder` from day one.** Centralises all `add_node`/`add_edge` calls — prevents scattered topology across files.
3. **Never modify after `compile()`.** The compiled graph is immutable. Build a new graph if topology must change.
4. **One `StateGraph` = one application boundary.** Sub-workflows are subgraphs (Module 8), not a second top-level graph.
5. **Most common mistake:** Using sync `graph.invoke()` when any node is `async def`. Fix: `await graph.ainvoke()` inside `asyncio.run()`.

---

### 🏗️ PRODUCTION PATTERN

```python
# graph.py — always build via GraphBuilder
from langgraph.graph import END
from core.graph_builder import GraphBuilder
from state import AgentState

def build_graph():
    return (
        GraphBuilder(AgentState)
        .node("classify_ticket", classify_ticket_node)
        .node("respond", respond_node)
        .edge("classify_ticket", "respond")
        .edge("respond", END)
        .entry("classify_ticket")
        .build()                    # no checkpointer yet — added in Module 4
    )

# main.py — async-first, always ainvoke for async nodes
import asyncio
result = await graph.ainvoke(initial_state, config=config)
```

---

### ✅ CONCEPT CHECKPOINT

**Three questions to answer confidently:**
1. Why does `graph.invoke()` fail with async nodes? What is the fix?
2. What problem does `GraphBuilder` solve over calling `StateGraph` methods directly?
3. Why is `core/settings.py` created before `graph.py`?

**Hands-on exercise:** Add a third node `log_ticket` between `classify_ticket` and `respond`. It should log the ticket type and return `{"audit_events": [{"node": "log_ticket", "ticket_type": state["ticket_type"]}]}`.

---

---

## 📌 CONCEPT 2: State

### 🔍 WHAT

**State** is the shared memory of your graph — a `TypedDict` that every node reads from and writes to. It is the single source of truth for everything happening in a graph execution: conversation history, the current ticket type, retrieved documents, authentication status.

**Mental model:** State is like a shared whiteboard in a hospital operating room. Each surgeon (node) reads what's already written, does their work, then writes only their results. No surgeon passes data directly to another — they all communicate through the whiteboard. When the operation finishes (`END`), the whiteboard contains the complete record.

**Where it lives:** `state.py` at the project root. Infrastructure layer — no logic, only the data contract. Every other layer imports from it; it imports from nothing internal.

---

### 🤔 WHY

**The problem:** Nodes need to share data. Without a typed state, you pass raw `dict` between functions — no autocomplete, no validation, silent `KeyError` bugs at runtime.

**What breaks without typed state:**

```python
# ❌ WITHOUT typed state — undetectable typos
async def classify_node(state: dict, config) -> dict:
    ticket = state["mesages"][-1].content   # typo — KeyError at runtime
    return {"tiket_type": "billing"}        # typo — silently dropped, never read

async def respond_node(state: dict, config) -> dict:
    ticket_type = state["ticket_type"]      # KeyError — key was never set correctly
    ...
```

**With `TypedDict`:**
```python
# ✅ WITH typed state — caught at analysis time by mypy
async def classify_node(state: AgentState, config: RunnableConfig) -> dict:
    ticket = state["messages"][-1].content  # IDE autocompletes, mypy validates
    return {"ticket_type": "billing"}       # mypy checks this key exists in AgentState
```

---

### 📍 WHEN TO USE

- ✅ Always — every `StateGraph` must have a state schema
- ✅ Add a new field when a new node introduces new data
- ✅ Use nested `TypedDict` for structured sub-objects (never raw `dict`)
- ❌ Don't put logic in state — it's a data container
- ❌ Don't use `Any` typed fields — defeats type safety

---

### 🌍 WHERE IT FITS

```
graph.py                     state.py                      agents/nodes.py
────────────                 ─────────────────────         ───────────────────────────
StateGraph(AgentState)  ◄──  class AgentState(TypedDict):  async def classify_node(
    │                            messages: ...                  state: AgentState,   ← reads
    │                            ticket_type: ...               config: RunnableConfig
    │                            final_response: ...        ) -> dict:               ← writes partial
    └── uses as schema           ...                            return {"ticket_type": "billing"}
```

---

### ⚖️ PROS & CONS

| Pros | Cons |
|------|------|
| Full IDE autocomplete on every field | Requires upfront schema design |
| mypy catches typos at analysis time | New fields require touching state.py |
| Checkpointing serialises typed state safely | TypedDict doesn't validate at runtime by default |
| Reducers (Module 2) work with typed fields | Deep nesting requires nested TypedDicts |

---

### 🔀 MULTIPLE APPROACHES

**Approach 1: `TypedDict`** — use this always
- Lightweight, no runtime overhead, full mypy support
- LangGraph's native state type — all tooling built around it

**Approach 2: Pydantic `BaseModel`**
- Runtime validation + coercion — heavier, but catches bad data at node boundaries
- When: state received from external sources (HTTP, queues) needing validation
- Requires `model_config = ConfigDict(arbitrary_types_allowed=True)`

**Approach 3: `dataclass`**
- Mutable, familiar — but loses `Annotated` reducer support. Not recommended.

---

### 🆚 DIFFERENCES & COMPARISONS

| | `TypedDict` | Pydantic `BaseModel` | `dataclass` |
|---|---|---|---|
| Runtime validation | ❌ | ✅ | ❌ |
| mypy support | ✅ | ✅ | ✅ |
| `Annotated` reducers | ✅ | ✅ (with config) | ❌ |
| Serialisation overhead | None | Low | None |
| **Use for** | Internal graph state | External input validation | Not in LangGraph |

---

### 💡 BEST PRACTICES

1. **Write `state.py` first — before any graph or node code.** It is the contract everything else depends on. Changing it mid-build is expensive.
2. **Use nested `TypedDict` for structured fields.** `TicketMetadata` and `RetrievedDocument` are TypedDicts, not `dict`. This enables autocomplete and mypy all the way down.
3. **Comment every field with which Module introduces it.** State grows across all 10 modules — comments make it clear when each field becomes active.
4. **Use `str | None` with a `None` default for fields not yet populated.** This lets you pass initial state without setting every field.
5. **Most common mistake:** Returning the full `AgentState` object from a node instead of a partial dict. LangGraph merges partial dicts — returning the full state can overwrite other nodes' work. Always return only what you changed.

---

### 🏗️ PRODUCTION PATTERN

```python
# state.py — the complete contract
from __future__ import annotations
import operator
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class TicketMetadata(TypedDict):
    ticket_id: str
    tenant_id: str
    user_id: str
    priority: str
    created_at: str
    sla_deadline: str

class RetrievedDocument(TypedDict):
    content: str
    source: str
    score: float
    metadata: dict

class AgentState(TypedDict):
    # Reducers explained in Module 2 — for now: Annotated = append, plain = replace
    messages: Annotated[list[BaseMessage], add_messages]   # appends, deduplicates
    ticket_type: str | None                                # replaced on update
    ticket_metadata: TicketMetadata | None                 # replaced on update
    classification_confidence: float
    thread_id: str | None
    retrieved_memories: Annotated[list[str], operator.add] # appends
    retrieved_docs: Annotated[list[RetrievedDocument], operator.add]  # appends
    agent_results: Annotated[list[dict], operator.add]     # appends
    audit_events: Annotated[list[dict], operator.add]      # append-only audit log
    rbac_roles: Annotated[list[str], operator.add]         # appends
    authenticated: bool
    escalation_level: int
    streaming_enabled: bool
    active_agent: str | None
    query: str | None
    retrieval_grade: str | None
    hallucination_check: bool | None
    pending_approval: dict | None
    approval_decision: str | None
    final_response: str | None
    error: str | None

# Nodes always return PARTIAL dicts:
async def classify_node(state: AgentState, config: RunnableConfig) -> dict:
    return {"ticket_type": "billing"}   # ✅ only what this node changes
    # NOT: return state                 # ❌ never return full state
```

---

### ✅ CONCEPT CHECKPOINT

**Three questions to answer confidently:**
1. Why does `state.py` use `TypedDict` instead of a plain `dict` or a Pydantic model?
2. What is the difference between an `Annotated[list[T], operator.add]` field and a plain `list[T]` field in state? (Full answer in Module 2 — but explain your current understanding.)
3. What happens if a node returns the full `AgentState` instead of a partial dict?

**Hands-on exercise:** Add a new field `response_language: str | None` to `AgentState` (default `None`). Update the `respond_node` in `graph.py` to set it to `"en"`. Confirm the smoke test still passes.

---

---

## 📌 CONCEPT 3: TypedDict

### 🔍 WHAT

`TypedDict` is Python's way of giving a plain `dict` a fixed schema — specific keys with specific types — checked by mypy and IDEs at analysis time with **zero runtime overhead**.

**Mental model:** A `TypedDict` is like a form with labeled fields. A plain `dict` is a blank piece of paper. You can write anything on blank paper, but you can't tell at a glance what's missing or misspelled. The form forces structure — every field has a name, a type, and a presence expectation.

**Where it lives:** `state.py` — `AgentState`, `TicketMetadata`, `RetrievedDocument` are all TypedDicts. Also used for structured return types from factory functions and node helpers.

---

### 🤔 WHY

**What breaks without it:**

```python
# ❌ Plain dict — three silent failures, zero warnings
async def billing_node(state: dict, config) -> dict:
    amount = state["refund_ammount"]          # typo — KeyError at runtime
    return {"refund_amount": amount * 1.1}    # wrong key — silently dropped
    return {"escalation_level": "high"}       # wrong type — crashes 3 nodes later
```

```python
# ✅ TypedDict — all three caught before running a single line
async def billing_node(state: AgentState, config: RunnableConfig) -> dict:
    amount = state["refund_ammount"]           # mypy: key not found in AgentState
    return {"refund_amount": amount * 1.1}     # mypy: key not in AgentState
    return {"escalation_level": "high"}        # mypy: str not assignable to int
```

---

### 📍 WHEN TO USE

- ✅ LangGraph state schema — the primary use case
- ✅ Structured return types from factory functions (`RetrievalResult`, `GradeResult`)
- ✅ Any dict with a fixed known shape passed between functions
- ❌ Dicts with dynamic/unknown keys — use `dict[str, Any]`
- ❌ Dicts needing runtime validation — use Pydantic `BaseModel`
- ❌ Objects needing methods — use `dataclass` or regular `class`

---

### ⚖️ PROS & CONS

| Pros | Cons |
|------|------|
| Zero runtime overhead | No runtime validation |
| Full mypy + IDE support | `total=False` semantics can be confusing |
| Works with `Annotated` for reducers | Can't add methods |
| Serialises trivially (IS a dict) | Deep nesting requires multiple TypedDict definitions |

---

### 🔀 MULTIPLE APPROACHES

**`total=True` (default):** All fields required. Use for `AgentState`, `TicketMetadata`, `RetrievedDocument`.

**`total=False`:** All fields optional. Useful for partial update shapes.

**Mixed inheritance:**
```python
class _Required(TypedDict, total=True):
    messages: list[BaseMessage]       # always present

class AgentState(_Required, total=False):
    ticket_type: str                  # optional — populated later
```
We use `str | None` defaults instead — simpler and more explicit.

---

### 🆚 DIFFERENCES & COMPARISONS

| | `TypedDict` | Pydantic `BaseModel` | `dataclass` | plain `dict` |
|---|---|---|---|---|
| Runtime validation | ❌ | ✅ | ❌ | ❌ |
| mypy support | ✅ | ✅ | ✅ | ❌ |
| `Annotated` reducers | ✅ | ✅ (with config) | ❌ | ❌ |
| IS a dict at runtime | ✅ | ❌ | ❌ | ✅ |
| Methods | ❌ | ✅ | ✅ | ❌ |
| Runtime overhead | None | Low–Med | None | None |
| **LangGraph state** | ✅ Best | Works | ❌ | ❌ |

---

### 💡 BEST PRACTICES

1. **One TypedDict per logical concept.** `TicketMetadata` groups related ticket fields; `RetrievedDocument` groups RAG result fields. Don't flatten everything into `AgentState`.
2. **`TypeAlias` for repeated literal types.** Define `TicketType`, `AgentName`, `GradeLabel` once in `state.py` — import everywhere else.
3. **`__all__` in every `__init__.py` and in `state.py`.** Declares the public API explicitly. Prevents accidental use of internal helpers.
4. **`from __future__ import annotations` at the top of every file.** Enables forward references and Python 3.9 compatibility with complex generics.
5. **Most common mistake:** Using `dict` for structured sub-objects inside state. `pending_approval: dict | None` is a code smell — define `PendingApproval(TypedDict)` instead (Module 7).

---

### 🏗️ PRODUCTION PATTERN

```python
# state.py — TypeAliases + __all__ + TypedDicts
from __future__ import annotations
from typing import Annotated, Literal, TypeAlias, TypedDict

# TypeAliases — define once, import everywhere
TicketType: TypeAlias = Literal["billing", "technical", "general", "research"]
AgentName: TypeAlias = Literal["billing", "technical", "general", "research", "supervisor"]
GradeLabel: TypeAlias = Literal["relevant", "irrelevant", "partial"]

__all__ = ["AgentState", "TicketMetadata", "RetrievedDocument",
           "TicketType", "AgentName", "GradeLabel", ...]

# Nested TypedDicts
class TicketMetadata(TypedDict):
    ticket_id: str
    tenant_id: str
    priority: Priority          # uses TypeAlias — mypy-checked

# Main state uses TypeAliases in field types
class AgentState(TypedDict):
    ticket_type: TicketType | None      # not str | None — catches routing typos
    active_agent: AgentName | None
    retrieval_grade: GradeLabel | None
    approval_decision: ApprovalDecision | None
    ...
```

---

### 🧪 TESTS

```
tests/unit/test_state.py — 38 tests, all passing
  TestAgentStateFields        — 22 tests: every field present, exact count guarded
  TestTicketMetadataFields    — 1 test: correct field set
  TestRetrievedDocumentFields — 1 test: correct field set
  TestTypeAliases             — 5 tests: correct literal values per alias
  TestPublicAPI               — 6 tests: __all__ completeness
  TestTypedDictRuntimeBehaviour — 3 tests: IS a dict, constructable, partial dicts valid
```

---

### ✅ CONCEPT CHECKPOINT

**Three questions to answer confidently:**
1. `TypedDict` has zero runtime overhead — what does this mean? What does it *not* catch?
2. Why define `TicketType: TypeAlias = Literal["billing", "technical", "general", "research"]` instead of just using `str` for `ticket_type` in `AgentState`?
3. Why does `test_total_field_count` assert exactly 21 fields? What is the engineering value of that test?

**Hands-on exercise:** Add a new `TypeAlias` called `SupportChannel` with values `"email"`, `"chat"`, `"phone"`. Add a field `channel: SupportChannel | None` to `AgentState`. Add it to `__all__`. Add a test for it in `test_state.py`. Run the tests — confirm the field count assertion needs updating.

---

---

## 📌 CONCEPT 4: Nodes

### 🔍 WHAT

A **node** is any Python function registered with `GraphBuilder.node(name, fn)`. It receives the current state snapshot and a `RunnableConfig`, performs exactly one unit of work, and returns a **partial dict** of only the fields it changed. That's the complete contract.

**Mental model:** A node is a specialist on a manufacturing assembly line. The product (state) arrives, the specialist performs their one task, marks what they changed on the workorder (partial dict), and sends it on. They never touch parts belonging to another station and never do two jobs at once.

**Where it lives:** Node *functions* in `agents/nodes.py` (Graph layer). Node *registration* in `graph.py`. Topology and logic are always kept separate.

---

### 🤔 WHY

**What breaks without the template:**

```python
# ❌ 6 production violations in 12 lines
async def classify_ticket(state, config):            # no types
    try:
        llm = ChatGroq(api_key="sk-...")             # hardcoded provider
        result = await llm.ainvoke(state["messages"])
        state["ticket_type"] = result.content        # mutating state directly
        print(f"classified: {result.content}")       # print, not logger
        return state                                 # returning full state
    except Exception as e:                           # bare except
        print(f"error: {e}")
        return state
```

**With the template:** retried, typed, testable, observable, safe.

---

### 📍 WHEN TO USE

- ✅ Every step in your graph = one node (SRP)
- ✅ One node = one reason to change
- ❌ Don't combine two concerns in one node
- ❌ No Rich printing, no stdin/stdout — those belong in `main.py`
- ❌ No `ChatGroq(...)` constructions — those belong in `llm_provider.py`

---

### ⚖️ PROS & CONS

| Pros | Cons |
|------|------|
| Single responsibility → trivially testable | More files than a monolithic function |
| `@with_retry` → survives Groq rate limits | Requires discipline to maintain SRP |
| `_prefixed` helpers → unit-testable pure functions | Node signature is fixed — no extra params |
| Partial dict return → composable with other nodes | Forgetting `@with_retry` on one node breaks resilience |

---

### 💡 BEST PRACTICES

1. **`@with_retry` on every node that calls an LLM.** Non-negotiable for Groq free tier.
2. **Extract logic into `_prefixed` pure helpers.** Node orchestrates; helpers do the work.
3. **`logger.debug` on entry, `logger.info` on result.** Never `print()`.
4. **Read all inputs at the top.** Extract everything from `state` in the first 3 lines.
5. **Most common mistake:** `state["ticket_type"] = "billing"` — direct mutation. Always `return {"ticket_type": "billing"}`.

---

### 🏗️ PRODUCTION TEMPLATE

```python
# agents/nodes.py
from __future__ import annotations
import logging
from langchain_core.runnables import RunnableConfig
from llm_provider import get_research_llm, with_retry
from state import AgentState

logger = logging.getLogger(__name__)

@with_retry                              # 1. retry decorator — always first
async def classify_ticket_node(
    state: AgentState,                   # 2. typed state — never dict
    config: RunnableConfig,              # 3. typed config
) -> dict:                               # 4. returns partial dict
    """
    Classifies incoming ticket using HF zero-shot (0 tokens consumed).

    Reads:  state["messages"]
    Writes: state["ticket_type"], state["classification_confidence"]
    """
    # 5. Extract all inputs at top
    messages = state.get("messages", [])
    if not messages:
        raise ValueError("classify_ticket_node: received empty messages list")
    ticket_text = messages[-1].content
    thread_id = _get_thread_id(config)   # 6. safe config access

    logger.debug("classify_ticket_node: thread=%s", thread_id)  # 7. debug on entry

    # 8. Business logic via pure helper
    ticket_type, confidence = _classify_with_hf(ticket_text)

    logger.info("classify_ticket_node: thread=%s ticket_type=%s", thread_id, ticket_type)

    return {                             # 9. partial dict — only what this node changes
        "ticket_type": ticket_type,
        "classification_confidence": confidence,
    }

def _classify_with_hf(text: str) -> tuple[TicketType, float]:
    """Pure function — no LLM, no state, trivially unit-testable."""
    from llm_provider import get_classifier
    classifier = get_classifier()
    result = classifier(text, ["billing", "technical", "general", "research"])
    label, score = result["labels"][0], result["scores"][0]
    if score < 0.4:
        return "general", score
    return label, score
```

---

### 🧪 TESTS

```
tests/unit/test_nodes.py — 24 tests, all passing
  TestGetThreadId          — 4 tests: safe config extraction (no KeyError)
  TestBuildResponsePrompt  — 6 tests: prompt structure (pure function)
  TestClassifyWithHf       — 4 tests: classification + fallbacks (mocked HF)
  TestClassifyTicketNode   — 4 tests: node integration + partial dict contract
  TestRespondNode          — 6 tests: LLM call + AIMessage + partial dict contract
```

---

### ✅ CONCEPT CHECKPOINT

**Three questions to answer confidently:**
1. Why are `_prefixed` helpers pure functions? What makes them easier to test than the node function itself?
2. What happens if a node does `state["ticket_type"] = "billing"` instead of returning a partial dict? Why is this especially dangerous in Module 8 (parallel fan-out)?
3. Why is `@with_retry` the first decorator on every node, not the last?

---

---

## 📌 CONCEPT 5: Edges

### 🔍 WHAT

An **edge** is a directed connection between two nodes that tells LangGraph where to route execution after a node completes. An **unconditional edge** always routes from A to B — no conditions, no decisions.

Three special values used in edges:
- `START` — implicit source of the graph entry point
- `END` — terminal sentinel; reaching it stops execution
- Any registered node name — routes to that node

**Mental model:** Edges are train tracks between stations. An unconditional edge is a fixed track with no switch — every train leaving station A takes the same track to station B. `END` is the terminus.

**Where it lives:** `graph.py` exclusively, declared via `GraphBuilder.edge()`. Node files know nothing about edges — they just return a partial dict.

---

### 🤔 WHY

**What breaks without correct edges:**

```python
# ❌ Missing edge — graph compiles but halts after first node
GraphBuilder(AgentState)
    .node("classify_ticket", classify_ticket_node)
    .node("respond", respond_node)
    # No edges declared — classify_ticket runs, then graph terminates silently
    .entry("classify_ticket")
    .build()

# ❌ Typo in node name — caught at compile time
.edge("clasify_ticket", "respond")   # ValueError immediately at .build()
```

---

### 📍 WHEN TO USE

- ✅ Next node is always the same — no branching needed
- ✅ Every terminal node must have `.edge(node, END)`
- ❌ Branching based on state → use Conditional Edges (Concept 6)

---

### ⚖️ PROS & CONS

| Pros | Cons |
|------|------|
| Simple — no logic | Only works when destination is fixed |
| Invalid node names caught at compile time | Can't branch |
| Self-documenting topology | Easy to forget edge to END (silent hang) |

---

### 💡 BEST PRACTICES

1. **Every terminal node needs `.edge(node, END)`.** Missing END = silent hang.
2. **All edges declared in `graph.py`** — never scattered across node files.
3. **`START` explicit when you want a self-contained edge list.** Reads like a sentence: `START → classify → respond → END`.
4. **Test topology with an integration test.** `test_no_unexpected_nodes` and `test_classify_runs_before_respond` guard the wiring.
5. **Most common mistake:** Edge to a node that was never registered with `.node()`. LangGraph catches it at `build()` — but only if your test calls `build_graph()`.

---

### 🏗️ PRODUCTION PATTERN

```python
# graph.py — START/END sentinels explicit
from langgraph.graph import END, START
from core.graph_builder import GraphBuilder
from state import AgentState

def build_graph():
    return (
        GraphBuilder(AgentState)
        .node("classify_ticket", classify_ticket_node)
        .node("respond", respond_node)
        .edge("classify_ticket", "respond")   # unconditional
        .edge("respond", END)                 # terminal — must have this
        .entry("classify_ticket")             # implicit START → classify_ticket
        .build()
    )
```

### ⚠️ Production Bug Discovered and Fixed

`@with_retry` stripped the original function signature. LangGraph uses `inspect.signature()` to decide whether to pass `config` to a node. `wrapper(*args, **kwargs)` has no explicit `config` parameter — LangGraph called nodes with only `(state,)`.

**Fix:** `functools.update_wrapper(wrapper, func)` in `llm_provider.py` sets `__wrapped__ = func`, so `inspect.signature` follows the chain and sees the original `(state: AgentState, config: RunnableConfig)` signature.

```python
# llm_provider.py — with_retry fix
import functools

def with_retry(func):
    @retry(...)
    async def wrapper(*args, **kwargs):
        return await func(*args, **kwargs)
    functools.update_wrapper(wrapper, func)  # ← preserves signature for LangGraph
    return wrapper
```

---

### 🧪 TESTS

```
tests/integration/test_graph.py — 13 tests, all passing
  TestGraphCompilation  — 4 tests: build succeeds, invalid edges fail at compile
  TestGraphNodes        — 4 tests: correct nodes registered, count guarded
  TestGraphEdges        — 2 tests: entry + Mermaid topology verified
  TestGraphInvocation   — 3 tests: full path runs, execution order, reducers work
```

---

### ✅ CONCEPT CHECKPOINT

**Three questions to answer confidently:**
1. What happens when you add `.edge("a", "b")` but never registered `"b"` with `.node()`? When does the error surface?
2. Why does a missing `.edge(last_node, END)` cause a silent hang rather than an error?
3. What did `functools.update_wrapper` fix, and why does LangGraph need the original function signature?

**Hands-on exercise:** Change the graph to have three nodes: `classify_ticket → sanitize → respond → END`. `sanitize_node` strips whitespace from the last message and returns `{"messages": [HumanMessage(content=cleaned)]}`. Add an integration test verifying the execution order is `["classify", "sanitize", "respond"]`.

---

---

---

## 📌 CONCEPT 6: Conditional Edges

### 🔍 WHAT

A **conditional edge** routes execution to different nodes based on state. A **router function** reads state and returns a string key; a **mapping dict** translates that key to a destination node name.

**Mental model:** A conditional edge is a railway switch controlled by a signal box. The operator (router function) reads the train manifest (ticket_type) and throws the switch. The operator does no work — they only read and decide. The work happens at the next station.

**Where it lives:** Router *functions* in `agents/nodes.py` — pure sync functions. Registration in `graph.py` via `GraphBuilder.conditional_edge()`.

---

### 🤔 WHY

**What breaks without conditional edges:**

```python
# ❌ Routing logic inside a node — violates OCP
async def respond_node(state, config):
    if state["ticket_type"] == "billing":
        result = await _billing_logic(state)     # can't test independently
    elif state["ticket_type"] == "technical":    # adding "legal" touches this node
        result = await _technical_logic(state)
    ...
```

With conditional edges, adding `"legal"` means one new node + one new mapping entry. Existing nodes are never touched.

---

### 📍 WHEN TO USE

- ✅ Graph takes different paths based on state values
- ✅ Multiple downstream specialists
- ✅ Retry/fallback loops (CRAG in Module 9)
- ✅ HITL approval branch (Module 7)
- ❌ Next node is always the same → use unconditional `.edge()`
- ❌ Putting I/O or LLM calls in the router — it **must** be a pure sync function

---

### ⚖️ PROS & CONS

| Pros | Cons |
|------|------|
| Open/Closed — new routes don't touch existing nodes | Router must return a key in the mapping (else `InvalidUpdateError`) |
| Each branch tested independently | Router must be sync — async router silently returns coroutine |
| Enables loops (CRAG, self-correction) | More nodes = more topology |

---

### 💡 BEST PRACTICES

1. **Router functions are pure sync — never `async def`.** An async router returns a coroutine object (truthy), silently routing all tickets to the first mapped key.
2. **Always include a fallback key.** `route_ticket` returns `"general"` for any unrecognised `ticket_type`. An unmatched key = `InvalidUpdateError` at runtime.
3. **`RouterFn` TypeAlias on every router.** Documents the contract; mypy checks `(AgentState) -> str`.
4. **Import `ROUTE_KEYS` from `agents/nodes.py` into `graph.py`.** Single source of truth — router and topology share the same mapping dict.
5. **Most common mistake:** `async def route_ticket(...)`. No error at build time, no error at routing time — tickets silently go to the wrong branch.

---

### 🏗️ PRODUCTION PATTERN

```python
# agents/nodes.py
ROUTE_KEYS = {
    "billing": "billing_node", "technical": "technical_node",
    "general": "general_node", "research": "research_node",
}

def route_ticket(state: AgentState) -> str:    # SYNC — never async
    ticket_type = state.get("ticket_type")
    if ticket_type not in ROUTE_KEYS:
        return "general"                       # safe fallback — never crash
    return ticket_type

# graph.py
_route_ticket: RouterFn = route_ticket        # mypy checks the type
(
    GraphBuilder(AgentState)
    .node("classify_ticket", classify_ticket_node)
    .node("billing_node", billing_node)
    ...
    .conditional_edge("classify_ticket", _route_ticket, ROUTE_KEYS)
    .edge("billing_node", "respond")
    ...
    .entry("classify_ticket")
    .build()
)
```

---

### 🧪 TESTS

```
tests/unit/test_nodes.py (additions):
  TestRouteTicket     — 8 tests: all 4 routes, None fallback, unknown fallback,
                        sync-not-coroutine check, return value in ROUTE_KEYS
  TestHandlerNodes    — 6 tests: active_agent, audit_events, no-messages contract

tests/integration/test_graph.py (additions):
  TestConditionalRouting — 6 tests: parametrised 4-route test, audit trail,
                           low-confidence → general fallback
```

---

### ✅ CONCEPT CHECKPOINT

**Three questions to answer confidently:**
1. What happens at runtime if `route_ticket` is `async def`? Why does LangGraph not raise an error immediately?
2. Why is `ROUTE_KEYS` defined in `agents/nodes.py` and imported into `graph.py` rather than defined in both places?
3. What does `InvalidUpdateError` mean and when does LangGraph raise it?

**Hands-on exercise:** Add a `"legal"` route: create `legal_node` (stub), add `"legal": "legal_node"` to `ROUTE_KEYS`, wire it in `graph.py`, add a test to `TestConditionalRouting`. Confirm `test_no_unexpected_nodes` fails until you update the expected set.

---

---

## 📌 CONCEPT 7: Compile

### 🔍 WHAT

`StateGraph.compile()` transforms a mutable graph *definition* into an immutable, runnable `CompiledStateGraph`. After compile, the topology is locked — but the graph gains its full runtime machinery: Pregel execution engine, state merging via reducers, checkpointing, interrupt support, and streaming.

**Mental model:** `StateGraph` is a blueprint. `compile()` constructs the building. The blueprint can be revised freely. Once built, you can't move walls — but you can live in it.

**Where it lives:** `graph.py` → `build_graph()` calls `GraphBuilder.build()` which calls `StateGraph.compile(**kwargs)`.

---

### 🤔 WHY

`StateGraph` is just a data structure — a list of nodes and edges. It has no execution engine. `compile()` attaches:
- **Pregel engine** — the loop that calls nodes in order
- **State merging** — applies reducers when nodes return partial dicts
- **Checkpointer** — serialises state to SQLite before each node
- **Interrupt machinery** — pauses execution for HITL
- **Streaming** — yields updates as nodes complete

---

### 📍 WHEN TO USE

- ✅ `compile()` — always, once, at app startup
- ✅ `compile(checkpointer=...)` — whenever you need persistence (Module 4)
- ✅ `compile(interrupt_before=[...])` — HITL pause points (Module 7)
- ❌ Never call inside a per-request handler — compile once, reuse
- ❌ Never add nodes after compile — they won't be seen

---

### ⚖️ PROS & CONS

| Pros | Cons |
|------|------|
| All runtime features in one call | Immutable after compile |
| Validation at build time | Must recompile for topology changes |
| `get_graph()` visualisation | Compile kwargs can't change after compile |

---

### 💡 BEST PRACTICES

1. **Compile once at startup** — reuse for all requests.
2. **Gate `debug=True` behind `settings.graph_debug`** — debug logs full state per node.
3. **`build_graph()` accepts explicit compile params** — self-documenting, testable with `InMemorySaver`.
4. **`get_graph_info()` for development** — paste Mermaid output into mermaid.live to verify topology visually.
5. **Most common mistake:** Calling `add_node()` on the `StateGraph` after `compile()`. No error raised — but the compiled graph won't see the change.

---

### 🏗️ PRODUCTION PATTERN

```python
# graph.py — explicit compile params, not **kwargs
def build_graph(
    checkpointer=None,
    interrupt_before=None,
    debug=False,
):
    compile_kwargs = {}
    if checkpointer:
        compile_kwargs["checkpointer"] = checkpointer
    if interrupt_before:
        compile_kwargs["interrupt_before"] = interrupt_before
    if debug:
        compile_kwargs["debug"] = debug

    return (
        GraphBuilder(AgentState)
        .node(...)
        .edge(...)
        .entry(...)
        .build(**compile_kwargs)   # ← compile happens here
    )

# main.py — compile once, reuse
graph = build_graph()              # no checkpointer yet
# graph = build_graph(checkpointer=get_checkpointer())  # Module 4

# tests — inject InMemorySaver
graph = build_graph(checkpointer=InMemorySaver())
```

### ⚠️ Production Discoveries

1. `interrupt_before` without checkpointer: **compiles successfully** but fails at resume time. LangGraph is permissive at compile time — the runtime enforces the constraint. Always pair `interrupt_before` with a checkpointer.

2. `draw_mermaid()` output format: LangGraph >= 0.2 prepends a YAML config block (`---`) before the `graph TD;` declaration. Tests checking the format must include `"---"` as a valid prefix.

---

### 🧪 TESTS

```
tests/unit/test_compile.py — 26 tests, all passing
  TestBuildGraphSignature      — 4 tests: kwargs accepted, InMemorySaver, interrupt
  TestCompiledGraphImmutability — 4 tests: no add_node/edge after compile, independence
  TestCompiledGraphInterface   — 7 tests: all runtime methods present
  TestGetGraphInfo             — 6 tests: topology metadata, Mermaid format
  TestDebugFlag                — 2 tests: default False, settings-gated
  TestCompileKwargsPassthrough — 2 tests: checkpointer attached / None
```

---

### ✅ CONCEPT CHECKPOINT

**Three questions to answer confidently:**
1. What does `compile()` actually attach to the graph that `StateGraph` doesn't have? Name three things.
2. Why must `build_graph()` accept `checkpointer` as a parameter instead of constructing it internally?
3. What happens if you call `sg.add_node("new_node", fn)` on the `StateGraph` object after you've already called `sg.compile()`?

**Hands-on exercise:** Call `get_graph_info()` and print the Mermaid output. Paste it into `https://mermaid.live`. Verify the topology matches what you expect from `graph.py`. Screenshot it — this is your graph at the end of Module 1.

---

**Hands-on exercise (Concept 4):**
Add a new node `sanitize_input_node` that:
- Reads `state["messages"][-1].content`
- Strips leading/trailing whitespace and truncates to 1000 chars
- Writes `state["messages"]` with the sanitized `HumanMessage`
- Returns only `{"messages": [sanitized_message]}`

Wire it between the entry point and `classify_ticket` in `graph.py`. Write 3 tests for it.

---

## 📌 CONCEPT 8: Invoke  [Module 1]

### 🔍 WHAT

**Invoke** is the act of running a compiled `StateGraph` to completion and getting back a result. LangGraph exposes four invocation methods:

| Method | When to use |
|--------|-------------|
| `graph.invoke(state, config)` | Sync, blocks until END |
| `await graph.ainvoke(state, config)` | **Async, blocks until END** ← our default |
| `graph.stream(state, config)` | Sync, yields state snapshots |
| `await graph.astream(state, config)` | Async, yields events as they happen |

**Mental model:** `ainvoke` is like ordering a meal at a restaurant — you place the order (initial state), hand it to the kitchen (graph), and wait at the table until the entire meal arrives (final state). You don't see the cooking. `astream` is like watching a sushi conveyor belt — dishes arrive as they're made.

We use `ainvoke` now. Module 6 replaces it with `astream` for token streaming.

---

### 🤔 WHY

**The problem it solves:** You have a compiled graph. You need to run it. `ainvoke` is the simplest interface — one async call, one result dict.

**What breaks without the correct config structure:**

```python
# ❌ WRONG — thread_id at the top level is silently ignored by LangGraph
config = {"thread_id": "abc-123"}
result = await graph.ainvoke(initial_state, config=config)
# LangGraph never sees the thread_id — checkpointer can't save state

# ✅ CORRECT — thread_id nested under "configurable"
config = {
    "configurable": {
        "thread_id": "abc-123",
    },
    "recursion_limit": 50,
}
result = await graph.ainvoke(initial_state, config=config)
```

**Why `configurable`?** LangGraph's Pregel engine reads `thread_id`, `user_id`, and checkpoint metadata exclusively from `config["configurable"]`. Anything at the top level is treated as execution metadata (like `recursion_limit`), not thread metadata.

---

### 📍 WHEN TO USE

**Use `ainvoke` when:**
- ✅ You need the complete final state before proceeding
- ✅ Building a CLI, API endpoint, or batch job
- ✅ Writing integration tests for graph behaviour
- ✅ The workflow is short enough that waiting is acceptable

**Don't use `ainvoke` when:**
- ❌ You need to stream tokens to the user as they're generated (use `astream`)
- ❌ You're building a chat UI that needs incremental updates
- ❌ Any node takes >5 seconds and the caller needs progress feedback

**Decision rule:** `ainvoke` = response arrives all at once. `astream` = response arrives piece by piece. Module 6 upgrades to `astream`.

---

### 🌍 WHERE IT FITS

```
[CLI user input]
      │
      ▼
_build_initial_state()   ← wraps ticket in HumanMessage, inits 21 fields
      │
      ▼
_build_config()          ← nests thread_id under "configurable"
      │
      ▼
await graph.ainvoke(initial_state, config=config)
      │
      └── classify_ticket → route_ticket ──► billing_node ──┐
                                          ──► technical_node ┤
                                          ──► general_node   ┤
                                          ──► research_node  ┘
                                                  │
                                              respond → END
                                                  │
                                            final state dict
      │
      ▼
_display_result(result)  ← surfaces final_response, ticket_type, confidence
```

**Where it lives in this app:**
- `main.py → run()` — the session loop that calls `ainvoke`
- `main.py → _build_config()` — produces the correct RunnableConfig
- `main.py → _build_initial_state()` — produces the 21-field initial AgentState

---

### ⚖️ PROS & CONS

| Pros | Cons |
|------|------|
| Simple: one call, one result | No incremental output — all-or-nothing latency |
| Complete final state returned | Can't interrupt mid-graph for HITL (use `astream` + `interrupt`) |
| Easy to test — just assert on the returned dict | Caller blocks for the full execution time |
| Works with both checkpointed and non-checkpointed graphs | No visibility into which node is currently running |

---

### 🔀 MULTIPLE APPROACHES

#### Approach 1: `ainvoke` — Full state result (current)
```python
result = await graph.ainvoke(initial_state, config=config)
# result is the complete AgentState dict at END
response = result.get("final_response") or "No response generated."
```
**When to choose:** API endpoints, CLI, batch jobs, tests. Default choice until streaming is needed.

#### Approach 2: `astream` with `stream_mode="values"` (Module 6)
```python
async for state in graph.astream(initial_state, config=config, stream_mode="values"):
    pass  # final state is the last emitted value
```
**When to choose:** When you want visibility into intermediate states (debugging) but still want the full state.

#### Approach 3: `astream` with `stream_mode="messages"` (Module 6)
```python
async for chunk in graph.astream(initial_state, config=config, stream_mode="messages"):
    if isinstance(chunk[0], AIMessageChunk):
        print(chunk[0].content, end="", flush=True)
```
**When to choose:** Token streaming to the user. The production UI mode.

---

### 🆚 DIFFERENCES & COMPARISONS

**`ainvoke` vs `invoke`:**
- `invoke` is synchronous — it blocks the event loop. Use it only when no node is `async def`.
- `ainvoke` runs nodes concurrently if they're in the same parallel fan-out. `invoke` can't.
- Rule: if any node is `async def`, use `ainvoke`. Always.

**`ainvoke` vs `astream`:**
- `ainvoke` returns a dict. `astream` returns an async generator.
- `ainvoke` is `astream` + "discard all intermediate events, return the last state". They're the same engine underneath.
- `astream` is strictly more capable. We use `ainvoke` for simplicity in Module 1.

**`config["configurable"]["thread_id"]` vs `config["thread_id"]`:**
```python
# LangGraph reads from configurable — this is the ONLY way to set thread context
config = {"configurable": {"thread_id": "abc"}}   # ✅ checkpointer sees this

# Top-level is execution config — thread_id here is invisble to checkpointer
config = {"thread_id": "abc"}                      # ❌ silently ignored
```

---

### 💡 BEST PRACTICES

**1. One `build_graph()` call per session, not per message.**
Compiling a graph is expensive (~50ms, involves `inspect.signature()` for every node). Compile once at startup, reuse the instance for every `ainvoke` call.

```python
# ✅ Correct — compiled once, reused
graph = build_graph(debug=settings.graph_debug)
while True:
    ticket = input("You: ")
    result = await graph.ainvoke(initial_state, config=config)

# ❌ Wrong — recompiles on every message
while True:
    graph = build_graph()          # compiles here — expensive
    result = await graph.ainvoke(...)
```

**2. Session thread_id = `uuid4()` generated once, reused for all messages.**
The thread_id is the key the checkpointer uses to resume a conversation. If you generate a new UUID per message, Module 4's persistence becomes useless — each message starts a new thread with no history.

```python
session_id = str(uuid.uuid4())   # once per process lifetime
# All messages: config["configurable"]["thread_id"] = session_id
```

**3. Always initialize all state fields upfront.**
LangGraph fills missing fields with `None`, but uninitialized list fields that have `operator.add` reducers will crash mid-graph when a node tries to append to `None`. Initialize everything in `_build_initial_state()`.

**4. `recursion_limit` belongs at the top level, not inside `configurable`.**
```python
config = {
    "configurable": {"thread_id": "abc"},    # thread context
    "recursion_limit": 50,                    # execution limit — top level
}
```

**5. Never surface raw exceptions to users — catch in the session loop.**
The graph can fail for reasons outside your control (provider down, rate limit, OOM). The session loop catches `Exception` and shows a friendly message, then continues. The session should never crash on a single bad invocation.

---

### 🏗️ PRODUCTION IMPLEMENTATION

```python
# main.py (delivery layer)
# Full implementation is in main.py — key functions summarised here.

def _build_config(thread_id: str, settings) -> dict:
    """
    Builds RunnableConfig for a graph invocation.

    WHY "configurable" namespace?
      LangGraph reads thread_id from config["configurable"] only.
      Placing it at the top level is the #1 ainvoke mistake.
    """
    return {
        "configurable": {
            "thread_id": thread_id,
        },
        # recursion_limit: execution parameter, not thread parameter.
        # At top level, not inside configurable.
        "recursion_limit": settings.graph_recursion_limit,
    }


def _build_initial_state(ticket: str) -> dict:
    """
    Builds the minimal initial AgentState.

    All 21 fields initialized — prevents KeyError mid-graph.
    LangGraph fills missing fields with None, but list reducers
    (operator.add) will crash if the field starts as None.
    """
    return {
        "messages": [HumanMessage(content=ticket)],
        "ticket_type": None,
        "ticket_metadata": None,
        "classification_confidence": 0.0,
        "thread_id": None,            # set by LangGraph from config
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
        "authenticated": True,        # Module 10 wires real JWT
        "rbac_roles": ["agent"],
        "audit_events": [],
        "final_response": None,
        "error": None,
    }


async def run() -> None:
    """Session loop: compile once, ainvoke on every message."""
    settings = get_settings()
    session_id = str(uuid.uuid4())         # one UUID per session
    graph = build_graph(debug=settings.graph_debug)   # compiled ONCE

    while True:
        try:
            ticket = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not ticket:
            continue

        config = _build_config(session_id, settings)    # same thread_id
        initial_state = _build_initial_state(ticket)

        try:
            result = await graph.ainvoke(initial_state, config=config)
            _display_result(result)
        except Exception as e:
            print(f"\nAgent: Sorry, something went wrong: {e}\n")
```

---

### 🧪 TESTS

```python
# tests/unit/test_invoke.py — 42 tests
# Key assertions:

def test_thread_id_nested_under_configurable():
    config = _build_config("my-thread-123", get_settings())
    assert config["configurable"]["thread_id"] == "my-thread-123"
    assert "thread_id" not in config   # must NOT be at top level

def test_recursion_limit_at_top_level():
    config = _build_config("t", get_settings())
    assert "recursion_limit" in config
    assert "recursion_limit" not in config["configurable"]

def test_all_21_fields_present():
    state = _build_initial_state("test ticket")
    expected = set(AgentState.__annotations__.keys())
    assert not (expected - set(state.keys()))

async def test_same_thread_id_reused_across_messages():
    # All messages in one session must share the same thread_id
    captured = []
    async def fake_ainvoke(state, config):
        captured.append(config["configurable"]["thread_id"])
        return {...}
    # ... mock graph, send 2 messages
    assert captured[0] == captured[1]   # same session = same thread

async def test_graph_exception_does_not_crash_session():
    # Graph fails on message 1, recovers on message 2
    # Session must survive — no exception propagated to the caller
```

---

### ✅ CONCEPT CHECKPOINT

**Three questions you should answer confidently:**

1. Why does `config = {"thread_id": "abc"}` break checkpointing, even though it looks right?
2. Why is `build_graph()` called once outside the loop, not inside the loop on every message?
3. What is the difference between `ainvoke` and `astream`, and when would you switch from one to the other?

**Hands-on exercise:**
Run the app: `.venv/bin/python main.py`

Send 3 messages. Observe that the `session_id` in the banner is always the same UUID. Then modify `_build_config` to accidentally put `thread_id` at the top level (outside `configurable`), add an `InMemorySaver` checkpointer, run again — observe that the graph works but the thread state is never saved. Fix it and confirm the thread IS saved.

---

---

# Module 2 — State Design & Reducers
*Concepts 9–14 · Session 1 (continued) · ~1.5 hours*
*Unlocks: reducer-safe state schema — multiple nodes can write to the same list field without data loss*

---

## 📌 CONCEPT 9: Reducers  [Module 2]

### 🔍 WHAT

A **reducer** is a function that tells LangGraph *how to merge* a node's return value into the current state for a specific field.

**Mental model:** State fields are like bank accounts with different transaction rules. A scalar field (`ticket_type: str`) overwrites the balance — last writer wins. A reducer field (`audit_events: Annotated[list[dict], operator.add]`) deposits into the account — each write accumulates.

**Declared in:** `Annotated[list[T], reducer_function]` in `state.py`. LangGraph reads this at `compile()` time and applies it on every node return.

---

### 🤔 WHY — The Silent Data Loss Bug

```python
# Two nodes both write to audit_events — NO reducer
class BrokenState(TypedDict):
    audit_events: list[dict]     # ← plain list, no reducer

# Node A (auth): {"audit_events": [{"action": "jwt_verified"}]}
# Node B (billing): {"audit_events": [{"action": "refund_requested"}]}
#
# Final state after both nodes:
# audit_events = [{"action": "refund_requested"}]   ← auth event GONE
```

No error. No warning. The auth audit event is silently dropped. In a compliance audit, this means a missing evidence trail.

**The fix:**
```python
class FixedState(TypedDict):
    audit_events: Annotated[list[dict], operator.add]  # ← reducer

# Final state after both nodes:
# audit_events = [{"action": "jwt_verified"}, {"action": "refund_requested"}]
```

---

### 📍 WHEN TO USE

| Scenario | Reducer? | Which? |
|----------|---------|--------|
| Multiple nodes append to a list | ✅ | `operator.add` |
| Chat history shared across all nodes | ✅ | `add_messages` |
| Single node owns the field | ❌ | Plain type |
| Scalar that gets replaced | ❌ | Plain type |

**Rule:** *"Can more than one node write to this field in a single run?"* → Yes = reducer.

---

### 🏗️ Production: the 6 reducer fields in `AgentState`

```python
# state.py — these 6 fields all use reducers

messages: Annotated[list[BaseMessage], add_messages]
# add_messages: append + dedup by id. Safe for HITL resume.

retrieved_memories: Annotated[list[str], operator.add]
# Multiple memory retrieval passes accumulate.

agent_results: Annotated[list[dict], operator.add]
# Fan-out workers all append their result to this list.

retrieved_docs: Annotated[list[RetrievedDocument], operator.add]
# Multi-query RAG appends results from each query variant.

rbac_roles: Annotated[list[str], operator.add]
# Base roles + tenant roles appended in two passes.

audit_events: Annotated[list[dict], operator.add]
# APPEND ONLY — every sensitive node appends, never overwrites.
```

---

## 📌 CONCEPT 10: Annotated State  [Module 2]

`Annotated[T, metadata]` is Python's standard way to attach extra information to a type. LangGraph uses the metadata slot to store the reducer function.

```python
from typing import Annotated
import operator

# Annotated[actual_type, reducer]
audit_events: Annotated[list[dict], operator.add]
#                       ^^^^^^^^^^  ^^^^^^^^^^^^^
#                       mypy sees   LangGraph reads

# Without include_extras=True, get_type_hints() strips Annotated:
# get_type_hints(AgentState) → {"audit_events": list[dict]}  ← reducer lost
# get_type_hints(AgentState, include_extras=True) → {"audit_events": Annotated[...]}
```

**Key:** Always use `get_type_hints(Schema, include_extras=True)` when inspecting LangGraph state schemas. Without `include_extras=True`, Python strips the `Annotated` wrapper and the reducer metadata disappears.

---

## 📌 CONCEPT 11: operator.add  [Module 2]

`operator.add(existing, new)` = `existing + new`. Standard Python list concatenation used as a LangGraph reducer.

```python
# Properties:
operator.add(["a", "b"], ["c"])     # → ["a", "b", "c"]  — preserves order
operator.add([], ["first"])         # → ["first"]         — safe for empty start
operator.add(["x"], [])             # → ["x"]             — no-op on empty new
operator.add(["x"], ["x"])          # → ["x", "x"]        — NO deduplication

# Node pattern — return ONLY new items:
async def billing_node(state, config) -> dict:
    return {"audit_events": [{"action": "refund_requested"}]}
    #  NOT: state["audit_events"] + [...]  ← doubles on checkpoint replay
```

---

## 📌 CONCEPT 12: add_messages  [Module 2]

`add_messages` is like `operator.add` with deduplication by `message.id`. Critical for HITL resume correctness.

```python
from langgraph.graph.message import add_messages

# Append — same as operator.add when IDs differ:
existing = [HumanMessage(content="hello", id="h-1")]
new      = [AIMessage(content="hi",       id="ai-1")]
add_messages(existing, new)  # → [HumanMessage(...), AIMessage(...)]

# Dedup — replaces when IDs match:
v1 = AIMessage(content="original", id="ai-1")
v2 = AIMessage(content="updated",  id="ai-1")
add_messages([v1], [v2])  # → [AIMessage("updated")] — NOT two messages
```

**Why this matters for HITL:** When a graph resumes from a checkpoint, LangGraph replays saved state. Without deduplication, every resume injects duplicate messages. `add_messages` makes replay idempotent — the same `id` is never stored twice.

---

## 📌 CONCEPT 13: State Merging  [Module 2]

Nodes return **partial dicts** — only the fields they changed. LangGraph merges this partial dict into the full state using reducers where declared.

```python
# ✅ CORRECT — partial return
async def classify_node(state, config) -> dict:
    return {"ticket_type": "billing"}   # only what this node changes

# ❌ WRONG — full state return with reducer field
async def classify_node(state, config) -> dict:
    return {
        "ticket_type": "billing",
        "audit_events": state["audit_events"] + [new_event],  # ← doubles on replay
        # ... all 21 fields ...
    }
```

**The doubling bug:** If you return `state["audit_events"] + [new_event]` from a node, and `audit_events` has `operator.add` reducer, LangGraph applies: `operator.add(state["audit_events"], state["audit_events"] + [new_event])`. Every pre-existing event is duplicated.

---

## 📌 CONCEPT 14: Reducer Design Patterns  [Module 2]

Any callable `(existing: T, new: T) -> T` works as a reducer. Three production patterns:

```python
# 1. Dedup by key (Module 9: multi-query RAG)
def dedup_by_source(existing: list, new: list) -> list:
    seen = {d["source"] for d in existing}
    return existing + [d for d in new if d["source"] not in seen]

retrieved_docs: Annotated[list[RetrievedDocument], dedup_by_source]

# 2. Bounded list (prevents OOM in long sessions)
def bounded_append(existing: list, new: list) -> list:
    return (existing + new)[-MAX_EVENTS:]

audit_events: Annotated[list[dict], bounded_append]

# 3. Custom merge (priority-based)
def keep_highest_score(existing: list, new: list) -> list:
    combined = existing + new
    return sorted(combined, key=lambda d: d["score"], reverse=True)[:TOP_K]
```

**MUST be pure:** No side effects, no I/O, no mutation of inputs. Reducers are called synchronously by LangGraph's merge engine inside `ainvoke`.

---

### ✅ MODULE 2 CHECKPOINT

Three questions to answer confidently:
1. What happens to `audit_events` if two nodes both write to it and there's no reducer?
2. Why is `operator.add` wrong for chat messages, and what does `add_messages` add?
3. What is the read-then-write bug, and why does it double entries on checkpoint replay?

**Hands-on:** Add a `print` inside `billing_node` in `agents/nodes.py` that shows `state["audit_events"]` at entry. Run two tickets back-to-back in the CLI. Observe the list growing correctly across turns. Then temporarily remove the reducer from `audit_events` in `state.py`, rerun — observe the events from the first turn disappearing.

---

# Module 3 — Routing, Tools & Agents
*Concepts 15–21 · Session 2 · ~2.5 hours*
*Unlocks: tool-capable agents, decoupled observability, safe config access*

---

## 📌 CONCEPT 15: Router Pattern  [Module 3]

### 🔍 WHAT

A **router** is a pure synchronous function that reads state and returns a string key telling LangGraph which branch to take next. It is the function passed as the second argument to `add_conditional_edges`.

**Mental model:** A router is a traffic light at an intersection. It looks at current conditions (state) and signals which road to take. It does not build the road, move the cars, or call anyone — it just reads and decides.

```
classify_ticket ──► route_ticket(state) ──► "billing"   ──► billing_node
                                       ──► "technical" ──► technical_node
                                       ──► "general"   ──► general_node
                                       ──► "research"  ──► research_node
```

---

### 🤔 WHY — Three Ways a Router Breaks

```python
# ❌ FAILURE 1: async router
async def route_ticket(state):           # returns a coroutine object, not a string
    return state.get("ticket_type")      # coroutine is truthy — silent wrong routing

# ❌ FAILURE 2: key not in mapping
def route_ticket(state):
    return "BILLING"                     # uppercase — not in {"billing": "billing_node"}
                                         # raises InvalidUpdateError at runtime

# ❌ FAILURE 3: I/O inside router
def route_ticket(state):
    cfg = requests.get("http://config")  # blocks event loop, no retry, untestable
    return cfg.json()["route"]
```

**Our implementation avoids all three:**
```python
def route_ticket(state: AgentState) -> str:   # sync ✅
    ticket_type = state.get("ticket_type")
    if ticket_type not in ROUTE_KEYS:          # validates key ✅
        return "general"                       # safe fallback ✅
    return ticket_type                         # no I/O ✅
```

### 💡 `ROUTE_KEYS` — Single Source of Truth

```python
# agents/nodes.py
ROUTE_KEYS: dict[str, str] = {
    "billing":   "billing_node",
    "technical": "technical_node",
    "general":   "general_node",
    "research":  "research_node",
}

# graph.py imports and uses the same dict — no duplication
.conditional_edge("classify_ticket", _route_ticket, ROUTE_KEYS)
```

If you add a new route, change `ROUTE_KEYS` in one place. Both the router fallback check and the graph topology update automatically.

---

## 📌 CONCEPT 16: ToolNode  [Module 3]

### 🔍 WHAT

A **Tool** is a function the LLM can choose to call. A **ToolNode** is a LangGraph node that executes whatever tools the LLM requested in its last `AIMessage`.

**Mental model:** The LLM is a project manager. Tools are contractors. `ToolNode` is the operations team that actually calls the contractors when the PM issues a work order.

```
LLM thinks: "I need to search the web"
     │
     ▼
AIMessage with tool_calls=[{"name": "web_search", "args": {"query": "GDPR art 17"}}]
     │
     ▼
ToolNode executes web_search("GDPR art 17")
     │
     ▼
ToolMessage with result added to state["messages"]
     │
     ▼
LLM reads the result and generates final answer
```

### 🔀 Two Approaches

| Approach | Code | When to use |
|----------|------|-------------|
| `@tool` decorator | `@tool\ndef search(q: str): ...` | Quick scripts, prototypes |
| `BaseTool` subclass | `class SearchTool(BaseTool): ...` | Production — typed, testable, documented |

We use `BaseTool` with a Pydantic `args_schema`. The schema does three things:
1. **Validates** LLM-provided arguments before the tool runs
2. **Documents** what arguments the tool accepts (LLM reads this)
3. **Enables** unit testing tool inputs without calling the tool

```python
class SearchToolInput(BaseModel):
    query: str = Field(..., min_length=3, max_length=500,
                       description="Be specific — include product names or error codes")
    max_results: int = Field(default=5, ge=1, le=10)

class SearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Search the web for current information..."
    args_schema: Type[BaseModel] = SearchToolInput

    def _run(self, query: str, max_results: int = 5) -> str:
        ...  # actual implementation
```

### 📁 Files Created

```
tools/
├── __init__.py         ← get_research_tools(), get_technical_tools()
├── search_tool.py      ← SearchTool (Tavily web search)
└── kb_tool.py          ← KnowledgeBaseTool (FAISS internal KB)
```

### 💡 Graceful Degradation

Both tools return a helpful message (not an exception) when dependencies are unavailable:
- `SearchTool` → no Tavily API key → `"[Search unavailable — TAVILY_API_KEY not configured]"`
- `KnowledgeBaseTool` → no FAISS index → `"[Knowledge base not yet indexed — run build_index()]"`

This keeps the graph running and lets the LLM inform the user instead of crashing.

---

## 📌 CONCEPT 17: create_react_agent  [Module 3]

### 🔍 WHAT

`create_react_agent` is LangGraph's factory function that builds a ReAct-style agent — a loop of: Think → Act (call tool) → Observe (read result) → repeat until done.

**When to use the factory vs hand-rolling:**

| Factory (`create_react_agent`) | Hand-rolled (`BaseAgent`) |
|-------------------------------|--------------------------|
| Simple tool-calling agents | Complex custom logic per agent type |
| Standard ReAct loop is sufficient | Need HITL, custom state, branching |
| Fast to prototype | Full control over execution |
| Harder to customise | Testable in isolation, extensible |

We use `BaseAgent` (hand-rolled). The factory is introduced here so you know it exists.

```python
# Factory approach — one line, standard ReAct loop
from langgraph.prebuilt import create_react_agent
agent = create_react_agent(llm, tools=[SearchTool()])

# Our approach — full control, SOLID design
class BillingAgent(BaseAgent):
    async def process(self, state, config) -> dict: ...
```

### The `BaseAgent` ABC

```python
# agents/base.py
class BaseAgent(ABC):
    @property
    @abstractmethod
    def agent_name(self) -> str: ...   # "billing", "technical", etc.

    @abstractmethod
    def get_tools(self) -> list: ...   # tools this agent has access to

    @abstractmethod
    async def process(self, state: AgentState, config: RunnableConfig) -> dict: ...
```

**Why an ABC?** Open/Closed Principle. The supervisor calls `agent.process()` for any registered agent without knowing its type. Adding a `LegalAgent` never touches the supervisor — just create a new subclass.

Attempting to instantiate `BaseAgent` directly raises `TypeError: Can't instantiate abstract class` — the contract is enforced by Python at instantiation time, not at call time.

---

## 📌 CONCEPT 18: Dynamic Routing  [Module 3]

### 🔍 WHAT

**Dynamic routing** means the supervisor resolves which agent to use at runtime from a registry — not from a hardcoded `if/elif` chain.

**Mental model:** A hotel concierge directory. The concierge doesn't know every specialist in the city. They look up the right one in a directory by name and connect you. The directory can be updated without training the concierge.

```python
# ❌ Static routing — fragile, requires touching supervisor for every new agent
def supervisor(state):
    if state["ticket_type"] == "billing":
        return BillingAgent().process(state, config)
    elif state["ticket_type"] == "technical":
        return TechnicalAgent().process(state, config)
    # Adding LegalAgent means editing this function

# ✅ Dynamic routing — supervisor never changes
def supervisor(state):
    agent = get_agent(state["ticket_type"])  # registry resolves
    return await agent.process(state, config)
```

### The Registry Pattern

```python
# agents/registry.py

# Register a factory (not an instance — factories create fresh agents each call)
@register_agent("billing")
def create_billing_agent() -> BillingAgent:
    return BillingAgent()

# Supervisor resolves by name
agent = get_agent("billing")        # calls create_billing_agent()
names = list_agents()               # ["billing", "technical", ...]
```

**Key design decisions:**
- Stores **factories**, not instances — `get_agent()` always returns a fresh object
- `@register_agent` returns the factory unchanged — it's pure side-effect decoration
- `clear_registry()` exists for tests — never call in production

---

## 📌 CONCEPT 19: Hooks  [Module 3]

### 🔍 WHAT

**Hooks** are functions called before (pre-hook) or after (post-hook) every node executes. They implement the **Observer pattern** — nodes don't know hooks exist; hooks don't know node internals.

**Mental model:** Security cameras in a building. The employees (nodes) do their work normally. The cameras (hooks) observe and record without interrupting. Adding a new camera doesn't change what employees do.

### 🤔 WHY — Without Hooks

```python
# ❌ Without hooks — every node must log its own latency
async def classify_node(state, config):
    start = time.perf_counter()                          # boilerplate
    # ... actual work ...
    duration = (time.perf_counter() - start) * 1000     # boilerplate
    logger.info("node=classify duration=%.1f", duration) # boilerplate
    return result

# To add tracing to 10 nodes = 30 lines of boilerplate added to business logic
```

```python
# ✅ With hooks — boilerplate extracted once, applied everywhere
register_post_hook(_latency_hook)   # registered once at startup
# ALL nodes now log latency automatically — zero node code changes
```

### Implementation

```python
# observability/hooks.py

def node_with_hooks(name: str, fn: Callable) -> Callable:
    @functools.wraps(fn)   # preserves signature for LangGraph
    async def wrapped(state, config):
        for hook in _pre_hooks:   # 1. pre-hooks
            hook(name, state, config)

        start = time.perf_counter()
        result = await fn(state, config)   # 2. the actual node
        duration_ms = (time.perf_counter() - start) * 1000

        for hook in _post_hooks:   # 3. post-hooks
            hook(name, {**result, "duration_ms": duration_ms}, config)

        return result   # duration_ms NOT returned to LangGraph — hooks only
    return wrapped

# Usage in graph.py (Module 4+ when we wire this in):
builder.node("classify", node_with_hooks("classify", classify_ticket_node))
```

### Critical Detail: `functools.wraps`

`node_with_hooks` uses `@functools.wraps(fn)` — the same fix as `with_retry`. Without it, `inspect.signature(wrapped)` shows `(*args, **kwargs)` and LangGraph omits `config` from the call. `functools.wraps` copies `__wrapped__ = fn`, so `inspect.signature` follows the chain to the original `(state, config)` signature.

### Hook Safety Rules

1. **Hooks never crash the node** — exceptions inside a hook are caught and logged; the node continues
2. **Hooks are sync** — no `await` inside hooks (use a background task for async observability)
3. **Post-hooks receive `duration_ms`** — injected by the wrapper, not present in the final state
4. **`clear_hooks()` in tests** — always reset hook state between tests

---

## 📌 CONCEPT 20: RunnableConfig  [Module 3]

### 🔍 WHAT

`RunnableConfig` is the second argument every LangGraph node receives. It carries per-invocation metadata: `thread_id`, `user_id`, `tenant_id`, and anything else passed in `config["configurable"]`.

### The Safe Access Pattern

```python
# ❌ WRONG — raises KeyError if config is empty or "configurable" is missing
thread_id = config["configurable"]["thread_id"]

# ✅ CORRECT — safe chain of .get() with fallbacks
thread_id = config.get("configurable", {}).get("thread_id", "unknown")
```

**Why does this matter?** During testing, nodes are often called with `config={}`. Any node that uses direct key access will crash in tests. The `.get()` pattern means nodes work correctly in all contexts.

```python
# agents/nodes.py — the pattern used everywhere
def _get_thread_id(config: RunnableConfig) -> str:
    return config.get("configurable", {}).get("thread_id", "unknown")
```

### What lives in `config["configurable"]`

| Key | Set by | Used for |
|-----|--------|---------|
| `thread_id` | `main.py` `_build_config()` | Checkpointer session key |
| `user_id` | Auth node (Module 10) | RBAC, audit logs |
| `tenant_id` | Auth node (Module 10) | Multi-tenancy isolation |
| `checkpoint_id` | LangGraph (time travel) | Resuming from a specific snapshot |

---

## 📌 CONCEPT 21: Runtime Configuration  [Module 3]

### 🔍 WHAT

**Runtime configuration** means behaviour that changes based on settings — without redeploying code. Settings are read **inside functions at call time**, not at module import time.

### 🤔 WHY — Import Time vs Call Time

```python
# ❌ WRONG — read at import time: frozen forever when the module is first imported
DEBUG = get_settings().graph_debug   # evaluated once when module is imported

def _latency_hook(name, result, config):
    if DEBUG:   # always the value from import time — can't be changed in tests
        logger.info(...)

# ✅ CORRECT — read at call time: fresh value on every invocation
def _latency_hook(name, result, config):
    if get_settings().graph_debug:   # read NOW — can be patched in tests
        logger.info(...)
```

**In tests**, `get_settings()` is patched via `mock_settings` fixture. If settings were read at import time, the patch would have no effect.

### Settings that control runtime behaviour

```python
# core/settings.py — these flags gate features at runtime
graph_debug: bool          # enables verbose node logging + latency hooks
streaming_enabled: bool    # switches ainvoke → astream (Module 6)
memory_top_k: int          # controls how many memories are retrieved
rag_score_threshold: float # controls retrieval relevance cutoff
```

### ✅ MODULE 3 CHECKPOINT

Three questions to answer confidently:
1. Why does making a router `async` silently break routing instead of raising an error?
2. What is the difference between `BaseTool._run()` and a plain function decorated with `@tool`?
3. Why do hooks use `functools.wraps` — and what breaks without it?

**Hands-on exercise:**
Register a new `EscalationAgent` in the registry:
```python
@register_agent("escalation")
def create_escalation_agent():
    class EscalationAgent(BaseAgent):
        @property
        def agent_name(self): return "escalation"
        def get_tools(self): return []
        async def process(self, state, config):
            return {"active_agent": "escalation", "escalation_level": 1}
    return EscalationAgent()
```
Verify `list_agents()` includes it. Notice the supervisor never needed to change.

---

# Module 4 — Persistence, Memory & Time Travel
*Concepts 22–34 · Session 2 (continued) · ~2 hours*
*Unlocks: resumable sessions — same thread_id + checkpointer = full conversation survives restarts*

---

## 📌 CONCEPT 22: Graph Memory

### 🔍 WHAT

Without a checkpointer, every `graph.invoke()` starts from a blank slate — the graph has no memory of previous calls. With a **checkpointer**, LangGraph saves a snapshot of the full state after every node runs. The next invocation on the same `thread_id` loads that snapshot and resumes from where it left off.

**Mental model:** Think of a whiteboard in a meeting room. Without a checkpointer, someone erases the board between every meeting. With a checkpointer, the board is photographed and restored exactly as it was. Every participant (node) can read everything that was written before.

### 🤔 WHY

Without a checkpointer, every support ticket interaction is stateless:

```python
# ❌ Without checkpointer — second message has no memory of the first
r1 = graph.invoke({"messages": [HumanMessage("My invoice is wrong")]}, config=config)
r2 = graph.invoke({"messages": [HumanMessage("I already explained this!")]}, config=config)
# r2 has NO context from r1 — the agent starts fresh, customer is frustrated
```

With a checkpointer, the second invocation resumes from the full accumulated state of the first.

### 🔑 The critical rule: don't override checkpoint fields in subsequent invocations

```python
# ❌ WRONG — this RESETS count to 0, defeating the checkpointer
graph.invoke({"messages": [], "count": 0}, config=config)
graph.invoke({"messages": [], "count": 0}, config=config)  # count resets to 0 again!

# ✅ RIGHT — pass {} for subsequent calls; checkpoint provides all existing state
graph.invoke({"messages": [], "count": 0}, config=config)  # first call seeds state
graph.invoke({}, config=config)                             # subsequent: let checkpoint fill in
```

**Why?** For fields without a reducer (plain `int`, `str`, `bool`), the input value REPLACES the checkpoint value. Passing `count: 0` on a second invocation overwrites the persisted `count: 1`. Only reducer fields (`Annotated[list[T], operator.add]`) MERGE the input with the checkpoint — they don't overwrite.

---

## 📌 CONCEPT 23: Checkpointing

### 🔍 WHAT

Checkpointing is the mechanism by which LangGraph saves and restores graph state. `get_checkpointer()` in `core/factories.py` is the factory that resolves which backend to use based on `CHECKPOINTER_BACKEND` in `.env`.

### 🏗️ PRODUCTION IMPLEMENTATION

The factory is NOT `lru_cached` — `SqliteSaver` holds an open database connection. Caching it would reuse the same connection across tests, causing `sqlite3.OperationalError: database is locked`.

```python
# core/factories.py
def get_checkpointer():
    settings = get_settings()
    backend = settings.checkpointer_backend

    if backend == "memory":
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()

    if backend == "sqlite":
        import sqlite3
        from langgraph.checkpoint.sqlite import SqliteSaver
        # Use direct constructor — SqliteSaver.from_conn_string() returns a context
        # manager, NOT an instance. Direct construction gives us a plain SqliteSaver.
        conn = sqlite3.connect(settings.sqlite_path, check_same_thread=False)
        return SqliteSaver(conn)

    raise ValueError(f"Unknown checkpointer_backend: {backend!r}.")
```

**`check_same_thread=False`** — SQLite's default is to forbid cross-thread access. LangGraph's async runtime may call the checkpointer from a different thread than the one that opened the connection. This flag disables that restriction safely for our single-writer CLI use case.

---

## 📌 CONCEPT 24: InMemorySaver

### 🔍 WHAT

`MemorySaver` (also called `InMemorySaver`) stores checkpoints in a Python dict. Identical API to `SqliteSaver` — the graph can't tell the difference. No disk I/O. Lost on process exit.

**When to use:** Tests, development without persistence, short-lived processes.

### 🧪 TEST FIXTURE PATTERN

```python
# tests/conftest.py
@pytest.fixture
def memory_checkpointer():
    return MemorySaver()  # fresh instance per test — no state leakage
```

**Why a fixture instead of creating inline?** A fixture creates a NEW `MemorySaver` for every test function. If you reused a single instance across tests, thread state from test A would bleed into test B.

---

## 📌 CONCEPT 25: SQLite Checkpointer

### 🔍 WHAT

`SqliteSaver` persists checkpoints to a local SQLite database file (`memory.db` by default). The file survives process restarts — this is the "resumable sessions" feature. The graph literally picks up from the exact node it was at when the process died.

### 🔑 LSP: InMemorySaver and SqliteSaver are substitutable

Both inherit `BaseCheckpointSaver`. A caller that accepts a `BaseCheckpointSaver` works identically with either:

```python
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

assert isinstance(MemorySaver(), BaseCheckpointSaver)    # ✅
assert isinstance(SqliteSaver(conn), BaseCheckpointSaver)  # ✅

# build_graph accepts either:
build_graph(checkpointer=MemorySaver())   # tests
build_graph(checkpointer=SqliteSaver(conn))  # production
```

---

## 📌 CONCEPT 26: Postgres Checkpointer  [Module 4]
**Layer:** Infrastructure | **Complexity:** Intermediate

> **Implementation status (read first):** this repo ships **no Postgres code path**.
> `Settings.checkpointer_backend` is `Literal["memory", "sqlite"]` (`core/settings.py`), and
> `get_checkpointer()` in `core/factories.py` raises `ValueError` for anything else. There is no
> `postgres_url` setting and no Postgres test. What this concept documents is the *seam* the code
> leaves for Postgres, plus the Liskov argument that makes the swap safe — not a working backend.

### 🔍 WHAT

`PostgresSaver` (package `langgraph-checkpoint-postgres`) is LangGraph's checkpointer for
multi-process / multi-host deployments. Like `MemorySaver` and `SqliteSaver`, it extends
`BaseCheckpointSaver`, so `graph.compile(checkpointer=...)` accepts it unchanged.

**Mental model:** SQLite is a notebook on your desk — fast, private, one writer. Postgres is a shared
filing cabinet in the office — many workers (API replicas) can read and write the same threads.

### 🤔 WHY

`SqliteSaver` is a single local file. Two API replicas each opening `memory.db` do not share a
coherent view of a thread, and SQLite's write lock is the wrong tool for concurrent writers. The
FastAPI delivery layer (`api/routes.py`, Concept 120) is where that limit would first bite.

### 📍 WHEN TO USE

- ✅ Multiple processes/replicas must resume the *same* `thread_id`
- ✅ You need backups, retention, and access control from your database platform
- ❌ Single-user CLI or local dev — SQLite is enough
- ❌ Unit tests — use `MemorySaver` (Concept 24)

### 🌍 WHERE IT FITS

```
get_checkpointer()  ──resolves──►  MemorySaver | SqliteSaver | (PostgresSaver — not wired)
        │
        └──► build_graph(checkpointer=...)  ──►  graph.compile(checkpointer=...)
```

Where the seam is in the code:

| File | What it says |
|------|--------------|
| `core/factories.py` | Module docstring: "swap SQLite → Postgres in one line" — aspirational; the `if backend == ...` chain has two branches |
| `core/graph_builder.py` | `build()` docstring mentions `PostgresSaver` as the Module 4+ production option |
| `main.py` | Uses `AsyncSqliteSaver.from_conn_string(settings.sqlite_path)` **directly**, not `get_checkpointer()` |

### 🔑 LSP — why the swap is safe

```python
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

assert isinstance(MemorySaver(), BaseCheckpointSaver)
# PostgresSaver would satisfy the same contract; build_graph(checkpointer=...) never needs to know.
```

### ⚖️ PROS & CONS

| Pros | Cons |
|------|------|
| Shared state across replicas | Needs a running Postgres + a connection string (a secret) |
| Production-grade durability | Extra dependency (`langgraph-checkpoint-postgres`) not in this project's setup |
| Same `BaseCheckpointSaver` contract | Not implemented here — must be added, tested, and wired into `main.py` |

### 💡 BEST PRACTICES

1. Hold the connection string as `SecretStr` (as `jwt_secret` is) so it never appears in logs.
2. Extend the `Literal` in `Settings` and add the `if backend == "postgres"` branch together — the `Literal` makes a typo fail at startup.
3. Keep the factory un-cached (as `get_checkpointer()` is today): checkpointers own connections.
4. Route `main.py` through `get_checkpointer()` first; today it bypasses the factory, so a new backend would not take effect in the CLI.

### 🏗️ PRODUCTION PATTERN (what adding it would look like)

```python
# core/settings.py — extend, don't modify existing branches (Open/Closed)
checkpointer_backend: Literal["memory", "sqlite", "postgres"] = Field(default="sqlite")
postgres_url: SecretStr | None = Field(default=None)

# core/factories.py — new branch only
if backend == "postgres":
    ...  # construct PostgresSaver from settings.postgres_url.get_secret_value()
```

### ✅ CONCEPT CHECKPOINT

1. Why is every checkpointer substitutable in `build_graph(checkpointer=...)`?
2. Which file would silently ignore a new `postgres` backend today, and why?
3. Why is the connection string a `SecretStr` rather than `str`?

**Exercise:** add the `postgres` branch behind a `Literal` extension, then write a factory test (mirroring `tests/unit/test_persistence.py::TestCheckpointerFactory`) asserting an unknown backend still raises `ValueError`.

---

## 📌 CONCEPTS 27–29: Thread IDs, User IDs, Metadata

### 🔍 WHAT

A **thread ID** is the unique key that scopes a checkpoint namespace. All invocations with the same `thread_id` share state. Different `thread_id`s are completely isolated.

**Mental model:** Think of thread IDs as file folder names. Every customer's conversation is stored in their own folder. Two customers can never read each other's files, even if they're in the same filing cabinet (checkpointer).

### 🏗️ PRODUCTION PATTERN

```python
# main.py — one UUID per CLI session
session_id = str(uuid.uuid4())

# Build config for every invocation in this session
config = {
    "configurable": {
        "thread_id": session_id,    # ← MUST be under "configurable"
        "user_id": "usr-123",       # optional — for multi-user systems
        "tenant_id": "acme-corp",   # optional — for multi-tenant systems
    },
    "recursion_limit": 50,         # ← NOT under "configurable"
}
```

**Common mistake:** `config = {"thread_id": "abc"}` — LangGraph ignores keys outside `"configurable"`. The checkpointer never sees the thread ID. State is never saved.

### TYPED METADATA

Never use raw dicts for metadata — use `TicketMetadata`:

```python
# state.py
class TicketMetadata(TypedDict):
    ticket_id: str
    tenant_id: str       # never "tenenat_id" — typos are caught by mypy
    user_id: str
    priority: str
    created_at: str      # ISO 8601
    sla_deadline: str    # ISO 8601
```

---

## 📌 CONCEPT 30: Persistent Threads

### 🔍 WHAT

Persistent threads are the practical payoff of checkpointing: a customer can close their browser, come back tomorrow, and the agent has full memory of the entire previous conversation.

### 🏗️ SIMULATING A PROCESS RESTART IN TESTS

```python
def test_same_thread_id_resumes_after_simulated_restart(self, tmp_path):
    import sqlite3
    db_path = str(tmp_path / "session.db")

    # "First process" — runs the graph once, writes checkpoint
    conn1 = sqlite3.connect(db_path, check_same_thread=False)
    cp1 = SqliteSaver(conn1)
    g1 = _make_simple_graph(checkpointer=cp1)
    r1 = g1.invoke({"messages": [], "count": 0}, config=config)
    assert r1["count"] == 1
    conn1.close()   # simulate: kill -9 python

    # "Second process" — NEW SqliteSaver reading same file, NEW graph
    conn2 = sqlite3.connect(db_path, check_same_thread=False)
    cp2 = SqliteSaver(conn2)
    g2 = _make_simple_graph(checkpointer=cp2)
    r2 = g2.invoke({}, config=config)   # {} = let checkpoint fill state
    assert r2["count"] == 2             # picked up exactly where g1 left off
```

---

## 📌 CONCEPT 31: Checkpoint Storage — `get_state_history()`

### 🔍 WHAT

`graph.get_state_history(config)` returns an iterator of all checkpoints for a thread, **most-recent-first**. Each snapshot is a `StateSnapshot` with `.values` (the state dict) and `.config` (contains the checkpoint ID).

```python
history = list(graph.get_state_history(config))

for snapshot in history:
    count = snapshot.values.get("count", 0)
    checkpoint_id = snapshot.config["configurable"]["checkpoint_id"]
    print(f"count={count} id={checkpoint_id[:8]}...")

# Output (most recent first):
# count=3 id=abc12345...
# count=2 id=def67890...
# count=1 id=ghi11223...
# count=0 id=jkl44556...
```

`graph.get_state(config)` — shortcut that returns just the latest snapshot.

---

## 📌 CONCEPTS 32–33: State Recovery & Durable Execution

### 🔍 WHAT

**Idempotency** means running a node twice produces the same result as running it once. This is mandatory for nodes that might be replayed (e.g., after a crash mid-execution).

**Pure computation nodes** are naturally idempotent — they only read and compute, never write to external systems.

**Side-effect nodes** (send email, charge card, issue refund) are NOT naturally idempotent. The **durable execution pattern** makes them safe:

```python
def refund_node(state: DurableState) -> dict:
    # Check BEFORE acting — is this already done?
    if state.get("refund_issued"):
        return {}   # already done — no side effect, no audit event

    # First time only — perform the side effect
    _charge_refund_api(state["amount"])
    return {
        "refund_issued": True,
        "audit_events": ["refund_issued"],
    }
```

**Critical:** On replay, pass `{}` as input — let the checkpoint provide `refund_issued: True`. If you pass `{"refund_issued": False}`, it overrides the checkpoint and the refund fires again.

---

## 📌 CONCEPT 34: Time Travel

### 🔍 WHAT

Time travel lets you resume execution from ANY past checkpoint, not just the latest. Pass `checkpoint_id` in `config["configurable"]` to fork from that exact point.

**Use cases:**
- **Debug**: replay from just before a node failed, try a different input
- **Testing**: fork from a known state and test different paths
- **Audit**: replay the exact conversation a customer had

```python
# Get all past checkpoints
history = list(graph.get_state_history(config))

# Find the checkpoint at count=1
snapshot = next(s for s in history if s.values.get("count") == 1)
checkpoint_id = snapshot.config["configurable"]["checkpoint_id"]

# Time travel: resume from that checkpoint
time_travel_config = {
    "configurable": {
        "thread_id": "my-thread",
        "checkpoint_id": checkpoint_id,   # ← this is the time travel key
    }
}

# Pass {} — let the past checkpoint provide all state
result = graph.invoke({}, config=time_travel_config)
```

**What happens:** LangGraph loads the checkpoint at `checkpoint_id` as the starting state, ignoring the latest checkpoint for this thread. The node runs from that historical state forward.

---

### ✅ MODULE 4 CHECKPOINT

Three questions to answer confidently:
1. Why must you pass `{}` (not `{"count": 0}`) on subsequent invocations when using a checkpointer?
2. Why is `get_checkpointer()` NOT `lru_cached`, but `get_settings()` IS?
3. What is the difference between `graph.get_state(config)` and `graph.get_state_history(config)`?

**Hands-on exercise:**
Run `main.py`, send two messages in one session, then restart the process with the same `thread_id`. Verify the agent remembers the first message. Then inspect `memory.db` with `sqlite3 memory.db "SELECT * FROM checkpoints"` to see the raw checkpoint data.

---

# Module 5 — Long-Term Memory System
*Concepts 35–41 · Session 3 · ~2 hours*
*Unlocks: cross-session user memory — agent remembers facts across conversations*

---

## 📌 CONCEPT 35: Long-Term Memory

### 🔍 WHAT

Short-term memory (Module 4) saves the state of one conversation thread and restores it on restart. Long-term memory stores facts, preferences, and history **across** threads — scoped to a `user_id`, not a `thread_id`.

**Mental model:** Short-term = whiteboard in a meeting room (erased when the project ends). Long-term = notebook you carry everywhere (persists across every project, every year).

```
Short-term (Module 4):             Long-term (Module 5):
  thread_id = "abc123"               user_id = "alice"
  ┌──────────────────────┐           ┌────────────────────────────────┐
  │ message 1            │           │ "prefers itemised invoices"    │
  │ message 2            │    +      │ "account: enterprise tier"     │
  │ message 3            │           │ "had billing bug #441 in June" │
  └──────────────────────┘           └────────────────────────────────┘
  Dies when thread ends              Shared across ALL threads for alice
```

---

## 📌 CONCEPT 36: Memory Store

### 🔍 WHAT

LangGraph provides `InMemoryStore` — a key-value store that lives **outside** the graph state and **outside** the checkpointer. `LongTermMemoryStore` wraps it with our application's interface.

**Two operations:**
```python
# Write
await store.astore(user_id="alice", key="pref_billing", text="prefers itemised invoices")

# Read (semantic search)
memories = await store.asearch(user_id="alice", query="invoice preference")
```

**Namespace isolation:** Every user's memories live in `("memories", user_id)` — a tuple that acts like a folder. Alice's memories and Bob's memories share the same `InMemoryStore` object but can never cross-contaminate.

### 🏗️ IMPLEMENTATION NOTES

```python
# InMemoryStore API — namespace_prefix is POSITIONAL-ONLY
await self._store.aput(namespace, key, value)          # NOT namespace=namespace
results = await self._store.asearch(namespace, query=query, limit=k)  # NOT namespace=namespace
```

**Why `lru_cache` on `get_memory_store()` but NOT on `get_checkpointer()`?**
- `InMemoryStore` = Python dict. No file handles. Safe to share across tests.
- `SqliteSaver` = open file handle. Sharing causes "database is locked".

---

## 📌 CONCEPT 37: Semantic Memory

### 🔍 WHAT

Regular storage retrieves by exact key. Semantic memory retrieves by **meaning** — "what does Alice prefer about billing?" finds the right memory even if it was stored with different words.

This works by attaching an **embed function** to the store. Every write converts the text to a float vector. Every search converts the query to a vector, then finds the nearest stored vectors by cosine similarity.

```python
# Factory: attach embed function to enable semantic search
def _embed_fn(texts: list[str]) -> list[list[float]]:
    return embeddings.embed_documents(texts)

store = InMemoryStore(index={"embed": _embed_fn, "dims": 384})
```

**Graceful degradation:** If embeddings can't load (test env, no model downloaded), `get_memory_store()` catches the exception and returns a store without semantic search. Exact-key lookups still work.

---

## 📌 CONCEPT 38: Memory Retrieval Node

### 🔍 WHAT

`memory_retrieval_node` runs **before** `classify_ticket_node` — it fetches relevant memories for the current user so every downstream agent has full context.

```python
async def memory_retrieval_node(state: AgentState, config: RunnableConfig) -> dict:
    query = state["messages"][-1].content  # last message = search query
    user_id = config.get("configurable", {}).get("user_id", "anonymous")
    memories = await store.asearch(user_id=user_id, query=query)
    return {"retrieved_memories": [m["text"] for m in memories]}
```

**Key design decisions:**
1. No `@with_retry` — this calls our own store, not an external LLM API. Rate-limit retries don't apply.
2. Failures return `[]`, never raise — memory must never block the ticket flow.
3. `user_id` defaults to `"anonymous"` — unauthenticated users still get a namespace.

---

## 📌 CONCEPTS 39–40: Summarisation & Memory Compression

### 🔍 WHAT

`memory_write_node` runs **after** `respond_node`. It calls `_summarise_for_memory()` to compress the conversation into one sentence, then writes it to the store.

```python
async def _summarise_for_memory(messages, final_response) -> str:
    llm = get_fast_llm()   # cheapest model — summaries don't need reasoning power
    user_msgs = [m.content for m in messages if m.type == "human"]
    ticket_summary = " ".join(user_msgs[-3:])  # last 3 human messages max
    prompt = [SystemMessage("Summarise in ONE sentence..."), HumanMessage(ticket_summary)]
    response = await llm.ainvoke(prompt)
    return response.content.strip()
```

**Why `get_fast_llm()` for summarisation?** Memory summaries are compression tasks — they need a model that follows instructions, not one that reasons deeply. `llama-3.1-8b-instant` is 5× cheaper than the research model and fast enough for this.

**Three-layer failure safety:**
1. Summarisation fails → `return {}` (response already delivered)
2. Store write fails → `return {}` (response already delivered)
3. Either missing input → `return {}` (nothing to summarise)

The response is always delivered. Memory is best-effort.

---

## 📌 CONCEPT 41: Config-Driven Workflows

### 🔍 WHAT

Memory behaviour is controlled by settings, not hardcoded:
- `MEMORY_TOP_K` — how many memories `asearch()` returns (default: 5)
- `MEMORY_SUMMARIZE_AFTER_TURNS` — how often to compress (default: 10)

```python
# In asearch() — reads settings to determine default limit
k = limit if limit is not None else settings.memory_top_k
```

Swap `MEMORY_TOP_K=3` in `.env` → all memory searches return at most 3 results. Zero code changes.

### Graph topology after Module 5:

```
[START]
   │
   ▼
[memory_retrieval]  ← NEW: fetches user memories
   │
   ▼
[classify_ticket]
   │  conditional_edge(route_ticket)
   ├──► [billing_node]   ──┐
   ├──► [technical_node] ──┤
   ├──► [general_node]   ──┼──► [respond] ──► [memory_write] ──► [END]
   └──► [research_node]  ──┘                        ↑
                                               NEW: writes memory summary
```

---

### ✅ MODULE 5 CHECKPOINT

Three questions to answer confidently:
1. What is the difference between `retrieved_memories` in AgentState and `InMemoryStore`? Which is short-term and which is long-term?
2. Why does `memory_retrieval_node` NOT use `@with_retry`, but `memory_write_node` DOES?
3. Why is `get_memory_store()` `lru_cached` but `get_checkpointer()` is not?

**Hands-on exercise:**
Send two separate tickets as the same user. On the second ticket, verify `retrieved_memories` in the state contains a summary of the first. Then change `MEMORY_TOP_K=1` and verify only the single most relevant memory is retrieved.

---

---

# MODULE 6 — STREAMING
*Concepts 42–51 | Unlocks: real-time CLI with token-by-token output*

---

## 📌 CONCEPT 42: graph.stream()  [Module 6]
**Layer:** Delivery | **Complexity:** Beginner

### 🔍 WHAT
`graph.stream()` is the **synchronous** variant of streaming. It's a regular Python generator — no `async`/`await` needed. Each `yield` gives you one event from the graph as nodes complete.

**Mental model:** Like reading a file line-by-line with `for line in file:` — you process each piece as it arrives rather than waiting for the whole thing.

```python
# Synchronous streaming — blocks the thread while running
for event in graph.stream(initial_state, config=config, stream_mode="updates"):
    for node_name, node_output in event.items():
        print(f"Node completed: {node_name}")
```

### 🤔 WHY
Without streaming, `graph.invoke()` blocks until the entire graph completes — including all LLM calls. For a graph with 5 nodes each taking 1–2 seconds, the user stares at a blank screen for 5–10 seconds. Streaming gives them incremental feedback.

### 📍 WHEN TO USE
✅ Simple scripts, Jupyter notebooks, synchronous Flask endpoints
❌ FastAPI, async CLI (use `graph.astream()` — event loops conflict with `graph.stream()`)
❌ Production async applications — use `astream()` everywhere

### ⚖️ PROS & CONS
| Pros | Cons |
|------|------|
| No async/await — simpler in sync contexts | Blocks the event loop if called from async code |
| Works in Jupyter without asyncio setup | Can't be used alongside `async for` |
| Same API as `astream()` — easy to upgrade | Rare use case — most apps are already async |

### 💡 BEST PRACTICES
- Use `graph.stream()` only in truly synchronous scripts. The moment you need `asyncio`, switch to `astream()`.
- Both share the same `stream_mode` options — learning one transfers to the other.

---

## 📌 CONCEPT 43: graph.astream()  [Module 6]
**Layer:** Delivery | **Complexity:** Intermediate

### 🔍 WHAT
`graph.astream()` is an **async generator** — it yields events one at a time using Python's `async for` syntax. The graph runs in the same event loop as your application; no extra threads needed.

**Mental model:** Like `ainvoke()` but instead of one big return value at the end, you get a steady stream of smaller results as each node finishes.

```python
# Async streaming — integrates cleanly with FastAPI, async CLI, WebSockets
async for event in graph.astream(state, config=config, stream_mode="updates"):
    # event arrives the moment a node completes — not at the end
    for node_name, node_output in event.items():
        print(f"  [{node_name}] completed")
```

### 🤔 WHY
In our CLI, the user types a billing question. Without `astream()`, they wait 3–5 seconds with no feedback. With `astream()`:
- After ~0.5s: `[Classifying]` appears
- After ~1s: `[Billing agent]` appears  
- After ~2s: `[Generating response]` appears
- Tokens stream word-by-word

The perceived latency drops from "5 seconds of nothing" to "immediate feedback".

### 📍 WHEN TO USE
✅ Any async application (FastAPI, async CLI, WebSocket handlers)
✅ When you need progress feedback during long multi-node graphs
✅ When doing two-pass streaming (updates then messages)
❌ Don't use in sync functions — `async for` requires an event loop

### 🌍 WHERE IT FITS
```
main.py (_stream_response)
  │
  ├── Pass 1: graph.astream(initial_state, stream_mode="updates")
  │     Yields after each node completes
  │     → collects ticket_type, active_agent, confidence
  │     → prints progress labels to terminal
  │
  └── Pass 2: graph.astream({}, stream_mode="messages")
        Resumes from checkpoint (passes {} not initial_state)
        Yields AIMessageChunk as LLM generates
        → prints tokens word-by-word
```

### ⚖️ PROS & CONS
| Pros | Cons |
|------|------|
| Non-blocking — other tasks run while graph executes | Requires async context throughout the call stack |
| Works with FastAPI, WebSockets, SSE natively | Two-pass pattern needed for metadata + tokens |
| Same checkpointer integration as `ainvoke()` | Slightly more complex test setup (async generators) |

### 💡 BEST PRACTICES
1. **Pass `{}` in Pass 2** — not `initial_state`. Passing initial_state restarts the graph; `{}` resumes from checkpoint.
2. **Filter by `langgraph_node` in messages mode** — many nodes emit messages, not just `respond`.
3. **All print/display in delivery layer** — nodes never call `print()`. `main.py` owns all output.
4. **Handle the error case** — `stream_mode="updates"` events include `error` field; check it before starting Pass 2.

### 🏗️ PRODUCTION IMPLEMENTATION
```python
# main.py — _stream_response() — the two-pass pattern
async def _stream_response(graph, initial_state: dict, config: dict) -> None:
    ticket_type = "unknown"
    active_agent = "general"
    confidence = 0.0
    final_error = None

    # ── Pass 1: updates mode — node progress + metadata ──────────────
    async for event in graph.astream(initial_state, config=config, stream_mode="updates"):
        for node_name, node_output in event.items():
            _print_node_progress(node_name)
            if isinstance(node_output, dict):
                if "ticket_type" in node_output and node_output["ticket_type"]:
                    ticket_type = node_output["ticket_type"]
                if "error" in node_output and node_output["error"]:
                    final_error = node_output["error"]

    if final_error:
        print(f"\nAgent [error]: {final_error}\n")
        return

    # ── Pass 2: messages mode — token streaming ───────────────────────
    print(f"\nAgent [{ticket_type}]: ", end="", flush=True)
    async for chunk, metadata in graph.astream(
        {},          # ← empty: resume from checkpoint, don't restart
        config=config,
        stream_mode="messages",
    ):
        if metadata.get("langgraph_node") == "respond" and chunk.content:
            print(chunk.content, end="", flush=True)
    print("\n")
```

### 🧪 TESTS
```python
# tests/unit/test_streaming.py — TestStreamVsInvoke
async def test_astream_is_async_generator(simple_graph):
    """astream() must return an async iterable, not a coroutine."""
    import inspect
    result = simple_graph.astream({"messages": []}, config={"configurable": {"thread_id": "t"}})
    assert inspect.isasyncgen(result) or hasattr(result, "__aiter__")
```

### ✅ CONCEPT CHECKPOINT
1. What is the difference between `graph.stream()` and `graph.astream()`?
2. Why does Pass 2 pass `{}` instead of `initial_state`?
3. What happens if you call `graph.stream()` from inside an `async def` function?

---

## 📌 CONCEPT 44: Stream Modes  [Module 6]
**Layer:** Delivery | **Complexity:** Beginner

### 🔍 WHAT
`stream_mode` tells LangGraph **what to put in each yielded event**. There are three modes:

| Mode | Yields | Use For |
|------|--------|---------|
| `"updates"` | `{node_name: partial_state_dict}` | Progress + metadata collection |
| `"values"` | Full `AgentState` dict | Debugging only |
| `"messages"` | `(AIMessageChunk, metadata)` tuples | Token streaming to users |

### 🤔 WHY
One streaming call can only have one `stream_mode`. LangGraph can't yield both node-completion events AND token chunks in the same call — they have fundamentally different shapes. The two-pass pattern solves this by using each mode for what it's best at.

### 📍 WHEN TO USE
- `"updates"`: **default for production** — lightweight, shows what changed
- `"values"`: **debugging only** — emits entire state on every node (expensive for large states)
- `"messages"`: **user-facing token display** — only works when node calls `llm.astream()` internally

### ⚖️ PROS & CONS — updates vs values
| | `"updates"` | `"values"` |
|--|-------------|------------|
| Payload size | Small — only changed fields | Large — entire state every time |
| Use case | Production progress + metadata | Debugging, test assertions |
| CPU/memory cost | Low | High (copies full state per node) |
| Visibility | What changed | Full snapshot |

### 🏗️ PRODUCTION PATTERN
```python
# ✅ updates mode — for production progress tracking
async for event in graph.astream(state, config=config, stream_mode="updates"):
    for node_name, delta in event.items():
        # delta is ONLY the fields this node returned — not the whole state
        logger.debug("node=%s changed_fields=%s", node_name, list(delta.keys()))

# ✅ messages mode — for token streaming
async for chunk, metadata in graph.astream({}, config=config, stream_mode="messages"):
    node = metadata.get("langgraph_node", "")
    if node == "respond" and chunk.content:
        print(chunk.content, end="", flush=True)

# ⚠️ values mode — debugging only, gate behind settings.graph_debug
if settings.graph_debug:
    async for full_state in graph.astream(state, config=config, stream_mode="values"):
        logger.debug("full_state_snapshot: %s", list(full_state.keys()))
```

### ✅ CONCEPT CHECKPOINT
1. Why can't you use `stream_mode="messages"` to also get node progress labels?
2. What would be the problem with using `stream_mode="values"` in a production CLI that handles 1000 tickets/day?
3. In `"updates"` mode, does an event contain ALL fields or only the fields the node changed?

---

## 📌 CONCEPT 45: values mode  [Module 6]
**Layer:** Delivery | **Complexity:** Beginner

### 🔍 WHAT
`stream_mode="values"` yields the **complete AgentState** after every single node completes — like a series of snapshots.

```python
async for full_state in graph.astream(state, config=config, stream_mode="values"):
    # full_state is the entire AgentState dict at this point in the graph
    print(full_state["ticket_type"])     # visible after classify_ticket runs
    print(full_state["active_agent"])    # visible after routing runs
    print(full_state["final_response"])  # visible after respond runs (last snapshot)
```

### 📍 WHEN TO USE
✅ Debugging — "what did the full state look like after each step?"
✅ Visualisation tools — building a graph execution tracer
❌ Production streaming — use `"updates"` instead (much lighter payload)

### 🔑 The hidden cost
With a `messages` field holding 50 conversation turns and `retrieved_docs` holding 20 documents, each `"values"` event copies the entire structure. A 7-node graph emits 7 full copies. Use `"updates"` in production.

---

## 📌 CONCEPT 46: updates mode  [Module 6]
**Layer:** Delivery | **Complexity:** Beginner

### 🔍 WHAT
`stream_mode="updates"` yields a dict of `{node_name: partial_state_dict}` — only the fields that node actually changed. This is the most **production-appropriate** mode for progress tracking.

```python
async for event in graph.astream(state, config=config, stream_mode="updates"):
    # event = {"classify_ticket": {"ticket_type": "billing", "classification_confidence": 0.91}}
    # NOT the full state — only what classify_ticket_node returned
    for node_name, changes in event.items():
        print(f"  [{node_name}] → {list(changes.keys())}")
```

### 🔑 Why updates mode works with the two-pass pattern
Pass 1 uses `"updates"` to collect `ticket_type`, `active_agent`, and `confidence` from node outputs as they flow through. These are captured from the partial state dicts — exactly what each node returned. Then Pass 2 uses the checkpoint for token streaming.

### 💡 BEST PRACTICES
- Always check `isinstance(node_output, dict)` before accessing keys — some internal events may not be dicts.
- Use `node_output.get("field")` not `node_output["field"]` — not every node writes every field.

---

## 📌 CONCEPT 47: messages mode  [Module 6]
**Layer:** Delivery | **Complexity:** Intermediate

### 🔍 WHAT
`stream_mode="messages"` yields `(AIMessageChunk, metadata)` tuples as the LLM generates tokens. This is the mode that gives true word-by-word streaming to users.

**Mental model:** Instead of "here's the complete paragraph", you get "h", "e", "r", "e", " ", "i", "s" — one chunk per `print()` call, appearing as they're generated.

```python
async for chunk, metadata in graph.astream({}, config=config, stream_mode="messages"):
    # chunk: AIMessageChunk with .content = "" or a few tokens
    # metadata: {"langgraph_node": "respond", "langgraph_step": 4, ...}
    node = metadata.get("langgraph_node", "")
    if node == "respond" and chunk.content:
        print(chunk.content, end="", flush=True)
```

### 🤔 WHY — The user experience difference
Without token streaming:
```
[Classifying] [Billing agent] [Generating response]
Agent: Your refund of $47.99 has been processed and will appear within 3–5 business days.
# ← 3 seconds of nothing, then all at once
```

With token streaming:
```
[Classifying] [Billing agent]
Agent [billing • billing • conf=91%]: Your refund of $47.99 has been processed...
# ← tokens appear word by word — feels instant
```

### 📍 WHEN TO USE
✅ Any LLM call the user is waiting on in real time
✅ CLI apps, WebSocket endpoints, SSE streams
✅ When `streaming_enabled=True` in state (CLI path)
❌ Batch processing, automated pipelines — use `ainvoke()` (no UI waiting for tokens)
❌ Non-LLM nodes — those don't emit message chunks

### 🌍 WHERE IT FITS
```
respond_node (graph layer)
  if streaming_enabled:
      chunks = []
      async for chunk in llm.astream(prompt, config=config):  ← LLM streams
          chunks.append(chunk.content)
      response_text = "".join(chunks)

main.py Pass 2 (delivery layer)
  async for chunk, metadata in graph.astream({}, stream_mode="messages"):
      if node == "respond" and chunk.content:
          print(chunk.content, end="", flush=True)  ← user sees tokens
```

### ⚖️ PROS & CONS
| Pros | Cons |
|------|------|
| Dramatically improves perceived latency | Only works with `llm.astream()` in the node |
| Works with FastAPI SSE, WebSocket natively | Must filter by `langgraph_node` — all nodes emit |
| No code change at the LangGraph level | Requires two-pass architecture with `"updates"` |
| `streaming_enabled` flag makes it opt-in | Empty `chunk.content` values must be skipped |

### 🏗️ PRODUCTION IMPLEMENTATION
The key insight is **two components working together**:

**Component 1 — Inside the node (graph layer):**
```python
# agents/nodes.py — respond_node
streaming_enabled = state.get("streaming_enabled", False)

if streaming_enabled:
    # llm.astream() emits chunks that LangGraph's messages mode captures
    chunks = []
    async for chunk in llm.astream(prompt, config=config):
        chunks.append(chunk.content)
    response_text = "".join(chunks)
else:
    # llm.ainvoke() — no streaming, used in tests and batch mode
    response = await llm.ainvoke(prompt, config=config)
    response_text = response.content
```

**Component 2 — In the delivery layer:**
```python
# main.py — _stream_response() Pass 2
async for chunk, metadata in graph.astream(
    {},                          # empty — resume from Pass 1 checkpoint
    config=config,
    stream_mode="messages",
):
    node = metadata.get("langgraph_node", "")
    if node == "respond" and hasattr(chunk, "content") and chunk.content:
        print(chunk.content, end="", flush=True)
```

### 🧪 TESTS
```python
# tests/unit/test_streaming.py — TestMessagesMode
async def test_messages_mode_yields_chunks_from_llm_node(simple_streaming_graph):
    chunks = []
    async for chunk, metadata in simple_streaming_graph.astream(
        {"messages": [HumanMessage(content="test")]},
        config={"configurable": {"thread_id": "t1"}},
        stream_mode="messages",
    ):
        if metadata.get("langgraph_node") == "llm_node" and chunk.content:
            chunks.append(chunk.content)
    assert len(chunks) > 0
    assert "".join(chunks) != ""
```

### ✅ CONCEPT CHECKPOINT
1. Why does `"messages"` mode require `llm.astream()` inside the node — not `llm.ainvoke()`?
2. What does `metadata.get("langgraph_node")` let you do that wouldn't be possible without filtering?
3. Why do we pass `{}` instead of `initial_state` in Pass 2?

---

## 📌 CONCEPT 48: debug mode  [Module 6]
**Layer:** Graph | **Complexity:** Beginner

### 🔍 WHAT
LangGraph's debug mode prints detailed internal execution traces — node inputs, outputs, transitions — to stdout. It's activated by passing `debug=True` to `graph.compile()`.

```python
# graph.py — build_graph()
from core.settings import get_settings

def build_graph(checkpointer=None, debug: bool = False):
    settings = get_settings()
    return (
        GraphBuilder(AgentState)
        ...
        .build(
            checkpointer=checkpointer or get_checkpointer(),
            debug=debug or settings.graph_debug,  # ← gated by settings
        )
    )
```

### 📍 WHEN TO USE
✅ Diagnosing unexpected routing — "why did billing go to general?"
✅ During development when node outputs look wrong
❌ Production — generates enormous log output per invocation

### 🔑 The production rule
Always gate debug mode behind `settings.graph_debug` (defaults to `False`). Never hardcode `debug=True`. In `.env`: `GRAPH_DEBUG=true` to enable for a specific session.

---

## 📌 CONCEPT 49: Token Streaming  [Module 6]
**Layer:** Graph + Delivery | **Complexity:** Intermediate

### 🔍 WHAT
Token streaming is the technique of **collecting `AIMessageChunk` objects** from `llm.astream()` and displaying them immediately rather than waiting for the complete response. Each chunk contains `""` to a few tokens of content.

**Mental model:** Like watching a sentence being typed character-by-character vs waiting for the person to finish typing before you see anything.

### 🤔 WHY
Groq's `llama-3.3-70b-versatile` takes 2–4 seconds to generate a full response. Without token streaming, the user sees a blank screen for 4 seconds. With streaming, they see words appearing after ~0.3 seconds — a 10× improvement in perceived responsiveness.

### 🌍 WHERE IT FITS — The full path of a token
```
Groq API (cloud)
  │  generates tokens one by one
  ▼
llm.astream() in respond_node
  │  yields AIMessageChunk objects
  │  (captured internally: chunks.append(chunk.content))
  ▼
graph.astream(stream_mode="messages")
  │  re-yields the chunks with metadata
  ▼
main.py _stream_response() Pass 2
  │  filters: node == "respond" AND chunk.content != ""
  ▼
print(chunk.content, end="", flush=True)
  │
  ▼
User's terminal — tokens appear word by word
```

### 🏗️ PRODUCTION PATTERN
```python
# The streaming_enabled flag: one env var switches the entire pipeline

# In state (_build_initial_state in main.py):
"streaming_enabled": True    # ← CLI always streams

# In base_state fixture (tests/conftest.py):
"streaming_enabled": False   # ← tests use ainvoke (deterministic, fast)

# In respond_node (agents/nodes.py):
streaming_enabled = state.get("streaming_enabled", False)
if streaming_enabled:
    chunks = []
    async for chunk in llm.astream(prompt, config=config):
        chunks.append(chunk.content)
    response_text = "".join(chunks)
else:
    response = await llm.ainvoke(prompt, config=config)
    response_text = response.content
```

### 💡 BEST PRACTICES
1. **`flush=True` is mandatory** — without it, stdout buffers tokens and they appear in bursts, not word-by-word.
2. **Skip empty chunks** — `chunk.content` can be `""` for metadata-only chunks. Always `if chunk.content:`.
3. **`streaming_enabled` flag** — makes streaming opt-in. Tests run with `False` → fast, deterministic, no async generator setup.
4. **Join chunks in the node** — `respond_node` returns a complete `response_text`. The delivery layer decides how to display it.

### ⚠️ Common Mistakes
- **`print(chunk.content)` with default `end="\n"`** — each token gets its own line. Use `end=""`.
- **Not checking `chunk.content`** — empty chunks will print nothing visually but slow down the loop.
- **Streaming in the node AND printing in main.py** — nodes must NEVER print. Collect chunks in the node, display in the delivery layer.

### 🧪 TESTS
```python
# tests/unit/test_streaming.py — TestTokenStreaming
async def test_respond_node_uses_astream_when_streaming_enabled(
    base_state, runnable_config
):
    """streaming_enabled=True triggers llm.astream() path."""
    base_state["streaming_enabled"] = True
    mock_llm = AsyncMock()

    # astream must be called, not ainvoke
    async def fake_astream(*args, **kwargs):
        yield AIMessageChunk(content="Hello")
        yield AIMessageChunk(content=" world")

    mock_llm.astream = fake_astream

    with patch("agents.nodes.get_research_llm", return_value=mock_llm):
        result = await respond_node(base_state, runnable_config)

    assert result["final_response"] == "Hello world"
    # astream was used, not ainvoke
    mock_llm.ainvoke.assert_not_called()
```

---

## 📌 CONCEPT 50: Event Streaming  [Module 6]
**Layer:** Delivery | **Complexity:** Advanced

### 🔍 WHAT
`graph.astream_events()` is a more granular streaming API that yields **typed event dicts** for every internal LangGraph action — tool calls, LLM starts, LLM ends, chain runs, etc.

```python
async for event in graph.astream_events(state, config=config, version="v2"):
    # event is a dict: {"event": "on_chat_model_stream", "data": {...}, "name": "...", ...}
    kind = event["event"]
    if kind == "on_chat_model_stream":
        chunk = event["data"].get("chunk")
        if chunk and chunk.content:
            print(chunk.content, end="", flush=True)
```

### 🆚 astream_events vs astream(stream_mode="messages")
| | `astream_events` | `astream(stream_mode="messages")` |
|--|--|--|
| Granularity | Every LangChain event (tool start, LLM end, etc.) | Only message chunks |
| Complexity | High — must parse event type strings | Low — `(chunk, metadata)` tuples |
| Use case | Custom observability, detailed traces | Standard token streaming UIs |
| Stability | Requires `version="v2"` parameter | Stable API |

**Production recommendation:** Use `astream(stream_mode="messages")` for token streaming. Use `astream_events()` only when you need fine-grained event types (e.g. detecting tool call starts/ends for a custom UI).

---

## 📌 CONCEPT 51: Streaming UIs  [Module 6]
**Layer:** Delivery | **Complexity:** Beginner

### 🔍 WHAT
A "Streaming UI" is the pattern of displaying LLM tokens and graph progress in real time. In our app this means: the CLI prints node progress labels as nodes complete, then prints tokens word-by-word as the LLM generates.

**The architectural rule:** **All display code lives in `main.py` only.** Nodes return state dicts. `main.py` decides how to render them.

### 🤔 WHY — The Clean Architecture Principle
If nodes called `print()`, they would be tied to the CLI delivery channel. You couldn't reuse the same nodes in:
- A FastAPI endpoint (can't print to stdout)
- A Slack bot (prints go nowhere)
- A test (stdout captured, breaks assertions)

By keeping all output in `main.py`, every delivery channel (CLI, API, bot) only needs its own `main.py` equivalent — no node code changes.

### 🏗️ PRODUCTION PATTERN — The complete UI pipeline

```python
# main.py — _print_node_progress(): progress labels
_NODE_LABELS = {
    "classify_ticket": "Classifying",
    "billing_node":    "Billing agent",
    "technical_node":  "Technical agent",
    "general_node":    "General agent",
    "research_node":   "Research agent",
    "respond":         "Generating response",
}
_SILENT_NODES = {"memory_retrieval", "memory_write"}  # fast; no label needed
_INTERNAL_NODES = {"__start__"}                        # LangGraph internal; never shown

def _print_node_progress(node_name: str) -> None:
    """Prints a brief progress label when a node completes."""
    if node_name in _INTERNAL_NODES or node_name in _SILENT_NODES:
        return
    label = _NODE_LABELS.get(node_name, node_name)
    print(f"  [{label}]", end="\r", flush=True)


# main.py — _stream_response(): orchestrates the two-pass display
async def _stream_response(graph, initial_state: dict, config: dict) -> None:
    ticket_type = "unknown"
    active_agent = "general"
    confidence = 0.0
    final_error = None

    # Pass 1: show progress labels + collect metadata
    async for event in graph.astream(initial_state, config=config, stream_mode="updates"):
        for node_name, node_output in event.items():
            _print_node_progress(node_name)
            if isinstance(node_output, dict):
                if node_output.get("ticket_type"):
                    ticket_type = node_output["ticket_type"]
                if node_output.get("active_agent"):
                    active_agent = node_output["active_agent"]
                if "classification_confidence" in node_output:
                    confidence = node_output["classification_confidence"]
                if node_output.get("error"):
                    final_error = node_output["error"]

    if final_error:
        print(f"\nAgent [error]: {final_error}\n")
        return

    # Pass 2: stream response tokens word-by-word
    print(f"\nAgent [{ticket_type} • {active_agent} • conf={confidence:.0%}]: ",
          end="", flush=True)
    token_count = 0
    async for chunk, metadata in graph.astream(
        {}, config=config, stream_mode="messages"
    ):
        if metadata.get("langgraph_node") == "respond" \
                and hasattr(chunk, "content") and chunk.content:
            print(chunk.content, end="", flush=True)
            token_count += 1
    print("\n")
```

### 💡 BEST PRACTICES
1. **`end="\r"` for progress labels** — overwrites the same line rather than stacking. The final response then appears on a fresh line.
2. **Silent nodes list** — `memory_retrieval` and `memory_write` run fast. Showing them creates flickery noise. Gate visibility explicitly.
3. **`flush=True` everywhere** — Python's stdout buffers by default. Without flush, tokens appear in batches rather than word-by-word.
4. **Never print in nodes** — if you need to debug a node, use `logger.debug()`. The test suite captures logs, not stdout.

### 🧪 TESTS
```python
# tests/unit/test_invoke.py — TestPrintNodeProgress
def test_classify_ticket_shows_label(capsys):
    from main import _print_node_progress
    _print_node_progress("classify_ticket")
    assert "Classifying" in capsys.readouterr().out

def test_internal_start_node_produces_no_output(capsys):
    from main import _print_node_progress
    _print_node_progress("__start__")
    assert capsys.readouterr().out == ""

def test_memory_retrieval_is_silent(capsys):
    from main import _print_node_progress
    _print_node_progress("memory_retrieval")
    assert capsys.readouterr().out == ""
```

### ✅ MODULE 6 CHECKPOINT

Three questions to answer confidently:
1. Why does Pass 2 of `_stream_response()` pass `{}` as the input instead of `initial_state`?
2. What is the `streaming_enabled` flag in `AgentState` for — what does it control, and why is it `False` in tests?
3. Why is `end="\r"` used for node progress labels but `end=""` used for token output?

**Hands-on exercise:**
Change `streaming_enabled` in `_build_initial_state()` to `False` and run the CLI. The response should still work but appear all at once. Then re-enable it. This confirms the `streaming_enabled` flag correctly switches the code path in `respond_node`.

---

### Graph topology after Module 6:
```
[START]
  │
  ▼
[memory_retrieval]  ← reads long-term memories (silent node)
  │
  ▼
[classify_ticket]   ← HF zero-shot: billing | technical | general | research
  │
  ▼ (conditional edge: route_ticket)
  ├──► [billing_node]   ──┐
  ├──► [technical_node] ──┤
  ├──► [general_node]   ──┼──► [respond] ──► [memory_write] ──► [END]
  └──► [research_node]  ──┘       ↑
                               Module 6: respond_node now calls
                               llm.astream() when streaming_enabled=True
```

**CLI output after Module 6:**
```
════════════════════════════════════════════════════
  Support Research Agent
  Provider : groq
  Session  : a3f7c2d1...
  Streaming: enabled
  Type your support ticket. Ctrl+C to exit.
════════════════════════════════════════════════════

You: My invoice shows double charge

  [Classifying]
  [Billing agent]
  [Generating response]

Agent [billing • billing • conf=91%]: I can see the duplicate charge on your account.
Let me look into this — it appears the payment processor submitted the transaction twice
on June 23rd. I've initiated a refund for $47.99 which will appear within 3–5 business days.

```

---

## Module 7: Human-in-the-Loop & Workflow Patterns
**Session 4 · Concepts 52–72 · Unlocks: approval flows, SLA tracking, audit trails, compliance**

Module 7 is the transition from automated pipelines to **supervised AI** — systems where humans retain authority over high-stakes decisions. After this module, the app can pause mid-run, collect a human decision, and resume exactly where it left off. It also gains SLA deadlines, immutable audit logs, PII detection, and ticket lifecycle tracking.

The central lesson: LangGraph's persistence layer (the checkpointer from Module 4) is what makes HITL possible. Without a checkpointer there is nothing to resume — the graph state would be gone the moment execution paused.

---

## 📌 CONCEPT 52 & 53: Interrupt & interrupt_before  [Module 7]
**Layer:** Graph | **Complexity:** Intermediate

### 🔍 WHAT

LangGraph provides two mechanisms for pausing a graph to wait for human input:

**`interrupt()` (Concept 52) — runtime dynamic interrupt**
A function you call *inside* a node to pause mid-execution. The node suspends at that line, surfaces a payload to the caller, and resumes from that exact line when the caller calls `Command(resume=value)`.

**`interrupt_before` (Concept 53) — compile-time static interrupt**
A compile kwarg: `graph.compile(interrupt_before=["node_name"])`. The graph pauses *before* the named node runs, every time execution would reach it. Resume is handled by passing a state update then calling `ainvoke(None, config)`.

**Mental model for `interrupt()`:** A `breakpoint()` call inside a function — execution freezes, you inspect the program, you tell it to continue.

**Mental model for `interrupt_before`:** A "require approval to proceed" gate on a specific door. Anyone who reaches that door waits until the operator waves them through.

### 🤔 WHY

Without HITL, the graph auto-approves everything. A customer requests a $5,000 refund — the billing node processes it, the LLM says "refund approved", and the money is gone. No human ever saw it.

HITL lets you define a threshold (in our app: $500) above which a human must review before the graph continues.

**The problem that revealed itself — the Python 3.10 interrupt() bug:**

When we tried using `interrupt()` inside `billing_node` on Python 3.10 with LangGraph 1.2.5, it failed with:

```
RuntimeError: No active LangGraph checkpoint found.
get_config() called outside of a LangGraph node execution context.
```

**Root cause:** In `langgraph/_internal/_runnable.py`, there is a constant:

```python
ASYNCIO_ACCEPTS_CONTEXT = sys.version_info >= (3, 11)
```

On Python 3.10, `loop.run_in_executor()` does **not** copy the current context's variables to the worker thread. This means `var_child_runnable_config` (a `contextvars.ContextVar` that LangGraph uses to track the active node's config) is `None` inside the executor thread. When `interrupt()` calls `get_config()` to record the pause point, it reads `None` and raises `RuntimeError`.

**Python 3.11 fixed `run_in_executor` context propagation.** On 3.11+, context vars are copied into the thread automatically — `interrupt()` works fine.

**Solution for Python 3.10: use `interrupt_before` instead.**

`interrupt_before` is a compile-time mechanism — it operates at the graph execution layer before any node code runs, so it doesn't depend on context var propagation inside a thread. It works on Python 3.10 without any workarounds.

### 📍 WHEN TO USE

| Mechanism | When to choose |
|-----------|---------------|
| `interrupt()` | Python 3.11+, you need to pause mid-node (e.g. pause after computing an amount but before taking action), the pause point is dynamic (not always the same node) |
| `interrupt_before` | Python 3.10 compatibility required, the pause point is always the same node, simplicity matters |

**Decision rule for this project:** We target Python 3.10+, so `interrupt_before` is the safe universal choice.

✅ `interrupt_before=["approval_check"]` — safe on all Python versions  
✅ `interrupt()` — use on Python 3.11+ only, when pause point must be dynamic  
❌ Don't use `interrupt()` on Python 3.10 — silent `RuntimeError` at the context var level  
❌ Don't use `interrupt_before` without a checkpointer — the graph state won't persist across the pause  

### 🌍 WHERE IT FITS

```
graph.compile(
    checkpointer=sqlite_saver,
    interrupt_before=["approval_check"]   ← pause point declared here
)

┌──────────────────────────────────────────────────────────┐
│ Pass 1: graph.astream(initial_state, stream_mode="updates")│
│                                                            │
│   [memory_retrieval] → [classify_ticket] → [billing_node] │
│                                               │            │
│                              pending_approval? ─YES→       │
│                              [route_after_billing returns   │
│                               "approval_check"]            │
│                                               │            │
│                         ✋ PAUSE HERE ◄─ interrupt_before  │
│                         (graph saves state to SQLite)      │
│                         state.next = ("approval_check",)   │
└──────────────────────────────────────────────────────────┘

     [Caller reads state.values["pending_approval"]]
     [Displays approval request to operator]
     [Operator types: approved / rejected / escalated]

┌──────────────────────────────────────────────────────────┐
│ Resume:                                                    │
│   graph.update_state(config, {"approval_decision": "approved"})  │
│   graph.astream(None, config=config, stream_mode="updates")      │
│                                                            │
│   [approval_check] → [respond] → [memory_write] → END     │
└──────────────────────────────────────────────────────────┘
```

### ⚖️ PROS & CONS

| Pros | Cons |
|------|------|
| Works on Python 3.10 (unlike `interrupt()`) | Pause point is static — always the same node(s) |
| Simple — no changes to node code | Can't pass a dynamic payload to the caller (payload goes via state) |
| Requires checkpointer (enforces good persistence habits) | If `interrupt_before` node is never reached, pause never triggers |
| Graph state is fully saved — crash-safe | Operator must resume via `ainvoke(None)`, not a fresh `ainvoke(state)` |

### 🏗️ PRODUCTION IMPLEMENTATION

**`graph.py` — compile with `interrupt_before` as default:**

```python
# graph.py — build_graph() extract (Module 7)

def build_graph(
    checkpointer: Any = None,
    interrupt_before: list[str] | None = None,
    debug: bool = False,
):
    # Default: always include approval_check in interrupt_before.
    # Only fires when billing_node routes to it (large refund detected).
    # Tests can override by passing interrupt_before=[] explicitly.
    if interrupt_before is None:
        interrupt_before = ["approval_check"]

    compile_kwargs: dict[str, Any] = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer
    if interrupt_before:
        compile_kwargs["interrupt_before"] = interrupt_before

    graph = (
        GraphBuilder(AgentState)
        # ...nodes and edges...
        .node("billing_node",   billing_node)
        .node("approval_check", approval_check_node)   # HITL gate
        # ...
        .conditional_edge("billing_node", route_after_billing, {
            "approval_check": "approval_check",
            "respond": "respond",
        })
        .edge("approval_check", "respond")
        .build(**compile_kwargs)
    )
    return graph
```

**`agents/nodes.py` — `billing_node` sets pending_approval, `route_after_billing` decides the path:**

```python
# agents/nodes.py — billing_node + router (Module 7)

def billing_node(state: AgentState, config: RunnableConfig) -> dict:
    """
    WHY synchronous? No LLM calls — only regex matching.
    WHY NOT interrupt() directly? Python 3.10 context var bug (see CLAUDE.md).

    HITL design:
      1. This node detects a large refund and sets pending_approval in state.
      2. route_after_billing() sees pending_approval → routes to approval_check.
      3. interrupt_before=["approval_check"] pauses the graph before it runs.
      4. Caller reads state.values["pending_approval"] for the payload.
      5. Operator inputs a decision; caller calls update_state() then astream(None).
      6. approval_check_node runs and reads approval_decision from state.
    """
    from workflows.approval import (
        _extract_refund_amount, check_approval_required, build_approval_payload
    )

    thread_id = _get_thread_id(config)
    messages = state.get("messages", [])
    ticket_text = messages[-1].content if messages else ""

    refund_amount = _extract_refund_amount(ticket_text)

    if check_approval_required(refund_amount):
        payload = build_approval_payload(refund_amount, thread_id)  # type: ignore[arg-type]
        # Signal HITL — route_after_billing() will see this and route to approval_check
        return {
            "active_agent": "billing",
            "pending_approval": dict(payload),
        }

    # Normal billing — no approval needed
    return {
        "active_agent": "billing",
        "audit_events": [{"node": "billing_node", "action": "ticket_received",
                          "thread_id": thread_id}],
    }


def route_after_billing(state: AgentState) -> str:
    """
    Pure synchronous router. Reads pending_approval — if set, route to
    approval_check (which triggers the interrupt_before pause). Else, respond.
    """
    if state.get("pending_approval"):
        return "approval_check"
    return "respond"
```

### 🧪 TESTS

```python
# tests/unit/test_billing_hitl.py (pattern)

def test_billing_node_sets_pending_approval_for_large_refund(base_state):
    """billing_node must set pending_approval when amount >= $500."""
    base_state["messages"] = [HumanMessage(content="I need a refund of $750")]
    config = {"configurable": {"thread_id": "t-001"}}
    result = billing_node(base_state, config)

    assert result["pending_approval"] is not None
    assert result["pending_approval"]["amount"] == 750.0
    assert result["active_agent"] == "billing"
    # No audit event yet — approval_check_node writes that after decision
    assert "audit_events" not in result


def test_billing_node_no_interrupt_for_small_refund(base_state):
    """billing_node must NOT set pending_approval when amount < $500."""
    base_state["messages"] = [HumanMessage(content="I need a refund of $49.99")]
    config = {"configurable": {"thread_id": "t-001"}}
    result = billing_node(base_state, config)

    assert result.get("pending_approval") is None
    assert result["active_agent"] == "billing"
    assert "audit_events" in result


def test_route_after_billing_routes_to_approval_when_pending(base_state):
    """Router must return 'approval_check' when pending_approval is set."""
    base_state["pending_approval"] = {"amount": 750.0, "message": "..."}
    assert route_after_billing(base_state) == "approval_check"


def test_route_after_billing_routes_to_respond_when_clear(base_state):
    """Router must return 'respond' when no pending_approval."""
    base_state["pending_approval"] = None
    assert route_after_billing(base_state) == "respond"
```

### ✅ CONCEPT CHECKPOINT

1. Why does `interrupt()` fail on Python 3.10 but work on Python 3.11?
2. What is `state.next` and how does the caller use it to detect an interrupt?
3. What is the difference between passing a state dict to `ainvoke(state, config)` vs calling `update_state(config, delta)` then `ainvoke(None, config)`?

**Exercise:** Change the `_REFUND_APPROVAL_THRESHOLD` in `workflows/approval.py` to `$100`. Run the CLI with a ticket mentioning "$150". Verify the graph pauses, enter "rejected", and confirm the audit event is written with `decision="rejected"`.

---

## 📌 CONCEPT 55: Resume  [Module 7]
**Layer:** Delivery | **Complexity:** Intermediate

### 🔍 WHAT

Resuming a paused graph means continuing execution from the point it was interrupted. With `interrupt_before`, the resume sequence has two steps:

1. **`graph.update_state(config, delta)`** — writes new field(s) into the persisted checkpoint *without* triggering a graph run. This is how the operator's decision gets into the state before the paused node sees it.

2. **`graph.ainvoke(None, config)` or `graph.astream(None, config)`** — continues the graph from where it paused. Passing `None` as the input signals "resume, don't restart". The graph reads the updated checkpoint and runs from the paused node onward.

**Mental model:** `update_state` is editing the save file. `ainvoke(None)` is pressing "load and continue" in a game.

### 🤔 WHY

**The critical mistake to avoid:**

```python
# ❌ WRONG — restarts the graph from the entry node (memory_retrieval)
# The operator's decision is ignored; the ticket is re-processed from scratch.
graph.ainvoke({"approval_decision": "approved"}, config=config)

# ✅ CORRECT — writes decision to checkpoint, then resumes from paused node
graph.update_state(config, {"approval_decision": "approved"})
graph.ainvoke(None, config=config)
```

Passing a non-None input to `ainvoke` always starts a new run from the entry node. The `None` input is the specific signal that tells LangGraph "resume the existing checkpoint run".

**Why `update_state` before `ainvoke(None)`?**

With `interrupt_before`, the paused node (`approval_check_node`) hasn't run yet. It needs to read `approval_decision` from state when it does run. `update_state` injects that value into the checkpoint before `ainvoke(None)` continues execution.

### 📍 WHEN TO USE

✅ After any `interrupt_before` pause — always use `update_state` + `ainvoke(None)`  
✅ When the caller needs to inject data into state between two nodes  
❌ Do NOT pass a non-None state dict to `ainvoke` after an interrupt — that restarts from entry  
❌ Do NOT skip `update_state` if the resumed node needs to read the human's decision  

### 🏗️ PRODUCTION IMPLEMENTATION

**`main.py` — `_handle_interrupt()` — the complete resume flow:**

```python
# main.py — _handle_interrupt() (Module 7)

async def _handle_interrupt(graph, state_snapshot, config: dict) -> bool:
    """
    Handles an interrupt_before pause. Reads the pending_approval payload,
    collects the operator decision, then resumes via update_state + astream(None).

    WHY update_state + ainvoke(None), not ainvoke({"approval_decision": ...})?
      ainvoke(state_dict) starts a NEW run from the entry node.
      ainvoke(None) resumes the EXISTING checkpoint from the paused position.
      update_state() injects the decision without triggering a run.
    """
    # ── Read payload from state (set by billing_node) ─────────────────────────
    payload = state_snapshot.values.get("pending_approval")
    if not payload:
        return True   # not a billing interrupt — let caller proceed

    # ── Display approval request to operator ──────────────────────────────────
    print(f"  ⚠  APPROVAL REQUIRED")
    print(f"  {payload.get('message', 'An action requires your approval.')}")
    for key, val in payload.items():
        if key not in {"message", "action", "ticket_type"}:
            print(f"    {key}: {val}")
    print("  Options: approved / rejected / escalated")

    # ── Collect decision ───────────────────────────────────────────────────────
    raw = input("  Decision: ").strip().lower()
    decision = {"approve": "approved", "reject": "rejected",
                "escalate": "escalated"}.get(raw, raw)
    if decision not in {"approved", "rejected", "escalated"}:
        decision = "rejected"

    # ── Resume ────────────────────────────────────────────────────────────────
    # Step 1: write decision into checkpoint WITHOUT triggering a run.
    graph.update_state(config, {"approval_decision": decision})

    # Step 2: resume from paused position (before approval_check_node).
    #   None input = "continue existing run", not "start new run".
    async for event in graph.astream(None, config=config, stream_mode="updates"):
        for node_name, _ in event.items():
            _print_node_progress(node_name)

    return True
```

**`agents/nodes.py` — `approval_check_node` reads the decision after resume:**

```python
# agents/nodes.py — approval_check_node (Module 7)

def approval_check_node(state: AgentState, config: RunnableConfig) -> dict:
    """
    Runs AFTER graph resumes from interrupt_before pause.
    Reads approval_decision (injected by update_state) and records audit event.

    Reads:
        state["approval_decision"]  — "approved" | "rejected" | "escalated"
        state["pending_approval"]   — original request (has amount)
    Writes:
        state["pending_approval"]  → None (cleared)
        state["audit_events"]      → refund_decision event
    """
    thread_id = _get_thread_id(config)
    approval_decision = state.get("approval_decision")
    pending = state.get("pending_approval") or {}
    amount = pending.get("amount", 0.0)

    logger.info(
        "approval_check_node: thread=%s amount=%.2f decision=%s",
        thread_id, amount, approval_decision,
    )

    return {
        "pending_approval": None,       # cleared — decision received
        "audit_events": [{
            "node": "approval_check_node",
            "action": "refund_decision",
            "thread_id": thread_id,
            "amount": amount,
            "decision": approval_decision,
        }],
    }
```

### ✅ CONCEPT CHECKPOINT

1. What happens if you call `graph.ainvoke({"approval_decision": "approved"}, config)` after an `interrupt_before` pause instead of `ainvoke(None, config)`?
2. Why is `graph.update_state()` needed before `ainvoke(None)`?
3. How does `approval_check_node` know the human's decision if it never interacted with the human directly?

---

## 📌 CONCEPT 60: Approval Workflows  [Module 7]
**Layer:** Graph / Service | **Complexity:** Intermediate

### 🔍 WHAT

An approval workflow is the full end-to-end loop: detect a trigger condition → pause for human review → record the decision → continue execution with the decision in state.

In our app, the billing approval workflow is triggered when a customer requests a refund of $500 or more. The `_REFUND_APPROVAL_THRESHOLD` constant in `workflows/approval.py` is the single source of truth for this threshold.

**Mental model:** A purchase order that requires manager sign-off above a certain amount. The purchase order is created automatically; it sits in the manager's queue until they approve or reject it; the downstream process (send the item, issue the refund) only runs after a decision.

### 🔍 THE COMPLETE BILLING HITL FLOW

```
Customer: "I need a refund of $750 for order #8821"

[memory_retrieval]     — retrieves past billing interactions for this user
       │
[classify_ticket]      — HF classifier → ticket_type = "billing"
       │
[billing_node]         — extracts $750.00 via _extract_refund_amount()
                       — check_approval_required(750.0) → True (≥ $500)
                       — builds ApprovalRequest payload
                       — writes pending_approval to state
                       — route_after_billing() → "approval_check"
                              │
                         ✋ interrupt_before=["approval_check"]
                         Graph pauses. State saved to SQLite.
                         state.next = ("approval_check",)

[main.py detects state.next is non-empty]
[reads state.values["pending_approval"]]
[displays to operator]:
  ─────────────────────────────────────
  ⚠  APPROVAL REQUIRED
  Refund of $750.00 requires approval. Approve or reject?
    amount: 750.0
    thread_id: a3f7c2d1...
    reason: customer refund request
  Options: approved / rejected / escalated
  ─────────────────────────────────────
  Decision: _

[operator types "approved"]

graph.update_state(config, {"approval_decision": "approved"})
graph.astream(None, config=config)

[approval_check_node]  — reads approval_decision from state
                       — clears pending_approval
                       — appends audit event: action="refund_decision",
                         amount=750.0, decision="approved"
       │
[respond_node]         — Groq LLM generates response acknowledging approval
       │
[memory_write]         — stores summary of this interaction
       │
      END
```

### 🏗️ PRODUCTION IMPLEMENTATION

**`workflows/approval.py` — the pure functions that power the detection:**

```python
# workflows/approval.py

_REFUND_APPROVAL_THRESHOLD: float = 500.0
"""Refund amounts above this require human approval."""

_AMOUNT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\$\s*(\d+(?:\.\d{1,2})?)"),                        # $750 or $750.00
    re.compile(r"(\d+(?:\.\d{1,2})?)\s*dollars?", re.IGNORECASE),   # 750 dollars
    re.compile(r"refund\s+(?:of\s+)?(\d+(?:\.\d{1,2})?)", re.IGNORECASE),  # refund of 750
    re.compile(r"(\d+(?:\.\d{1,2})?)\s*(?:USD|usd)"),               # 750 USD
]


def _extract_refund_amount(text: str) -> float | None:
    """Extracts the first dollar amount from ticket text. Pure function."""
    for pattern in _AMOUNT_PATTERNS:
        match = pattern.search(text)
        if match:
            try:
                return float(match.group(1).replace(",", ""))
            except (ValueError, IndexError):
                continue
    return None


def check_approval_required(amount: float | None) -> bool:
    """Returns True if the amount requires human approval. Pure function."""
    if amount is None:
        return False
    return amount >= _REFUND_APPROVAL_THRESHOLD


def build_approval_payload(
    amount: float,
    thread_id: str,
    reason: str = "customer refund request",
) -> ApprovalRequest:
    """Builds the structured payload surfaced to the operator. Pure function."""
    return ApprovalRequest(
        message=f"Refund of ${amount:.2f} requires approval. Approve or reject?",
        amount=amount,
        ticket_type="billing",
        action="approve_refund",
        thread_id=thread_id,
        reason=reason,
    )
```

**`ApprovalRequest` TypedDict — what the operator sees:**

```python
class ApprovalRequest(TypedDict):
    """Structured payload passed via state["pending_approval"]."""
    message: str      # Human-readable description shown to operator
    amount: float     # Dollar amount of the refund
    ticket_type: str  # Always "billing" for refund approvals
    action: str       # Always "approve_refund"
    thread_id: str    # For looking up the full ticket if needed
    reason: str       # Why the approval is needed
```

### 🧪 TESTS

```python
# tests/unit/test_approval.py

def test_extract_refund_amount_dollar_sign():
    assert _extract_refund_amount("I need a $750 refund") == 750.0

def test_extract_refund_amount_words():
    assert _extract_refund_amount("Charge me 299.99 dollars") == 299.99

def test_extract_refund_amount_none():
    assert _extract_refund_amount("My billing is wrong") is None

def test_check_approval_required_above_threshold():
    assert check_approval_required(500.0) is True
    assert check_approval_required(500.01) is True

def test_check_approval_required_below_threshold():
    assert check_approval_required(499.99) is False
    assert check_approval_required(None) is False

def test_build_approval_payload_structure():
    payload = build_approval_payload(750.0, "t-001")
    assert payload["amount"] == 750.0
    assert payload["action"] == "approve_refund"
    assert "t-001" in payload["thread_id"]
```

### ✅ CONCEPT CHECKPOINT

1. Why are `_extract_refund_amount`, `check_approval_required`, and `build_approval_payload` pure functions in a separate `workflows/approval.py` rather than inline in `billing_node`?
2. What field does `billing_node` write to signal that HITL is needed?
3. If `check_approval_required` returns `False`, what path does `route_after_billing` take?

---

## 📌 CONCEPT 65: Workflow State Machines  [Module 7]
**Layer:** Service | **Complexity:** Intermediate

### 🔍 WHAT

A workflow state machine models a ticket's lifecycle as a set of explicit **states** and the **transitions** between them. Instead of inferring a ticket's status from scattered boolean flags in state (`is_approved`, `is_escalated`, `is_resolved`), a state machine names every valid status and defines which transitions are legal.

In our app, `TicketStatus` is a `str`-Enum in `state.py` that represents every stage of a ticket's lifecycle.

**Mental model:** A traffic light — it's not just a light that happens to be green or red, it's a machine with explicit states (red, yellow, green) and defined transitions (red → green, green → yellow, yellow → red). Skipping states (green → red) is not allowed.

### 🤔 WHY

Without a state machine, status is scattered:
```python
# ❌ WRONG — status inferred from four separate flags
if state.get("approval_decision") == "approved" and not state.get("error"):
    # are we resolved? approved but not yet responded? impossible to know
```

With `TicketStatus`:
```python
# ✅ RIGHT — status is explicit, one field, one place
if state.get("ticket_status") == TicketStatus.RESOLVED:
    # unambiguous — no combination of flags needed
```

**Why `str`-Enum?** Because LangGraph's SQLite checkpointer serialises state to JSON. If `TicketStatus` were a standard Enum, SQLite would store `"TicketStatus.RESOLVED"` (the repr), which is not a clean string and requires deserialisation logic. `class TicketStatus(str, Enum)` makes values serialise as plain `"resolved"` — checkpoint-safe by design.

### 📍 STATE MACHINE TRANSITIONS

```
                    ┌─────────────────────────────────────────┐
                    │         TicketStatus State Machine        │
                    └─────────────────────────────────────────┘

          PENDING ──────────────────────► IN_PROGRESS
                                              │
                       ┌──────────────────────┼──────────────────┐
                       │                      │                  │
                       ▼                      ▼                  ▼
               AWAITING_APPROVAL         ESCALATED            RESOLVED
                  │        │                 │
            ┌────┘        └────┐             │
            ▼                  ▼             ▼
         APPROVED           REJECTED     IN_PROGRESS
            │                  │        (higher level)
            └────────┬──────────┘
                     ▼
                 RESOLVED
```

### 🏗️ PRODUCTION IMPLEMENTATION

**`state.py` — `TicketStatus` str-Enum:**

```python
# state.py (Module 7 addition)

from enum import Enum

class TicketStatus(str, Enum):
    """
    Lifecycle state of a support ticket — Concept 65: Workflow State Machines.

    str-Enum: values serialise as plain strings — checkpoint-safe.
    SQLite stores "pending", not "TicketStatus.PENDING".

    Transitions:
      PENDING → IN_PROGRESS          (ticket assigned to agent)
      IN_PROGRESS → AWAITING_APPROVAL (large refund detected)
      IN_PROGRESS → ESCALATED         (threshold/error triggered)
      AWAITING_APPROVAL → APPROVED | REJECTED | ESCALATED (human decides)
      APPROVED | REJECTED → RESOLVED  (response sent)
      ESCALATED → IN_PROGRESS | RESOLVED
    """
    PENDING           = "pending"
    IN_PROGRESS       = "in_progress"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED          = "approved"
    REJECTED          = "rejected"
    ESCALATED         = "escalated"
    RESOLVED          = "resolved"
```

**Usage in workflow nodes:**

```python
# In billing_node — transition to AWAITING_APPROVAL when large refund detected
if check_approval_required(refund_amount):
    return {
        "active_agent": "billing",
        "pending_approval": dict(payload),
        # Status can be tracked separately via audit events or a ticket_status field
    }

# In approval_check_node — status is reflected in the audit event's decision field
return {
    "pending_approval": None,
    "audit_events": [{
        "action": "refund_decision",
        "decision": approval_decision,   # "approved" | "rejected" | "escalated"
    }],
}
```

### ✅ CONCEPT CHECKPOINT

1. Why does `TicketStatus` inherit from `str` as well as `Enum`?
2. What is the legal transition from `AWAITING_APPROVAL` after the operator enters "escalated"?
3. Can a ticket go from `RESOLVED` back to `IN_PROGRESS`? What does the state machine tell you about that?

---

## 📌 CONCEPT 62: Escalation Flows  [Module 7]
**Layer:** Graph | **Complexity:** Intermediate

### 🔍 WHAT

Escalation is the process of moving a ticket to a higher support tier when the current tier cannot resolve it. LangGraph enables escalation via the `should_escalate` router function and the `escalate_node`, which increments `escalation_level` in state.

Escalation can be triggered two ways:
1. **Automatic** — `should_escalate()` router detects `escalation_level >= L2` or an unrecoverable error
2. **Human-triggered** — operator types "escalated" during a HITL review → `approval_decision` is "escalated" → `should_escalate()` returns `"escalate_node"`

### 📍 ESCALATION LEVELS

```
EscalationLevel.L1 = 0  — Frontline agent (default, all tickets start here)
EscalationLevel.L2 = 1  — Senior agent / specialist
EscalationLevel.L3 = 2  — Engineering / management (MAX — never go higher)
```

`EscalationLevel` is a class of integer constants rather than an Enum because:
- `escalation_level` in state is a plain `int` (JSON-safe, no Enum overhead)
- `escalation_level >= EscalationLevel.L3` reads naturally
- Adding levels later does not break existing comparisons

### 🏗️ PRODUCTION IMPLEMENTATION

**`workflows/escalation.py` — `should_escalate` router and `escalate_node`:**

```python
# workflows/escalation.py

class EscalationLevel:
    L1: int = 0   # Frontline — default
    L2: int = 1   # Senior agent / specialist
    L3: int = 2   # Engineering / management
    MAX: int = L3  # Cap — never exceed

def should_escalate(state: AgentState) -> str:
    """
    Pure synchronous router. Returns "escalate_node" or "respond".
    Triggers on: HITL escalation decision, unrecoverable error, or level >= L2.
    """
    if state.get("approval_decision") == "escalated":
        return "escalate_node"
    if state.get("error") and state.get("escalation_level", 0) < EscalationLevel.MAX:
        return "escalate_node"
    if state.get("escalation_level", 0) >= EscalationLevel.L2:
        return "escalate_node"
    return "respond"


async def escalate_node(state: AgentState, config: RunnableConfig) -> dict:
    """
    Increments escalation_level (capped at MAX) and writes audit event.
    Reached via automatic router or Command(goto="escalate_node") from HITL.
    """
    thread_id = _get_thread_id(config)
    current_level = state.get("escalation_level", 0)
    new_level = min(current_level + 1, EscalationLevel.MAX)

    # Determine reason for audit trail
    if state.get("approval_decision") == "escalated":
        reason = "operator escalated via HITL"
    elif state.get("error"):
        reason = f"unrecoverable error: {state.get('error', 'unknown')}"
    else:
        reason = "automatic escalation threshold reached"

    logger.warning(
        "escalate_node: thread=%s level=%d→%d reason=%s",
        thread_id, current_level, new_level, reason,
    )

    return {
        "escalation_level": new_level,
        "audit_events": [{
            "node": "escalate_node",
            "action": "ticket_escalated",
            "thread_id": thread_id,
            "previous_level": current_level,
            "new_level": new_level,
            "reason": reason,
            "ticket_type": state.get("ticket_type"),
        }],
    }
```

### ✅ CONCEPT CHECKPOINT

1. Why is `should_escalate` a synchronous function rather than async?
2. Why is `EscalationLevel.MAX` set to `L3` and what happens if `escalate_node` runs when `escalation_level` is already at `L3`?
3. What are the two ways an operator can trigger escalation through the HITL flow?

---

## 📌 CONCEPT 69: SLA Workflows  [Module 7]
**Layer:** Service | **Complexity:** Beginner

### 🔍 WHAT

A Service Level Agreement (SLA) defines the expected response and resolution time for a support ticket based on its priority. Without SLA tracking, every ticket is treated equally — a critical outage ticket and a low-priority cosmetic bug wait in the same queue.

Our SLA implementation is entirely **pure functions** (`compute_sla_deadline`, `is_sla_breached`, `hours_until_deadline`) that do only datetime arithmetic — no LLM calls, no I/O. The `sla_tracking_node` calls these and writes the result to `audit_events`.

### 📍 SLA TIERS

```python
_SLA_HOURS: dict[str, int] = {
    "critical": 1,    # 1 hour  — system outages, data loss
    "high":     4,    # 4 hours — major feature broken
    "medium":   24,   # 24 hours — normal issues (default)
    "low":      72,   # 72 hours — cosmetic, documentation
}
```

Unknown priorities default to `"medium"` — fail-safe over fail-open.

### 🏗️ PRODUCTION IMPLEMENTATION

**`workflows/sla.py` — pure SLA functions and `sla_tracking_node`:**

```python
# workflows/sla.py

def compute_sla_deadline(priority: str, created_at: datetime | None = None) -> str:
    """
    Returns ISO 8601 SLA deadline string.
    Pure function — deterministic, no I/O.
    Unknown priority defaults to "medium" (24 hours).
    """
    hours = _SLA_HOURS.get(priority, _SLA_HOURS["medium"])
    base = created_at or datetime.now(tz=timezone.utc)
    if base.tzinfo is None:
        base = base.replace(tzinfo=timezone.utc)
    return (base + timedelta(hours=hours)).isoformat()


def is_sla_breached(deadline_str: str) -> bool:
    """Returns True if current UTC time is past the deadline. Pure function."""
    deadline = datetime.fromisoformat(deadline_str)
    if deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=timezone.utc)
    return datetime.now(tz=timezone.utc) > deadline


def hours_until_deadline(deadline_str: str) -> float:
    """
    Returns hours remaining. Negative value = already breached.
    Pure function — safe for sorting and prioritisation logic.
    """
    deadline = datetime.fromisoformat(deadline_str)
    if deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=timezone.utc)
    delta = deadline - datetime.now(tz=timezone.utc)
    return delta.total_seconds() / 3600


async def sla_tracking_node(state: AgentState, config: RunnableConfig) -> dict:
    """
    Computes SLA deadline and writes to audit_events. No LLM calls.

    Reads state["ticket_metadata"] for priority and created_at.
    Falls back to "medium" priority and now if ticket_metadata is None.

    Writes: state["audit_events"] — one SLA event with deadline and breach status.
    """
    thread_id = _get_thread_id(config)
    metadata = state.get("ticket_metadata")

    priority = "medium"
    created_at: datetime | None = None
    if metadata:
        priority = metadata.get("priority", "medium")
        raw = metadata.get("created_at")
        if raw:
            try:
                created_at = datetime.fromisoformat(raw)
            except ValueError:
                logger.warning("sla_tracking_node: invalid created_at=%r, using now", raw)

    deadline_str = compute_sla_deadline(priority, created_at)
    breached = is_sla_breached(deadline_str)
    remaining = hours_until_deadline(deadline_str)

    if breached:
        logger.warning(
            "sla_tracking_node: SLA BREACHED thread=%s priority=%s",
            thread_id, priority,
        )

    return {"audit_events": [{
        "node": "sla_tracking_node",
        "action": "sla_breach_detected" if breached else "sla_computed",
        "thread_id": thread_id,
        "priority": priority,
        "deadline": deadline_str,
        "breached": breached,
        "hours_remaining": round(remaining, 2),
    }]}
```

### 🧪 TESTS

```python
# tests/unit/test_sla.py

from datetime import datetime, timezone, timedelta
from workflows.sla import compute_sla_deadline, is_sla_breached, hours_until_deadline

def test_critical_sla_is_one_hour():
    now = datetime.now(tz=timezone.utc)
    deadline = datetime.fromisoformat(compute_sla_deadline("critical", now))
    delta = deadline - now
    assert 3599 < delta.total_seconds() < 3601   # ~1 hour

def test_low_sla_is_72_hours():
    now = datetime.now(tz=timezone.utc)
    deadline = datetime.fromisoformat(compute_sla_deadline("low", now))
    delta = deadline - now
    assert abs(delta.total_seconds() - 72 * 3600) < 2

def test_unknown_priority_defaults_to_medium():
    now = datetime.now(tz=timezone.utc)
    deadline = datetime.fromisoformat(compute_sla_deadline("unknown_priority", now))
    delta = deadline - now
    assert abs(delta.total_seconds() - 24 * 3600) < 2

def test_is_sla_breached_past_deadline():
    past = (datetime.now(tz=timezone.utc) - timedelta(hours=1)).isoformat()
    assert is_sla_breached(past) is True

def test_is_sla_breached_future_deadline():
    future = (datetime.now(tz=timezone.utc) + timedelta(hours=1)).isoformat()
    assert is_sla_breached(future) is False

def test_hours_until_deadline_negative_when_breached():
    past = (datetime.now(tz=timezone.utc) - timedelta(hours=2)).isoformat()
    assert hours_until_deadline(past) < 0
```

### ✅ CONCEPT CHECKPOINT

1. Why are the SLA functions pure functions rather than methods on a class?
2. If `ticket_metadata` is `None` (common in tests), what priority and deadline does `sla_tracking_node` compute?
3. How would you wire `sla_tracking_node` into the graph topology?

---

## 📌 CONCEPT 70: Audit Trails  [Module 7]
**Layer:** Service (observability) | **Complexity:** Beginner

### 🔍 WHAT

An audit trail is an **append-only, tamper-evident log** of every significant action that happened during a session. In our app, `state["audit_events"]` uses `operator.add` as its reducer — it can only accumulate entries, never overwrite or delete them.

**Mental model:** A paper ledger where you can only write forward. You can add new lines; you can never erase or change a past line.

**Why it matters:** Compliance, debugging, and customer disputes all require knowing exactly what happened and in what order. If a customer calls back asking "why was my refund rejected?", the audit trail has the answer — permanently, in the checkpoint.

### 🤔 WHY

Without `operator.add` on `audit_events`:

```python
# ❌ WRONG — each node replaces the entire audit_events list
# After approval_check_node, only the approval event remains
# billing_node's ticket_received event is gone
audit_events: list[dict]   # plain list — gets replaced, not appended
```

With `operator.add`:

```python
# ✅ RIGHT — every node's events accumulate
# audit_events after the full run:
# [billing_node:ticket_received, approval_check:refund_decision, memory_write:memory_written]
audit_events: Annotated[list[dict], operator.add]  # in state.py
```

The checkpointer persists `audit_events` after every node. Even if the process crashes mid-run, the events written so far survive.

### 🏗️ PRODUCTION IMPLEMENTATION

**`observability/audit.py` — `AuditEvent` TypedDict, factory, and formatter:**

```python
# observability/audit.py

class AuditEvent(TypedDict, total=False):
    """
    A single audit trail entry.

    total=False: optional fields present only when relevant.
    Required fields always set by create_audit_event(): node, action, thread_id, timestamp.
    """
    node: str        # Which node created this event
    action: str      # What happened: "ticket_received", "refund_approved", etc.
    thread_id: str   # Session identifier
    timestamp: str   # ISO 8601 UTC (always set by factory)
    # Optional:
    amount: float    # Financial events
    decision: str    # Approval events
    pii_types: list  # Compliance events
    level: int       # Escalation events
    user_id: str     # User-attributed events
    details: dict    # Event-specific catch-all


def create_audit_event(
    node: str,
    action: str,
    thread_id: str,
    **extra: Any,
) -> AuditEvent:
    """
    Factory: builds a typed AuditEvent with UTC timestamp.
    Pure function. All required fields always present.

    Usage:
        event = create_audit_event("billing_node", "refund_approved", "t-001",
                                   amount=750.0, decision="approved")
    """
    return AuditEvent(
        node=node,
        action=action,
        thread_id=thread_id,
        timestamp=datetime.now(tz=timezone.utc).isoformat(),
        **extra,  # type: ignore[misc]
    )


def format_audit_trail(events: list[dict]) -> str:
    """
    Formats audit_events into a human-readable multi-line string.
    Pure function — no I/O.

    Example output:
        [billing_node] ticket_received
        [approval_check_node] refund_decision — amount=750.0 decision=approved
    """
    lines = []
    for event in events:
        node = event.get("node", "unknown")
        action = event.get("action", "unknown")
        extras = {k: v for k, v in event.items()
                  if k not in {"node", "action", "thread_id", "timestamp"}}
        extra_str = " ".join(f"{k}={v}" for k, v in extras.items())
        line = f"[{node}] {action}"
        if extra_str:
            line += f" — {extra_str}"
        lines.append(line)
    return "\n".join(lines)
```

**`audit_trail_node` — writes a session summary event at graph END:**

```python
async def audit_trail_node(state: AgentState, config: RunnableConfig) -> dict:
    """
    Summary audit node. Placed at END of graph to record session completion.
    Reads: state["audit_events"], state["final_response"], state["escalation_level"]
    Writes: state["audit_events"] — appends one session_completed event.
    """
    thread_id = _get_thread_id(config)
    existing = state.get("audit_events", [])

    summary = create_audit_event(
        node="audit_trail_node",
        action="session_completed",
        thread_id=thread_id,
        total_events=len(existing),
        response_generated=state.get("final_response") is not None,
        escalation_level=state.get("escalation_level", 0),
        ticket_type=state.get("ticket_type"),
    )

    return {"audit_events": [summary]}
```

### ✅ CONCEPT CHECKPOINT

1. What LangGraph mechanism makes `audit_events` append-only within a session?
2. Why does `AuditEvent` use `total=False` in its TypedDict definition?
3. If the process crashes after `billing_node` but before `approval_check_node`, what audit events survive in the checkpoint?

---

## 📌 CONCEPT 71: Compliance Pipelines  [Module 7]
**Layer:** Service (security) | **Complexity:** Beginner

### 🔍 WHAT

A compliance pipeline is a node (or sequence of nodes) that enforces regulatory or policy requirements before processing continues. In our app, `compliance_gate_node` detects PII (Personally Identifiable Information) in every incoming ticket and records the finding in `audit_events`.

**Mental model:** A metal detector at an airport. Everyone goes through it — not because most people carry prohibited items, but because the rule must apply to everyone and every check must be logged.

**Design decision: detect-and-flag, not detect-and-block.** Blocking on PII detection would also block legitimate tickets where a customer provides their email to get a callback. Instead, the compliance node flags PII in `audit_events` so compliance officers can review it — the LLM still processes the ticket but won't unnecessarily echo PII.

### 📍 PII PATTERNS DETECTED

| PII Type | Pattern example | Regex |
|----------|----------------|-------|
| `email` | `user@example.com` | `\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b` |
| `phone` | `555-123-4567` | `\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b` |
| `ssn` | `123-45-6789` | `\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b` |
| `credit_card` | `4111 1111 1111 1111` | `\b(?:\d{4}[-\s]?){3}\d{4}\b` |
| `ip_address` | `192.168.1.1` | `\b(?:\d{1,3}\.){3}\d{1,3}\b` |

Patterns are **intentionally broad** — better to flag a false positive than miss real PII. Fail-safe over fail-open.

### 🏗️ PRODUCTION IMPLEMENTATION

**`security/compliance.py` — `detect_pii` and `compliance_gate_node`:**

```python
# security/compliance.py

_PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email":       re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "phone":       re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"),
    "ssn":         re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    "ip_address":  re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}


def detect_pii(text: str) -> list[str]:
    """
    Returns list of PII type names found in text. Pure function.
    Empty list if no PII detected.

    Examples:
        detect_pii("Contact me at john@example.com")  → ["email"]
        detect_pii("Call 555-123-4567 or email me")   → ["phone"]
    """
    return [pii_type for pii_type, pattern in _PII_PATTERNS.items()
            if pattern.search(text)]


def _redact_pii(text: str, pii_types: list[str]) -> str:
    """
    Replaces PII matches with [REDACTED_<TYPE>] placeholders.
    Pure function — used for safe logging. Does NOT modify the ticket in state.
    """
    redacted = text
    for pii_type in pii_types:
        if pii_type in _PII_PATTERNS:
            redacted = _PII_PATTERNS[pii_type].sub(
                f"[REDACTED_{pii_type.upper()}]", redacted
            )
    return redacted


async def compliance_gate_node(state: AgentState, config: RunnableConfig) -> dict:
    """
    Compliance gate: detects PII in the last message, writes to audit_events.

    Runs BEFORE specialist handler nodes — PII is flagged before any LLM sees it.
    Design: detect-and-flag, not detect-and-block (legal tickets may contain
    contact info intentionally).

    Always writes an audit event — even clean tickets get a "pii_scan_complete"
    event, creating a complete trail showing every ticket was checked.

    Reads:  state["messages"]    — last message scanned
    Writes: state["audit_events"] — one pii_scan_complete event
    """
    messages = state.get("messages", [])
    thread_id = _get_thread_id(config)

    if not messages:
        return {}

    ticket_text = messages[-1].content
    pii_types = detect_pii(ticket_text)

    if pii_types:
        # Redact before logging — NEVER log raw PII
        redacted = _redact_pii(ticket_text, pii_types)
        logger.warning(
            "compliance_gate_node: PII detected thread=%s types=%s text=%r",
            thread_id, pii_types, redacted[:100],
        )

    return {"audit_events": [{
        "node": "compliance_gate_node",
        "action": "pii_scan_complete",
        "thread_id": thread_id,
        "pii_detected": bool(pii_types),
        "pii_types": pii_types,
    }]}
```

### 🧪 TESTS

```python
# tests/unit/test_compliance.py

def test_detect_pii_email():
    assert detect_pii("Contact me at user@example.com") == ["email"]

def test_detect_pii_phone():
    assert "phone" in detect_pii("Call me at 555-123-4567")

def test_detect_pii_none():
    assert detect_pii("My subscription is too expensive") == []

def test_detect_pii_multiple():
    found = detect_pii("Email user@test.com or call 555-000-1234")
    assert "email" in found
    assert "phone" in found

def test_redact_pii_email():
    result = _redact_pii("Email user@example.com please", ["email"])
    assert "user@example.com" not in result
    assert "[REDACTED_EMAIL]" in result

@pytest.mark.asyncio
async def test_compliance_gate_node_writes_audit_event(base_state, runnable_config):
    """Node must always write an audit event — even for clean tickets."""
    result = await compliance_gate_node(base_state, runnable_config)
    assert "audit_events" in result
    assert len(result["audit_events"]) == 1
    assert result["audit_events"][0]["action"] == "pii_scan_complete"

@pytest.mark.asyncio
async def test_compliance_gate_flags_pii(base_state, runnable_config):
    """Node must detect and flag email PII."""
    base_state["messages"] = [HumanMessage(content="Email me at john@example.com")]
    result = await compliance_gate_node(base_state, runnable_config)
    event = result["audit_events"][0]
    assert event["pii_detected"] is True
    assert "email" in event["pii_types"]
```

### ✅ CONCEPT CHECKPOINT

1. Why does `compliance_gate_node` write an audit event even when no PII is detected?
2. Why does `_redact_pii` not modify the ticket content stored in `state["messages"]`?
3. Where in the graph topology should `compliance_gate_node` run — before or after `classify_ticket_node`?

---

## Module 7: Graph Topology Change

**Before Module 7** (Modules 1–6):
```
[START] → [memory_retrieval] → [classify_ticket]
                                       │
             ┌────────────────────────┬┴──────────────────────┐
             ▼                        ▼                        ▼
      [billing_node]         [technical_node]           [general_node]
             │                        │                        │
             └────────────────────────┼────────────────────────┘
                                      ▼
                                  [respond] → [memory_write] → END
```

**After Module 7** (billing now has HITL gate):
```
[START] → [memory_retrieval] → [classify_ticket]
                                       │
             ┌────────────────────────┬┴──────────────────────┐
             ▼                        ▼                        ▼
      [billing_node]         [technical_node]           [general_node]
             │
       route_after_billing()
             │
     pending? ─YES──► [approval_check] ◄── interrupt_before
     no ─────────────────┘      │
                                │ (runs after resume)
                                ▼
                           [respond] → [memory_write] → END
```

**Compile change:**
```python
# Before Module 7
graph = build_graph(checkpointer=get_checkpointer())

# After Module 7 — interrupt_before added as default in build_graph()
graph = build_graph(checkpointer=get_checkpointer())
# (interrupt_before=["approval_check"] is the default — no extra kwarg needed)
```

---

## Module 7: Key Lessons

| Lesson | Detail |
|--------|--------|
| `interrupt()` vs `interrupt_before` | `interrupt()` requires Python 3.11+ due to `asyncio` context var propagation. Use `interrupt_before` for Python 3.10 compatibility. |
| Resume pattern | `update_state(config, delta)` + `ainvoke(None, config)`. NOT `ainvoke(state_dict, config)` — that restarts from the entry node. |
| Payload via state | With `interrupt_before`, the payload goes in state (e.g. `pending_approval`). With `interrupt()`, the payload goes in `task.interrupts`. |
| Pure functions first | All detection/computation logic (`_extract_refund_amount`, `detect_pii`, `compute_sla_deadline`, `check_approval_required`) lives in pure functions. Nodes just call them. |
| Audit trail = `operator.add` | `audit_events: Annotated[list[dict], operator.add]` ensures the trail accumulates and is never overwritten. |
| `str`-Enum for state | `class TicketStatus(str, Enum)` serialises to plain strings — checkpoint-safe across SQLite and JSON. |
| Test count | Module 7 added 23 HITL tests + 54 workflow tests = 77 new tests. All 387 tests pass. |

---

## 📌 CONCEPT 73: Subgraphs  [Module 8]
**Layer:** Graph | **Complexity:** Advanced

> **Implementation status (read first):** the one real subgraph in this repo is the CRAG graph built by
> `get_rag_graph()` in `rag/graph.py`. It is compiled and unit-tested, but **nothing in the running app calls it**
> (`grep get_rag_graph` hits only its own module, `rag/__init__.py`, and a stub comment in
> `agents/technical_agent.py`). The multi-agent fan-out in `agents/supervisor.py` is **not** subgraphs:
> `kb_worker`, `web_worker`, and `research_collect` are plain nodes registered on the parent graph in `graph.py`,
> dispatched with `Send`. `kb_worker_node` calls `make_retriever_node()` (one node function), not the CRAG graph.

### 🔍 WHAT

A **subgraph** is a compiled `StateGraph` used as a unit — either added as a node in a parent graph, or invoked
from inside a node. The parent sees one box; the internal topology is encapsulated.

**Mental model:** a function call whose body is itself a flowchart.

### 🤔 WHY

CRAG is five nodes and a loop: `retrieve → grade_retrieval → (generate | rewrite_query → retrieve) → hallucination_check`.
Inlining that into the main graph would bury the ticket-routing topology under retrieval detail, and make the pipeline
impossible to reuse from a second call site.

### 📍 WHEN TO USE

- ✅ A multi-node pipeline with its own loop/branching that more than one caller needs
- ✅ You want a typed boundary: caller supplies `query`, receives `retrieved_docs` / `final_response`
- ❌ A linear 2–3 node sequence — plain nodes are simpler
- ❌ Parallel workers that must each appear in the parent's streaming/`updates` output — plain nodes (as here) keep them visible

### 🌍 WHERE IT FITS

```
parent graph (graph.py)                         CRAG subgraph (rag/graph.py)
 supervisor_node ─Send─► kb_worker ─┐            retrieve ─► grade_retrieval ─┬─► generate ─► hallucination_check
                         web_worker ─┼► research_collect        ▲             └─► rewrite_query ─┐
                                     │                          └──────────────────────────────────┘
          (kb_worker calls make_retriever_node(); get_rag_graph() is NOT in this path today)
```

Both use `AgentState` as the schema, so there is no state-mapping layer between them (shared-schema subgraph).

### 🔑 What the code does

```python
# rag/graph.py
@lru_cache(maxsize=1)
def get_rag_graph():
    ...
    graph = (
        GraphBuilder(AgentState)
        .node("retrieve", retriever)
        .node("grade_retrieval", retrieval_grade_node)
        .node("rewrite_query", rewrite_query_node)
        .node("generate", generate_node)
        .node("hallucination_check", hallucination_check_node)
        .edge("retrieve", "grade_retrieval")
        .conditional_edge("grade_retrieval", route_by_grade,
                          {"generate": "generate", "rewrite_query": "rewrite_query"})
        .edge("rewrite_query", "retrieve")
        .edge("generate", "hallucination_check")
        .entry("retrieve")
        .build()          # no checkpointer= — see below
    )
```

- **No `checkpointer=`**: a subgraph used inside a parent inherits the parent's checkpointer; giving it its own is
  not needed here and is not done.
- **`lru_cache`**: compile once per process.
- `route_by_grade` treats `relevant` and `partial` as `generate`, anything else as `rewrite_query`.

### ⚖️ PROS & CONS

| Pros | Cons |
|------|------|
| Encapsulated, reusable pipeline | One more graph to trace/debug |
| Parent topology stays readable | Shared `AgentState` couples subgraph and parent schemas |
| Compile-once via `lru_cache` | Not integrated yet — currently dead code from the app's perspective |

### 💡 BEST PRACTICES

1. Decide the boundary contract first (inputs/outputs), then build the graph.
2. Don't give a subgraph its own checkpointer when the parent already has one.
3. Guard loops: CRAG's `rewrite_query → retrieve` cycle relies on `settings.graph_recursion_limit` (default 50) as the backstop.
4. Be honest in docs/tests about what is wired versus merely built.

### ✅ CONCEPT CHECKPOINT

1. Why is `get_rag_graph()` cached but `get_checkpointer()` is not?
2. What would you change in `kb_worker_node` to use the CRAG subgraph instead of the bare retriever node?
3. Why are `kb_worker`/`web_worker` plain nodes rather than subgraphs?

**Exercise:** write an integration test that invokes `get_rag_graph()` with a mocked retriever and asserts `retrieved_docs` is populated and the `rewrite_query` loop exits for an irrelevant-then-relevant grade sequence.

---

## 📌 CONCEPT 74: Supervisor Pattern  [Module 8]
**Layer:** Graph | **Complexity:** Intermediate

### 🔍 WHAT

The Supervisor Pattern is a multi-agent orchestration approach where a single **supervisor node** receives every routed ticket, looks up the correct specialist agent from a central registry, and delegates processing to it. The supervisor is a thin dispatcher — it does not know the agents' internals, only their registered names.

**Mental model:** An air traffic control tower. The tower doesn't fly the planes; it knows which planes are registered, which runway each one belongs on, and it directs them accordingly. Adding a new airline never requires rebuilding the tower.

**Where it lives:** `agents/supervisor.py` (graph layer) calls `agents/registry.py` (service layer) which maps names to agent factories.

### 🤔 WHY

Without a supervisor, the routing logic grows with every new agent:

```python
# ❌ WRONG — hard-coded agent dispatch in graph.py
if ticket_type == "billing":
    result = await billing_agent.process(state, config)
elif ticket_type == "technical":
    result = await technical_agent.process(state, config)
elif ticket_type == "research":
    result = await research_agent.process(state, config)
# Adding a "legal" agent means editing this file
```

With the Supervisor Pattern:

```python
# ✅ RIGHT — supervisor delegates via registry, no changes needed for new agents
agent = get_agent(ticket_type)          # KeyError if unregistered
result = await agent.process(state, config)
```

**Why a registry?** The registry is defined once in `agents/registry.py`. Each agent registers itself via a `@register_agent("name")` decorator on its factory function. The supervisor never imports the agents directly — it only imports `get_agent`. This satisfies the Open/Closed Principle: new agents are added by adding new files, not by modifying existing ones.

### 📍 WHEN TO USE

- You have 3+ specialist agents that handle different sub-problems
- You want to add new agents without touching existing routing code
- You need a single point for fallback and error handling (unknown ticket types)
- You need future extensibility to swap out any individual agent

Do NOT use the supervisor pattern when:
- You have only two agents — a simple conditional edge is cleaner
- All agents run in parallel always — use the Send API directly instead

### 🌍 WHERE IT FITS

```
[classify_ticket]
       │
       │ ROUTE_KEYS: billing→billing_node, others→supervisor_node
       │
       ├──► [billing_node] ──► [approval_check] ──► [respond]
       │
       └──► [supervisor_node]                      ← Concept 74
                   │
             registry.get_agent(ticket_type)
                   │
         ┌─────────┼────────────┐
         ▼         ▼            ▼
  TechnicalAgent  GeneralAgent  ResearchAgent
         │                       │
         │               route_after_supervisor()
         │                       │
         ▼               [Send("kb_worker"), Send("web_worker")]
       [respond]         [research_collect] → [respond]
```

The ROUTE_KEYS change from Module 7 is significant: `technical`, `general`, and `research` all route to `supervisor_node` now. The supervisor dispatches from there.

### ⚖️ PROS & CONS

| Pros | Cons |
|------|------|
| Adding a new agent requires no changes to supervisor or graph | Registry is a global dict — test teardown must clear it manually |
| Fallback handling in one place (KeyError → "general") | Agent factories are called fresh on every dispatch — no singleton agents |
| Clean separation: router knows names, supervisor knows protocol | All non-billing tickets touch the supervisor — adds one hop |
| OCP: open for extension (new agents), closed for modification | Registry errors surface at runtime, not at compile time |

### 🆚 DIFFERENCES & COMPARISONS

| | Supervisor Pattern (this app) | `create_supervisor` (LangGraph library) |
|--|-------------------------------|----------------------------------------|
| Agent discovery | Manual `@register_agent` decorator | LLM-driven — supervisor LLM picks the agent |
| Routing decision | Deterministic (ticket_type string) | Non-deterministic (LLM choice) |
| Cost | Zero LLM tokens for routing | Costs tokens on every dispatch |
| Predictability | 100% — same input always same agent | Varies with model response |
| When to use | Known, bounded ticket types | Open-ended, emergent tasks |

For our support agent, deterministic routing wins: we know the ticket types at design time, and we want HITL-auditable, reproducible routing.

### 💡 BEST PRACTICES

1. **Always implement a fallback.** If `get_agent(ticket_type)` raises `KeyError`, catch it and fall back to the "general" agent. Never let an unregistered type crash the graph.
2. **Keep the supervisor thin.** It should contain zero business logic — only `get_agent(type)` + `agent.process(state, config)`.
3. **Register agents at module import time.** The `@register_agent` decorator runs when the module is first imported. Ensure all agent modules are imported before the graph runs (see `agents/__init__.py`'s `ensure_agents_registered()`).
4. **Call `get_agent(name)()` — note the two sets of parens.** The registry stores factories (callables that return agents), not agent instances. `_REGISTRY[name]()` instantiates a fresh agent. This avoids state leaking between requests.
5. **Clear the registry in tests.** `TestAgentRegistry.teardown_method()` must call `_REGISTRY.clear()` because the global dict persists across tests in the same process.

### 🏗️ PRODUCTION IMPLEMENTATION

**`agents/registry.py`:**

```python
# agents/registry.py
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from agents.base import BaseAgent

AgentFactory = Callable[[], "BaseAgent"]
_REGISTRY: dict[str, AgentFactory] = {}


def register_agent(name: str) -> Callable[[AgentFactory], AgentFactory]:
    """
    Decorator: registers an agent factory under a name.
    The supervisor resolves agents by name at dispatch time.

    Usage:
        @register_agent("billing")
        def _create_billing_agent() -> BillingAgent:
            return BillingAgent(llm=get_router_llm())
    """
    def decorator(factory: AgentFactory) -> AgentFactory:
        _REGISTRY[name] = factory
        return factory
    return decorator


def get_agent(name: str) -> "BaseAgent":
    """Resolves and instantiates a registered agent by name."""
    if name not in _REGISTRY:
        raise KeyError(f"No agent registered under '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name]()


def list_agents() -> list[str]:
    """Returns all registered agent names. Used by the supervisor."""
    return list(_REGISTRY.keys())
```

**`agents/supervisor.py` — supervisor_node (key section):**

```python
@with_retry
async def supervisor_node(state: AgentState, config: RunnableConfig) -> dict:
    """
    Dispatcher node: looks up the correct agent from the registry and delegates.

    Falls back to 'general' agent if ticket_type is unregistered.

    Reads:  state["ticket_type"]
    Writes: whatever the resolved agent's process() returns
    """
    ticket_type = state.get("ticket_type") or "general"
    thread_id = _get_thread_id(config)
    logger.info("supervisor_node: thread=%s ticket_type=%s", thread_id, ticket_type)
    try:
        agent = get_agent(ticket_type)
        result = await agent.process(state, config)
        return result
    except KeyError:
        logger.warning(
            "supervisor_node: no agent registered for ticket_type=%r — falling back to general",
            ticket_type,
        )
        fallback_agent = get_agent("general")
        return await fallback_agent.process(state, config)
```

**`agents/__init__.py` — ensuring registration before graph build:**

```python
def ensure_agents_registered() -> None:
    """
    Imports all specialist agent modules to trigger @register_agent decorators.
    Called in graph.py before compile() — guarantees all agents are in the
    registry even after test teardown clears it.
    """
    import agents.billing_agent   # noqa: F401
    import agents.technical_agent # noqa: F401
    import agents.general_agent   # noqa: F401
    import agents.research_agent  # noqa: F401
```

### 🧪 TESTS

```python
# tests/unit/test_multi_agent.py — TestAgentRegistry class

class TestAgentRegistry:
    def teardown_method(self):
        _REGISTRY.clear()   # must clear — global dict persists across tests

    def test_register_and_get_agent(self):
        @register_agent("test_agent")
        def factory():
            return MagicMock()
        agent = get_agent("test_agent")
        assert agent is not None

    def test_get_unregistered_agent_raises(self):
        with pytest.raises(KeyError, match="No agent registered"):
            get_agent("nonexistent")

    def test_list_agents_returns_registered(self):
        @register_agent("alpha")
        def f1(): return MagicMock()
        @register_agent("beta")
        def f2(): return MagicMock()
        assert "alpha" in list_agents()
        assert "beta" in list_agents()

    @pytest.mark.asyncio
    async def test_supervisor_node_falls_back_to_general(self, base_state, runnable_config):
        """Supervisor must fall back gracefully when ticket_type is unregistered."""
        base_state["ticket_type"] = "unknown_type"
        mock_general = AsyncMock()
        mock_general.process = AsyncMock(return_value={"active_agent": "general"})
        with patch("agents.supervisor.get_agent", side_effect=[KeyError("unknown_type"), mock_general]):
            result = await supervisor_node(base_state, runnable_config)
        assert result["active_agent"] == "general"
```

### ✅ CONCEPT CHECKPOINT

1. What is the difference between storing an agent instance vs a factory in `_REGISTRY`? Why does this app store factories?
2. Why does `agents/__init__.py` need `ensure_agents_registered()` if the decorators already run at import time?
3. If you add a new `LegalAgent`, which files do you need to change? Which files must NOT change?

---

## 📌 CONCEPT 75: Worker Agents  [Module 8]
**Layer:** Graph | **Complexity:** Intermediate

### 🔍 WHAT

A Worker Agent is a class that implements the `BaseAgent` abstract base class, providing a consistent interface for the supervisor to call. Every specialist agent in this app is a worker agent: it receives the full `AgentState`, does its specific work, and returns a partial state dict.

**Mental model:** Specialists in a hospital. The ER triage nurse (supervisor) routes patients; the cardiologist, neurologist, and radiologist (workers) each have a well-defined scope and a common protocol for receiving and returning patient information. The ER nurse doesn't know cardiology — it only knows how to route.

### 🤔 WHY

Without a base class, every agent would have an ad-hoc interface:

```python
# ❌ WRONG — each agent has a different signature
class BillingAgent:
    def handle(self, state, cfg, user_id): ...    # three params

class TechnicalAgent:
    async def process_ticket(self, s): ...         # different name, one param

class ResearchAgent:
    def run(self, query, ctx): ...                 # completely different
```

With `BaseAgent`:

```python
# ✅ RIGHT — supervisor can call any agent with the same two-line pattern
agent = get_agent(ticket_type)
result = await agent.process(state, config)
```

The abstract base also enforces that every agent declares its tools, which the graph uses at build time for tool-node wiring.

### 📍 WHEN TO USE

- You have multiple agents that all need to be called the same way by a supervisor
- You want to enforce that every agent exposes `process()` and `get_tools()`
- You want mypy to catch agents that forget to implement required methods at type-check time

### 🌍 WHERE IT FITS

```
agents/
├── base.py           ← BaseAgent ABC defined here
├── billing_agent.py  ← BillingAgent(BaseAgent)   @register_agent("billing")
├── technical_agent.py← TechnicalAgent(BaseAgent) @register_agent("technical")
├── general_agent.py  ← GeneralAgent(BaseAgent)   @register_agent("general")
├── research_agent.py ← ResearchAgent(BaseAgent)  @register_agent("research")
└── registry.py       ← get_agent() resolves name → factory → instance
```

### ⚖️ PROS & CONS

| Pros | Cons |
|------|------|
| Liskov Substitution: any agent is substitutable for any other at the call site | All agents must implement both `process()` and `get_tools()` even if trivially empty |
| mypy enforces the contract — missing methods are caught at type check time | Abstract classes require Python's `ABC` machinery — slightly more boilerplate |
| Supervisor code is O(1) — same two lines for 4 agents or 40 agents | Agents that need drastically different signatures require adapter wrappers |

### 🆚 DIFFERENCES & COMPARISONS

| | `BaseAgent` ABC (this app) | `Protocol` class |
|--|---------------------------|-----------------|
| Enforcement | `@abstractmethod` raises `TypeError` at instantiation | Structural — mypy checks at type time only |
| Runtime check | `isinstance(agent, BaseAgent)` works | `isinstance(agent, AgentProtocol)` requires `@runtime_checkable` |
| Registration | `@register_agent` decorator handles it | Must register manually |
| When to use | All agents inherit from a shared base | Agents come from third-party libs you don't control |

### 💡 BEST PRACTICES

1. **Keep `BaseAgent` lean.** It should declare only what the supervisor needs: `process()` and `get_tools()`. Don't bloat it with agent-specific helpers.
2. **`agent_name` as `@property`, not `__init__` param.** Hard-coding the name in the class makes it impossible to accidentally instantiate a `TechnicalAgent` with `name="billing"`.
3. **Return partial dicts from `process()`.** The base class contract says `process()` returns `dict`. Enforce the partial-dict pattern in tests, not in the base.
4. **Always append to `audit_events` in `process()`.** Every agent should record that it processed the ticket. This creates an unbroken audit trail through the supervisor dispatch chain.

### 🏗️ PRODUCTION IMPLEMENTATION

**`agents/base.py`:**

```python
"""
Module: agents.base
Layer:  Graph
Owner:  agents/

Purpose:
    Abstract base class for all specialist worker agents.
    Enforces a consistent interface (process / get_tools) for the supervisor.

Public API:
    BaseAgent — ABC every specialist agent inherits from
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from langchain_core.runnables import RunnableConfig

if TYPE_CHECKING:
    from state import AgentState

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base for all worker agents.

    Subclasses must implement:
        agent_name  — property returning the agent's registry key
        get_tools() — list of LangChain tools available to this agent
        process()   — async method that receives state and returns a partial dict
    """

    @property
    @abstractmethod
    def agent_name(self) -> str:
        """Registry key for this agent. Must match the @register_agent name."""
        ...

    @abstractmethod
    def get_tools(self) -> list:
        """Returns the LangChain tools available to this agent."""
        ...

    @abstractmethod
    async def process(self, state: AgentState, config: RunnableConfig) -> dict:
        """
        Process a ticket and return a partial state update dict.

        Args:
            state:  Current AgentState (read-only — never mutate).
            config: RunnableConfig carrying thread_id, tenant_id, etc.

        Returns:
            Partial state dict with only the fields this agent updates.
        """
        ...
```

### 🧪 TESTS

```python
# tests/unit/test_multi_agent.py — TestBaseAgentContract class

class TestBaseAgentContract:
    def test_cannot_instantiate_abstract_base(self):
        """BaseAgent cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseAgent()

    def test_concrete_agent_has_agent_name(self):
        agent = TechnicalAgent()
        assert agent.agent_name == "technical"

    def test_concrete_agent_get_tools_returns_list(self):
        agent = TechnicalAgent()
        assert isinstance(agent.get_tools(), list)

    def test_billing_agent_is_substitutable_for_base(self):
        """LSP: BillingAgent is a valid BaseAgent."""
        agent = BillingAgent(llm=AsyncMock(), tools=[])
        assert isinstance(agent, BaseAgent)
```

### ✅ CONCEPT CHECKPOINT

1. What would happen if `process()` were not declared `@abstractmethod`? Why is that dangerous?
2. Why is `agent_name` a `@property` rather than a constructor argument?
3. `TechnicalAgent.get_tools()` returns `[]` in Module 8. When should it return real tools?

---

## 📌 CONCEPT 76: Specialist Agents  [Module 8]
**Layer:** Graph | **Complexity:** Intermediate

### 🔍 WHAT

Specialist agents are concrete implementations of `BaseAgent` that handle a specific ticket category. In Module 8, four specialist agents are registered: `BillingAgent`, `TechnicalAgent`, `GeneralAgent`, and `ResearchAgent`. Each has a distinct role, a distinct tool list (fully populated in future modules), and registers itself with the global registry at import time.

**Mental model:** A law firm's practice groups. The intake coordinator (supervisor) routes clients; the M&A team, litigation team, and IP team (specialists) each have domain expertise and their own toolkits.

### 🤔 WHY

Without specialist agents, all logic lives in monolithic nodes:

```python
# ❌ WRONG — one mega-node handles all ticket types
async def handle_ticket_node(state, config):
    if state["ticket_type"] == "billing":
        # billing logic...
    elif state["ticket_type"] == "technical":
        # technical logic...
    elif state["ticket_type"] == "research":
        # research logic + RAG + web search...
    # This function grows unboundedly with every new feature
```

With specialist agents, each concern is isolated:

```python
# ✅ RIGHT — each specialist owns one vertical
# billing_agent.py   handles billing, wraps billing_node, owns billing tools
# technical_agent.py handles technical tickets, will own KB tool in Module 9
# research_agent.py  handles research tickets, triggers Send fan-out
```

### 📍 AGENT ROLES (Module 8)

| Agent | `agent_name` | LLM in Mod 8? | Tools in Mod 8 | Key state writes |
|-------|-------------|--------------|----------------|-----------------|
| `BillingAgent` | `"billing"` | Yes (via billing_node) | None yet | active_agent, delegates to billing_node |
| `TechnicalAgent` | `"technical"` | No | None yet | active_agent, agent_results, audit_events |
| `GeneralAgent` | `"general"` | No | None yet | active_agent, agent_results, audit_events |
| `ResearchAgent` | `"research"` | No | None yet | active_agent, query, agent_results, audit_events |

**Why don't TechnicalAgent and GeneralAgent call an LLM in Module 8?** The final response is always generated by `respond_node` at the end of the graph. Technical and general agents set `active_agent` so `respond_node` knows which persona to use. The full tool-calling loop comes in Module 9.

### 🌍 WHERE IT FITS

```
@register_agent("billing")
def _create_billing_agent() → BillingAgent:
    # returns BillingAgent(llm=get_router_llm(), tools=[])

@register_agent("technical")
def _create_technical_agent() → TechnicalAgent:
    # returns TechnicalAgent()

@register_agent("general")
def _create_general_agent() → GeneralAgent:
    # returns GeneralAgent()

@register_agent("research")
def _create_research_agent() → ResearchAgent:
    # returns ResearchAgent()
```

The `@register_agent` decorator runs when the module is first imported. `ensure_agents_registered()` in `agents/__init__.py` guarantees all four modules are imported before the graph compiles.

### ⚖️ PROS & CONS

| Pros | Cons |
|------|------|
| Each agent is independently testable | Four separate files to maintain |
| Adding `LegalAgent` never modifies existing agents (OCP) | Registration is side-effectful — test teardown must clear the registry |
| Each agent's `get_tools()` return is the explicit tool manifest | Module 8 stubs mean tests must mock LLM calls for BillingAgent but not others |

### 🆚 DIFFERENCES & COMPARISONS

| | This app's specialist agents | LangGraph `create_react_agent` per agent |
|--|------------------------------|------------------------------------------|
| Control | Full — you define every aspect of `process()` | Limited — React loop is fixed |
| Tool calling | Manual in `process()` | Automatic via tool-calling loop |
| Streaming | Handled in `respond_node` downstream | Built in to React agent |
| When to use | Custom logic, HITL, complex routing | Straightforward tool-use pattern |

### 💡 BEST PRACTICES

1. **BillingAgent wraps `billing_node`, not vice versa.** The node existed before the agent class. The agent's `process()` calls the node function directly, avoiding duplication.
2. **ResearchAgent sets `query` from the last message content.** It does not call an LLM to rewrite the query in Module 8 — that optimisation comes in Module 9 (query rewriting node).
3. **Every `process()` writes `audit_events`.** Even technical/general stubs write a `"ticket_received"` audit event, so the audit trail never has gaps.
4. **Keep `get_tools()` a manifest, not a factory.** Return a list of tool instances rather than lazy-building them inside `process()`. Tools are stateless — sharing instances is safe and avoids repeated instantiation.

### 🏗️ PRODUCTION IMPLEMENTATION

**`agents/technical_agent.py` (representative pattern — general_agent.py is identical except `agent_name`)**:

```python
class TechnicalAgent(BaseAgent):
    """
    Handles technical support tickets.
    Module 8: sets active_agent and writes audit event.
    Module 9: will use KB retrieval tool.
    """

    @property
    def agent_name(self) -> str:
        return "technical"

    def get_tools(self) -> list:
        return []  # KB tool added in Module 9

    async def process(self, state: AgentState, config: RunnableConfig) -> dict:
        thread_id = _get_thread_id(config)
        logger.info("TechnicalAgent.process: thread=%s", thread_id)
        return {
            "active_agent": self.agent_name,
            "agent_results": [{"agent": self.agent_name, "action": "ticket_received", "thread_id": thread_id}],
            "audit_events": [{"node": "technical_agent", "action": "ticket_received", "thread_id": thread_id}],
        }


@register_agent("technical")
def _create_technical_agent() -> TechnicalAgent:
    return TechnicalAgent()
```

**`agents/research_agent.py` (sets `query` for fan-out):**

```python
class ResearchAgent(BaseAgent):
    """
    Handles research tickets. Extracts query from last message.
    route_after_supervisor detects active_agent="research" and emits Send fan-out.
    """

    @property
    def agent_name(self) -> str:
        return "research"

    def get_tools(self) -> list:
        return []  # Tavily + KB tools added in Module 9

    async def process(self, state: AgentState, config: RunnableConfig) -> dict:
        thread_id = _get_thread_id(config)
        messages = state.get("messages", [])
        query = messages[-1].content if messages else ""
        logger.info("ResearchAgent.process: thread=%s query_len=%d", thread_id, len(query))
        return {
            "active_agent": self.agent_name,
            "query": query,
            "agent_results": [{"agent": self.agent_name, "action": "research_started", "thread_id": thread_id}],
            "audit_events": [{"node": "research_agent", "action": "research_started", "thread_id": thread_id}],
        }


@register_agent("research")
def _create_research_agent() -> ResearchAgent:
    return ResearchAgent()
```

### 🧪 TESTS

```python
# tests/unit/test_multi_agent.py — TestSpecialistAgents class

class TestSpecialistAgents:
    @pytest.mark.asyncio
    async def test_technical_agent_sets_active_agent(self, base_state, runnable_config):
        agent = TechnicalAgent()
        result = await agent.process(base_state, runnable_config)
        assert result["active_agent"] == "technical"

    @pytest.mark.asyncio
    async def test_technical_agent_writes_audit_event(self, base_state, runnable_config):
        agent = TechnicalAgent()
        result = await agent.process(base_state, runnable_config)
        assert len(result["audit_events"]) == 1
        assert result["audit_events"][0]["action"] == "ticket_received"

    @pytest.mark.asyncio
    async def test_research_agent_extracts_query(self, base_state, runnable_config):
        base_state["messages"] = [HumanMessage(content="How does JWT auth work?")]
        agent = ResearchAgent()
        result = await agent.process(base_state, runnable_config)
        assert result["query"] == "How does JWT auth work?"
        assert result["active_agent"] == "research"

    @pytest.mark.asyncio
    async def test_general_agent_returns_partial_dict(self, base_state, runnable_config):
        """Architecture: agent must return only the fields it changes."""
        agent = GeneralAgent()
        result = await agent.process(base_state, runnable_config)
        assert "messages" not in result
        assert "final_response" not in result
        assert "active_agent" in result
```

### ✅ CONCEPT CHECKPOINT

1. `TechnicalAgent.process()` returns `active_agent="technical"` but doesn't call an LLM. Where does the actual response text come from?
2. `ResearchAgent.process()` sets `query` from `messages[-1].content`. What happens in Module 9 when the query needs to be rewritten before retrieval?
3. Why does `BillingAgent.process()` delegate to `billing_node` rather than duplicate its logic?

---

## 📌 CONCEPT 84: Send API  [Module 8]
**Layer:** Graph | **Complexity:** Advanced

### 🔍 WHAT

The `Send` API is LangGraph's mechanism for dispatching state to a node with a **custom payload** rather than routing through the shared `AgentState`. A `Send("node_name", {"key": value})` object tells the graph runtime: "run this node right now, but give it this specific sub-state rather than the full current state."

When a router function returns a **list of `Send` objects**, LangGraph runs all of them **in parallel** — this is the native fan-out primitive.

**Mental model:** A print shop taking a big order. Instead of printing 500 pages one at a time on a single printer (sequential), the manager splits the job into five 100-page batches and sends each to a different printer simultaneously. The `Send` is the "here's your batch" instruction to each printer.

### 🤔 WHY

Before the Send API, parallel node execution required workarounds:

```python
# ❌ WRONG — sequential, no true parallel execution
async def research_node(state, config):
    kb_result = await _search_kb(state["query"])     # waits
    web_result = await _search_web(state["query"])   # waits for kb to finish
    return {"agent_results": [kb_result, web_result]}
```

With `Send`:

```python
# ✅ RIGHT — both workers run in parallel, each gets its own state slice
def route_after_supervisor(state: AgentState):
    if state.get("active_agent") == "research":
        query = state.get("query") or ""
        return [
            Send("kb_worker", {"query": query}),
            Send("web_worker", {"query": query}),
        ]
    return "respond"
```

The graph runtime executes `kb_worker` and `web_worker` concurrently. Their `agent_results` are merged back into the main state via the `operator.add` reducer.

### 📍 WHEN TO USE

Use `Send` when:
- Multiple workers need to process the **same input independently** (fan-out)
- Each worker needs a **different sub-state** (not the full `AgentState`)
- You want **true parallelism** within one graph turn, not asyncio concurrency within one node

Do NOT use `Send` when:
- Workers need to communicate with each other (they share no state until fan-in)
- You have only one downstream target — a simple edge is clearer
- Workers depend on each other's results (you need sequential edges there)

### 🌍 WHERE IT FITS

```
[supervisor_node]
       │
route_after_supervisor()
       │
       ├── active_agent == "research":
       │       returns [Send("kb_worker", {"query": q}),
       │                Send("web_worker", {"query": q})]
       │                       │          │
       │              (parallel)          (parallel)
       │           [kb_worker]        [web_worker]
       │                │                  │
       │                └────────┬──────────┘
       │                         ▼
       │                [research_collect]
       │
       └── active_agent != "research":
               returns "respond"
                         ▼
                     [respond]
```

**Graph topology change needed for Send:** `conditional_edge` mapping must be **optional** when using Send, because the router returns `list[Send]` rather than a string key. Module 8 updated `core/graph_builder.py`:

```python
def conditional_edge(
    self,
    src: str,
    router_fn: Callable,
    mapping: dict[str, str] | None = None,   # ← optional for Send API
) -> "GraphBuilder":
    if mapping is not None:
        self._graph.add_conditional_edges(src, router_fn, mapping)
    else:
        self._graph.add_conditional_edges(src, router_fn)
    return self
```

### ⚖️ PROS & CONS

| Pros | Cons |
|------|------|
| True parallel graph execution — not just async concurrency | Workers cannot communicate — each gets only its Send payload |
| Each worker gets a clean, minimal state slice (not the full state) | Groq free tier has rate limits — parallel LLM calls can exhaust tokens/min |
| Fan-in is automatic via `operator.add` reducers | Harder to debug — parallel branches need careful logging |
| No changes to the worker nodes — they don't know they're parallelised | `conditional_edge` mapping must be None when router returns list[Send] |

### 🆚 DIFFERENCES & COMPARISONS

| | `Send` (LangGraph) | `asyncio.gather` within a node |
|--|--------------------|-------------------------------|
| Where the work happens | Separate named nodes in the graph | Inside a single node function |
| Visibility in traces | Each worker is a named node — visible in LangSmith | Invisible — all inside the parent node |
| State isolation | Each worker gets its own state slice | Shared scope, potential contention |
| Fan-in | Automatic via reducers | Manual — must merge results in node |
| When to use | When workers should be separate, observable units | When sub-calls are internal helpers |

For production systems, prefer `Send` over `asyncio.gather` inside a node: each worker is named, traceable, and independently testable.

### 💡 BEST PRACTICES

1. **Pass only what the worker needs.** `Send("kb_worker", {"query": query})` — not the full `AgentState`. Workers should be self-contained with minimal input.
2. **Every worker must append to `agent_results`.** This is how the fan-in `research_collect` node knows all parallel work has completed and its results accumulated.
3. **Guard against Groq rate limits.** If both `kb_worker` and `web_worker` call the LLM simultaneously, you may hit tokens/min limits. In Module 8, `kb_worker` is a stub (no LLM) — full parallel LLM calls are introduced in Module 9 with rate-limit guards.
4. **Edges from workers must all converge on the fan-in node.** Both `kb_worker` and `web_worker` have `.edge("kb_worker", "research_collect")` and `.edge("web_worker", "research_collect")` in the graph.
5. **The mapping parameter on `conditional_edge` must be `None` for Send routes.** Passing a `dict` when the router returns `list[Send]` causes a runtime error.

### 🏗️ PRODUCTION IMPLEMENTATION

**`agents/supervisor.py` — `route_after_supervisor` (the Send router):**

```python
def route_after_supervisor(state: AgentState):
    """
    Router: decides what happens after the supervisor dispatches.

    - research tickets: emit Send fan-out to kb_worker + web_worker
    - all other tickets: go directly to respond

    Returns:
        list[Send] for research, str "respond" for all others.
    """
    active_agent = state.get("active_agent")
    if active_agent == "research":
        query = state.get("query") or ""
        return [
            Send("kb_worker", {"query": query}),
            Send("web_worker", {"query": query}),
        ]
    return "respond"
```

**`graph.py` — wiring Send-capable conditional edge:**

```python
.conditional_edge("supervisor_node", route_after_supervisor, {"respond": "respond"})
```

Note: the mapping `{"respond": "respond"}` handles the non-research path. LangGraph merges string returns from the router with the mapping, and passes `list[Send]` returns directly to the scheduler.

**`core/graph_builder.py` — updated to support optional mapping:**

```python
def conditional_edge(
    self,
    src: str,
    router_fn: Callable,
    mapping: dict[str, str] | None = None,
) -> "GraphBuilder":
    if mapping is not None:
        self._graph.add_conditional_edges(src, router_fn, mapping)
    else:
        self._graph.add_conditional_edges(src, router_fn)
    return self
```

### 🧪 TESTS

```python
# tests/unit/test_multi_agent.py — TestSendAPIRouting class

class TestSendAPIRouting:
    def test_route_after_supervisor_returns_send_list_for_research(self, base_state):
        base_state["active_agent"] = "research"
        base_state["query"] = "How does SSL work?"
        result = route_after_supervisor(base_state)
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(s, Send) for s in result)

    def test_route_after_supervisor_targets_correct_nodes(self, base_state):
        base_state["active_agent"] = "research"
        base_state["query"] = "SSL"
        sends = route_after_supervisor(base_state)
        targets = {s.node for s in sends}
        assert "kb_worker" in targets
        assert "web_worker" in targets

    def test_route_after_supervisor_sends_query_payload(self, base_state):
        base_state["active_agent"] = "research"
        base_state["query"] = "network latency"
        sends = route_after_supervisor(base_state)
        for send in sends:
            assert send.arg["query"] == "network latency"

    def test_route_after_supervisor_returns_string_for_non_research(self, base_state):
        base_state["active_agent"] = "technical"
        result = route_after_supervisor(base_state)
        assert result == "respond"

    def test_route_after_supervisor_handles_none_active_agent(self, base_state):
        base_state["active_agent"] = None
        result = route_after_supervisor(base_state)
        assert result == "respond"
```

### ✅ CONCEPT CHECKPOINT

1. `Send("kb_worker", {"query": query})` sends only `{"query": query}` to `kb_worker`. How does `kb_worker` access `thread_id` for logging?
2. Why does `route_after_supervisor` return `"respond"` (a string) for non-research tickets instead of `Send("respond", {})`?
3. What happens to the graph if you add `.edge("kb_worker", "respond")` instead of `.edge("kb_worker", "research_collect")`? Is the fan-in still correct?

---

## 📌 CONCEPT 85 & 86: Fan-Out & Fan-In  [Module 8]
**Layer:** Graph | **Complexity:** Advanced

### 🔍 WHAT

**Fan-out** is the act of dispatching one piece of work to multiple parallel workers simultaneously. **Fan-in** is collecting those workers' results back into a single coherent state.

In our app:
- **Fan-out**: `route_after_supervisor` returns `[Send("kb_worker", ...), Send("web_worker", ...)]`
- **Fan-in**: `research_collect_node` runs after both workers complete and reads `agent_results` (which already contains both workers' output via `operator.add`)

Together, fan-out + fan-in is the **map-reduce pattern** applied to LangGraph nodes.

**Mental model:** Sending survey questionnaires to 100 customers simultaneously (fan-out) and then tallying all the responses when they return (fan-in).

### 🤔 WHY

Without fan-out, research must be sequential:

```python
# ❌ WRONG — sequential: waits for KB before starting web search
kb_result = await _search_kb(query)     # 2 seconds
web_result = await _search_web(query)   # 2 seconds
# Total: ~4 seconds
```

With fan-out:
```python
# ✅ RIGHT — parallel: both searches run at the same time
Send("kb_worker", {"query": query}),    # runs concurrently
Send("web_worker", {"query": query}),   # runs concurrently
# Total: ~2 seconds (limited by the slower of the two)
```

Without the `operator.add` reducer on `agent_results`, the fan-in would silently lose one worker's results:

```python
# ❌ WRONG — scalar field: second worker overwrites first
agent_results: list[dict]           # kb_worker writes, web_worker OVERWRITES

# ✅ RIGHT — operator.add reducer: both are appended
agent_results: Annotated[list[dict], operator.add]   # both workers preserved
```

### 📍 SEQUENCE THROUGH THE FAN

```
supervisor_node writes:
  active_agent = "research"
  query = "How does JWT work?"

route_after_supervisor → [Send("kb_worker"), Send("web_worker")]

  kb_worker (concurrent)          web_worker (concurrent)
  ─────────────────────           ─────────────────────
  reads:  state["query"]          reads:  state["query"]
  writes: agent_results += [      writes: agent_results += [
    {agent:"kb_worker", ...}        {agent:"web_worker", ...}
  ]                               ]
  writes: audit_events += [...]   writes: audit_events += [...]

  ───── both complete ──────────────────────────────────

  operator.add merges both agent_results lists

research_collect_node
  reads: state["agent_results"]   # contains BOTH workers' results
  filters: research sources only
  writes: audit_events += [{action:"results_collected", source_count:2}]

respond_node
  reads: state["agent_results"]   # merged research results
  generates final response
```

### ⚖️ PROS & CONS

| Pros | Cons |
|------|------|
| Halves latency when two searches take equal time | Workers share no state — cannot use one worker's output in another |
| Each worker is independently testable and observable | Groq rate limits: parallel LLM calls exhaust tokens/min faster |
| `operator.add` reducer handles merging automatically — no manual code | Fan-in node must handle the case where one worker fails and produces no output |
| Clean graph topology — no asyncio wiring in node code | Order of results in `agent_results` is non-deterministic |

### 🆚 DIFFERENCES & COMPARISONS

| | Fan-Out via Send | `asyncio.gather` inside one node |
|--|-----------------|----------------------------------|
| Observability | Each worker is a named node in LangSmith | Single opaque node in traces |
| Failure isolation | One worker failing does not stop the other | One exception in `gather` propagates to all |
| State isolation | Each worker gets only its Send payload | All coroutines share the closure scope |
| Reducer needed | Yes — `operator.add` on the shared field | No — results collected manually |

### 💡 BEST PRACTICES

1. **`operator.add` is mandatory on any field written by multiple parallel workers.** Without it, one worker silently overwrites the other. This is a silent data-loss bug — no exception, no warning.
2. **Fan-in node is always a simple aggregator.** `research_collect_node` does NOT synthesise or call an LLM — it just counts sources and writes an audit event. Synthesis happens in `respond_node` downstream.
3. **Handle empty results in the fan-in.** A worker might return an empty list (stub, failed search, etc.). `research_collect_node` filters by agent name to count only the workers that actually produced output — it never assumes both workers succeeded.
4. **Log worker identity in every `agent_results` entry.** Include `"agent": "kb_worker"` or `"agent": "web_worker"` so `research_collect_node` can distinguish them and `respond_node` can attribute each result.
5. **`audit_events` needs `operator.add` too.** Both workers append to `audit_events`. Without the reducer, one audit event is lost — and a silent gap in the audit trail is a compliance problem.

### 🏗️ PRODUCTION IMPLEMENTATION

**`agents/supervisor.py` — kb_worker_node, web_worker_node, research_collect_node:**

```python
@with_retry
async def kb_worker_node(state: AgentState, config: RunnableConfig) -> dict:
    """
    Fan-out worker: searches the internal knowledge base.
    Module 8 stub — full CRAG retrieval in Module 9.

    Reads:  state["query"]       — the search query
    Writes: state["agent_results"] — one KB result entry (appended)
            state["audit_events"]  — one audit event (appended)
    """
    query = state.get("query") or ""
    thread_id = _get_thread_id(config)
    # Module 8 stub — returns empty content, doc_count=0
    return {
        "agent_results": [{
            "agent": "kb_worker",
            "source": "knowledge_base",
            "query": query,
            "content": "",
            "doc_count": 0,
        }],
        "audit_events": [{
            "node": "kb_worker_node",
            "action": "kb_searched",
            "thread_id": thread_id,
            "query_len": len(query),
        }],
    }


@with_retry
async def web_worker_node(state: AgentState, config: RunnableConfig) -> dict:
    """
    Fan-out worker: performs live web search via Tavily/SearchTool.

    Reads:  state["query"]       — the search query
    Writes: state["agent_results"] — one web result entry (appended)
            state["audit_events"]  — one audit event (appended)
    """
    query = state.get("query") or ""
    thread_id = _get_thread_id(config)
    search_content = ""
    if query:
        try:
            from tools.search_tool import SearchTool
            search_content = SearchTool()._run(query=query, max_results=5)
        except Exception as e:
            search_content = f"[Search failed: {e}]"
    return {
        "agent_results": [{
            "agent": "web_worker",
            "source": "web_search",
            "query": query,
            "content": search_content,
        }],
        "audit_events": [{
            "node": "web_worker_node",
            "action": "web_searched",
            "thread_id": thread_id,
            "query_len": len(query),
        }],
    }


async def research_collect_node(state: AgentState, config: RunnableConfig) -> dict:
    """
    Fan-in collector: runs after both kb_worker and web_worker complete.
    Does NOT synthesise results — that is respond_node's job.
    Counts sources and writes an audit event.

    Reads:  state["agent_results"] — already merged by operator.add reducer
    Writes: state["audit_events"]  — one collected event (appended)
    """
    thread_id = _get_thread_id(config)
    agent_results = state.get("agent_results", [])
    research_sources = [
        r for r in agent_results
        if r.get("agent") in ("kb_worker", "web_worker")
    ]
    logger.info("research_collect_node: thread=%s sources=%d", thread_id, len(research_sources))
    return {
        "audit_events": [{
            "node": "research_collect_node",
            "action": "results_collected",
            "thread_id": thread_id,
            "source_count": len(research_sources),
        }],
    }
```

**`state.py` — reducers that make fan-in safe:**

```python
# These two reducers are what make parallel workers safe:
agent_results: Annotated[list[dict], operator.add]
# kb_worker appends its entry, web_worker appends its entry — both survive

audit_events: Annotated[list[dict], operator.add]
# Both workers append audit events — neither overwrites the other
```

### 🧪 TESTS

```python
# tests/unit/test_multi_agent.py — TestFanOutFanIn class

class TestFanOutFanIn:
    @pytest.mark.asyncio
    async def test_kb_worker_node_returns_agent_results(self, runnable_config):
        state = {"query": "How do I cancel?", "agent_results": [], "audit_events": []}
        result = await kb_worker_node(state, runnable_config)
        assert len(result["agent_results"]) == 1
        assert result["agent_results"][0]["agent"] == "kb_worker"

    @pytest.mark.asyncio
    async def test_web_worker_node_returns_agent_results(self, runnable_config):
        state = {"query": "test query", "agent_results": [], "audit_events": []}
        with patch("agents.supervisor.SearchTool") as mock_tool_cls:
            mock_tool_cls.return_value._run.return_value = "search results"
            result = await web_worker_node(state, runnable_config)
        assert len(result["agent_results"]) == 1
        assert result["agent_results"][0]["agent"] == "web_worker"
        assert result["agent_results"][0]["content"] == "search results"

    @pytest.mark.asyncio
    async def test_research_collect_counts_both_workers(self, base_state, runnable_config):
        """Fan-in node sees both workers' results after operator.add merges them."""
        base_state["agent_results"] = [
            {"agent": "kb_worker", "content": ""},
            {"agent": "web_worker", "content": "results"},
        ]
        result = await research_collect_node(base_state, runnable_config)
        event = result["audit_events"][0]
        assert event["action"] == "results_collected"
        assert event["source_count"] == 2

    @pytest.mark.asyncio
    async def test_research_collect_handles_empty_results(self, base_state, runnable_config):
        """Fan-in must not crash when agent_results is empty."""
        base_state["agent_results"] = []
        result = await research_collect_node(base_state, runnable_config)
        assert result["audit_events"][0]["source_count"] == 0

    def test_operator_add_reducer_accumulates_across_workers(self):
        """Demonstrates why operator.add is required for parallel workers."""
        import operator
        existing = [{"agent": "kb_worker", "content": "kb data"}]
        new_entry = [{"agent": "web_worker", "content": "web data"}]
        merged = operator.add(existing, new_entry)
        assert len(merged) == 2
        agents = {r["agent"] for r in merged}
        assert "kb_worker" in agents
        assert "web_worker" in agents
```

### ✅ CONCEPT CHECKPOINT

1. Both `kb_worker` and `web_worker` write to `agent_results`. What would happen if `agent_results` used a plain `list[dict]` type annotation without `operator.add`?
2. `research_collect_node` does not call an LLM. What is the purpose of this node — why not just let `respond_node` read `agent_results` directly?
3. The `kb_worker` is a stub in Module 8 — it returns `content: ""`. Will this cause `research_collect_node` to fail? What does it count?

---

## 📌 CONCEPT 82: Agent Coordination  [Module 8]
**Layer:** Graph | **Complexity:** Intermediate

### 🔍 WHAT

Agent Coordination is how multiple agents share information and collaborate to produce a single final response. In LangGraph, coordination happens through the **shared state** and **append-only reducers**: each agent writes its contribution to `agent_results`, and downstream nodes read the accumulated list.

No agent has a direct reference to another agent. They coordinate indirectly — through the state that the graph manages.

**Mental model:** A relay race baton. Runner 1 (supervisor) carries the state to the handoff zone, Runner 2 (specialist agent) takes the baton (reads state), adds their contribution to the baton (writes to `agent_results`), and passes it to Runner 3 (respond_node). Runners never hand off to each other directly — the baton moves through the track in a defined sequence.

### 🤔 WHY

Without `agent_results` as a coordination mechanism:

```python
# ❌ WRONG — final_response is overwritten by whichever agent runs last
class BillingAgent:
    async def process(self, state, config):
        return {"final_response": "billing response"}   # ← writes final_response

class TechnicalAgent:
    async def process(self, state, config):
        return {"final_response": "technical response"} # ← overwrites billing's output
```

With `agent_results`:

```python
# ✅ RIGHT — each agent appends its contribution; respond_node synthesises
class BillingAgent:
    async def process(self, state, config):
        return {"agent_results": [{"agent": "billing", "action": "...", ...}]}

class TechnicalAgent:
    async def process(self, state, config):
        return {"agent_results": [{"agent": "technical", "action": "...", ...}]}

# respond_node reads all agent_results and generates one coherent response
```

### 📍 COORDINATION PROTOCOL

The `agent_results` protocol used consistently across all agents:

```python
# Every agent process() appends exactly one entry to agent_results:
{
    "agent":      str,    # agent name: "billing" | "technical" | "general" | "research"
    "action":     str,    # what this agent did: "ticket_received" | "research_started"
    "thread_id":  str,    # for cross-agent correlation in logs
    # optional fields added per agent:
    "source":     str,    # "knowledge_base" | "web_search" (workers only)
    "query":      str,    # search query (workers only)
    "content":    str,    # retrieved content (workers only)
    "doc_count":  int,    # KB retrieval count (kb_worker only)
}
```

This is an **informal protocol** — no Pydantic model enforces it in Module 8. In a production system with strict contracts between agents, you would use a `TypedDict` or `Pydantic` model per agent type.

### 🌍 WHERE IT FITS

```
AgentState.agent_results: Annotated[list[dict], operator.add]
                                                  ↑
                                    This reducer is the coordination glue

BillingAgent.process()    → agent_results += [{"agent":"billing",  ...}]
TechnicalAgent.process()  → agent_results += [{"agent":"technical",...}]
kb_worker_node()          → agent_results += [{"agent":"kb_worker",...}]
web_worker_node()         → agent_results += [{"agent":"web_worker",...}]
                                                ↓
respond_node reads all entries and generates the final response.
```

### ⚖️ PROS & CONS

| Pros | Cons |
|------|------|
| Agents are decoupled — no agent imports another agent | `agent_results` is untyped `list[dict]` — protocol violations are silent |
| `operator.add` makes multi-agent writes safe by design | Growing list across a long session — must be pruned or it grows indefinitely |
| `respond_node` has a single, consistent place to read all agent output | No ordering guarantee for parallel workers (kb vs web might swap) |
| Easy to add agent metadata (tokens used, latency, sources) | Debugging requires inspecting the full list — not obvious at a glance |

### 🆚 DIFFERENCES & COMPARISONS

| | `agent_results` (this app) | LangGraph `messages` field |
|--|---------------------------|---------------------------|
| Schema | Untyped `dict` — flexible | `BaseMessage` typed — strict |
| Reducer | `operator.add` — plain append | `add_messages` — deduplicates by ID |
| Purpose | Agent metadata + content for respond_node | Conversation history for LLM context |
| Ordering | Append order (non-deterministic in parallel) | Insertion order preserved |

### 💡 BEST PRACTICES

1. **Every agent appends exactly one `agent_results` entry.** Appending zero entries silently breaks `research_collect_node`'s source count. Appending two entries from one agent inflates the count.
2. **Always include `"agent"` and `"thread_id"` in every entry.** These are the minimum fields needed for `research_collect_node` to filter and correlate.
3. **Move to typed entries in production.** The `dict` protocol works for four agents in a learning project. A production system with 20+ agents should use a `TypedDict` per agent type and validate on write.
4. **`respond_node` reads `agent_results` but does not clear it.** State is immutable within a turn. If you need to reset `agent_results` between turns, handle it in the entry node (e.g., `memory_retrieval_node` could initialize it to `[]`).
5. **Log `agent_results` length in `research_collect_node`, not in individual workers.** This gives you one observable signal for the fan-in rather than scattered per-worker logs.

### 🏗️ PRODUCTION IMPLEMENTATION

**`state.py` — the reducer that enables coordination:**

```python
# This single line is what makes multi-agent coordination safe:
agent_results: Annotated[list[dict], operator.add]
# Without operator.add: each agent.process() return value overwrites the previous one.
# With operator.add: each return value is appended — all agents' outputs coexist.
```

**How `respond_node` uses the accumulated results:**

```python
# respond_node reads agent_results to understand which agent handled the ticket
async def respond_node(state: AgentState, config: RunnableConfig) -> dict:
    active_agent = state.get("active_agent") or "general"
    agent_results = state.get("agent_results", [])

    # Find research sources for research tickets
    research_results = [
        r for r in agent_results
        if r.get("agent") in ("kb_worker", "web_worker")
        and r.get("content")
    ]

    # Build context from research results (if any)
    context = "\n\n".join(r["content"] for r in research_results) if research_results else ""

    # Call LLM with the accumulated context
    response = await _call_llm(state["messages"], active_agent, context, config)
    return {"final_response": response}
```

### 🧪 TESTS

```python
# tests/unit/test_multi_agent.py — TestAgentCoordination class

class TestAgentCoordination:
    @pytest.mark.asyncio
    async def test_all_agents_append_to_agent_results(self, base_state, runnable_config):
        """Every specialist agent must contribute to agent_results."""
        for agent_cls, ticket_type in [
            (TechnicalAgent, "technical"),
            (GeneralAgent, "general"),
        ]:
            base_state["ticket_type"] = ticket_type
            agent = agent_cls()
            result = await agent.process(base_state, runnable_config)
            assert "agent_results" in result
            assert len(result["agent_results"]) >= 1
            assert result["agent_results"][0]["agent"] == ticket_type

    @pytest.mark.asyncio
    async def test_agent_results_accumulate_via_operator_add(self, base_state, runnable_config):
        """Simulates operator.add behaviour: two agents' results coexist."""
        import operator
        technical_result = [{"agent": "technical", "action": "ticket_received"}]
        kb_result = [{"agent": "kb_worker", "source": "knowledge_base"}]
        merged = operator.add(technical_result, kb_result)
        assert len(merged) == 2
        agents = {r["agent"] for r in merged}
        assert "technical" in agents
        assert "kb_worker" in agents

    @pytest.mark.asyncio
    async def test_all_agents_write_audit_events(self, base_state, runnable_config):
        """Every agent must write at least one audit event."""
        for agent_cls in [TechnicalAgent, GeneralAgent, ResearchAgent]:
            agent = agent_cls()
            result = await agent.process(base_state, runnable_config)
            assert "audit_events" in result
            assert len(result["audit_events"]) >= 1
```

### ✅ CONCEPT CHECKPOINT

1. `agent_results` uses `operator.add` as its reducer. What is the difference between `operator.add` on a list and `operator.add` on an integer? Why does the type of the annotated field matter?
2. `respond_node` reads `agent_results` to find research content. What happens on a billing ticket where `active_agent = "billing"` and `agent_results` contains only `[{"agent": "billing", "action": "ticket_received"}]`? Does the `research_results` filter find anything?
3. The coordination protocol uses plain `dict` entries. Name one advantage and one disadvantage of switching to a `TypedDict` per agent type.

---

## Module 8: Graph Topology Change

**Before Module 8** (Modules 1–7 — technical, general, research had dedicated nodes):

```
[START] → [memory_retrieval] → [classify_ticket]
                                       │
           ┌───────────────────────────┼─────────────────────────┐
           ▼                           ▼                          ▼
    [billing_node]             [technical_node]            [general_node]
           │                           │                          │
    route_after_billing                │               (research_node — Module 7+)
           │                           └──────────────────────────┤
    ┌──────┴──────┐                                               ▼
    │             │                                          [respond]
    ▼             ▼                                               │
[approval_check]  └──► [respond]                        [memory_write] → END
    │
    ▼
[respond] → [memory_write] → END
```

**After Module 8** (technical, general, research all route through supervisor; 11 nodes):

```
[START] → [memory_retrieval] → [classify_ticket]
                                       │
           ┌───────────────────────────┤ ROUTE_KEYS
           │                           │
    billing ▼                 others   ▼
    [billing_node]         [supervisor_node]
           │                      │
    route_after_billing    registry.get_agent(ticket_type).process()
           │                      │
    ┌──────┴──────┐         route_after_supervisor()
    │             │                │
    ▼             ▼      research  │         non-research
[approval_check]  └─►   [Send("kb_worker"), Send("web_worker")]   "respond"
    │                         │           │                           │
    ▼                  (parallel)  (parallel)                        │
[respond] ◄──────────── [kb_worker]  [web_worker]                   │
    ▲                         │           │                          │
    │                         └─────┬─────┘                         │
    │                               ▼                                │
    └───────────────── [research_collect] ◄──────────────────────────┘
                                    │
                               [respond]
                                    │
                          [memory_write] → END
```

**ROUTE_KEYS change in `agents/nodes.py`:**

```python
# Before Module 8
ROUTE_KEYS = {
    "billing":   "billing_node",
    "technical": "technical_node",
    "general":   "general_node",
    "research":  "research_node",
}

# After Module 8
ROUTE_KEYS = {
    "billing":   "billing_node",    # keeps its own node (HITL intact)
    "technical": "supervisor_node", # ← now routes through supervisor
    "general":   "supervisor_node", # ← now routes through supervisor
    "research":  "supervisor_node", # ← now routes through supervisor
}
```

**Why billing keeps its own node:** Billing has HITL (`interrupt_before=["approval_check"]`) that was built in Module 7. Moving billing through the supervisor would require the supervisor to know about HITL, violating Single Responsibility. The supervisor handles non-HITL agents; billing keeps its specialised graph path.

**Node count: 10 → 11:**
`memory_retrieval`, `classify_ticket`, `billing_node`, `approval_check`, `supervisor_node` *(new)*, `kb_worker` *(new)*, `web_worker` *(new)*, `research_collect` *(new)*, `respond`, `memory_write` — plus `END`.

---

## Module 8: Key Lessons

| Lesson | Detail |
|--------|--------|
| Supervisor = thin dispatcher | `supervisor_node` contains zero business logic. It calls `get_agent(ticket_type)` and delegates. All logic lives in the agents. |
| Registry = OCP in practice | Adding a new agent requires one new file + one `@register_agent` decorator. Zero changes to `supervisor_node`, `graph.py`, or any existing agent. |
| `ensure_agents_registered()` | Test teardown clears `_REGISTRY`. Python's module cache means `@register_agent` won't re-run on re-import. `ensure_agents_registered()` force-imports all modules to re-register. |
| Send API returns a list, not a string | `route_after_supervisor` returns `list[Send]` for research tickets and `"respond"` (string) for others. The same router function handles both. `conditional_edge` mapping must be provided for the string path. |
| `operator.add` is mandatory for parallel writers | Both `kb_worker` and `web_worker` write to `agent_results` and `audit_events`. Without `operator.add`, one silently overwrites the other — no exception, no warning, data lost. |
| Fan-in is an aggregator, not a synthesiser | `research_collect_node` counts sources and writes an audit event. It does NOT call an LLM. Synthesis is `respond_node`'s job. Conflating the two violates Single Responsibility. |
| Billing HITL unchanged | Module 7's `interrupt_before=["approval_check"]` is preserved. The supervisor doesn't touch the billing path. Multi-agent refactoring should never require rewiring HITL flows. |
| Module 8 test count | Module 8 added 49 multi-agent tests. All 436 tests pass. |

---

# Module 9 — RAG Pipeline
*Concepts 89–108 · Session 6 · ~3.5 hours*
*Unlocks: CRAG knowledge base — tickets get answers from internal docs before hitting the LLM*

---

## 📌 CONCEPT 89: RAG Subgraphs  [Module 9]

### 🔍 WHAT

A RAG (Retrieval-Augmented Generation) subgraph is a self-contained LangGraph `StateGraph` that encapsulates the full retrieval pipeline — search, grade, rewrite, generate, verify — as a composable unit callable by the parent graph.

**Mental model:** Think of it like a reference library attached to a call centre. When an agent doesn't know the answer from memory, they walk to the library (invoke the RAG subgraph), search for relevant documents, read them, and come back with a grounded answer. The library is a separate room with its own process — the agent doesn't need to know how it works internally.

**Where it lives:** `rag/graph.py` — Graph Layer. The subgraph is compiled with `get_rag_graph()` (cached via `lru_cache`) and invoked by `kb_worker_node` in `agents/supervisor.py`.

---

### 🤔 WHY

**The problem it solves:** Without RAG, the LLM answers from training data alone. It hallucinates company-specific policies, invents refund amounts, and cites procedures that don't exist. RAG grounds the LLM's answers in your actual documentation.

**What breaks without RAG:**

```python
# ❌ WITHOUT RAG — LLM answers from training data
response = llm.invoke("What is our refund policy?")
# → "Refunds are typically processed in 7–10 business days..." (invented!)
# → Our actual policy: 3–5 days for amounts under $500
```

**With RAG:**

```python
# ✅ WITH RAG — LLM answers from retrieved KB documents
docs = faiss_store.similarity_search("refund policy", k=5)
response = llm.invoke(f"Context: {docs}\n\nQuestion: What is our refund policy?")
# → "According to our billing policy, refunds under $500 are processed in 3–5 business days."
```

**In our app:** The research path (`kb_worker_node`) needs to answer questions about billing rules, API usage, and general FAQ — all company-specific. The RAG subgraph retrieves from `data/kb/` (FAISS-indexed), grades relevance, rewrites the query if needed, generates a grounded answer, and checks for hallucination before returning.

---

### 📍 WHEN TO USE

**Use RAG subgraphs when:**
- ✅ Answers must be grounded in internal documentation
- ✅ Questions are company-specific (policies, procedures, API details)
- ✅ Hallucination is unacceptable (billing, compliance, legal)
- ✅ Knowledge base is updated independently of the LLM
- ✅ You want traceability — which docs produced this answer?

**Don't use RAG when:**
- ❌ The question is general knowledge (use the LLM directly — RAG adds latency for no gain)
- ❌ The KB has fewer than ~50 documents (just include them all in the system prompt)
- ❌ Retrieval latency is a hard blocker (RAG adds 200–800ms per query)

---

### 🌍 WHERE IT FITS

```
Parent graph:
  classify_ticket → supervisor_node
                          │
                   route_after_supervisor
                    /              \
              kb_worker         web_worker
                  │
          [RAG subgraph boundary]
                  │
              retrieve         ← FAISS similarity search + reranking
                  │
          grade_retrieval      ← LLM grades each doc
                  │
         ┌────────┴────────┐
         ▼                 ▼
      generate          rewrite_query    ← CRAG loop
         │                  │
  hallucination_check    retrieve (loop)
         │
      [result back to parent: retrieved_docs, final_response]
```

**Interactions with prior concepts:**
- **Module 8 Send API**: `kb_worker_node` is dispatched via `Send("kb_worker", {...})` — the RAG subgraph runs inside this worker.
- **Module 5 Memory**: retrieved documents feed into `retrieved_docs` (Annotated list reducer), same as memory results.
- **Module 4 Checkpointing**: the subgraph shares the parent's checkpoint context — no separate checkpointer needed.

---

### ⚖️ PROS & CONS

| Pros | Cons |
|------|------|
| Grounded answers — KB docs are authoritative | Added latency: 200–800ms per query |
| Traceable — know exactly which docs produced each answer | Cold start: must build FAISS index before first run |
| Updatable: rebuild index without retraining the LLM | Chunking quality affects retrieval precision |
| CRAG adds self-correction: bad retrieval → query rewrite | More complex graph topology |
| Hallucination check adds a quality gate | Embedding cost (one-time at ingest; free at query time) |

---

### 🔀 MULTIPLE APPROACHES

**Standard RAG:** retrieve → generate. Simple. No quality checking. Use when speed matters and occasional hallucination is tolerable.

**CRAG (Corrective RAG):** retrieve → grade → {generate if relevant, rewrite+retrieve if not}. Adds the query rewrite loop. Use when retrieval quality is inconsistent or queries are ambiguous (our app uses this).

**Self-RAG:** retrieve → generate → self-reflect → {accept or regenerate}. The LLM critiques its own output. Higher quality, higher cost. Use when precision is critical (medical, legal).

**Adaptive RAG:** route the query first — complex queries go to RAG, simple ones go directly to the LLM. Saves latency on easy questions.

```python
# Standard RAG (simplest)
docs = store.similarity_search(query, k=5)
answer = llm.invoke(f"{docs}\n\n{query}")

# CRAG (what we built)
# → Adds grade_retrieval → rewrite_query loop
# → Adds hallucination_check after generate
```

---

### 💡 BEST PRACTICES

1. **Package RAG as a subgraph, not inline nodes.** Inline nodes scatter retrieval logic across your main graph. A subgraph encapsulates it — swap CRAG for Self-RAG by changing one import.

2. **Cap document content in prompts.** Pass at most 5 chunks, each truncated to ~600 chars. Longer contexts increase latency, cost, and risk of the LLM ignoring key passages.

3. **Always build the index before the app starts.** Run `python -m rag.ingest` as part of your deployment pipeline, not lazily on first request. First-request latency spikes are user-visible.

4. **Keep graders as pure functions.** Grade logic lives in `rag/grader.py` as plain functions — not nodes. Pure functions are unit-testable without LangGraph, RunnableConfig, or state mocking.

5. **Return gracefully when the index is missing.** `get_vector_store()` returns `None` (not raises) when the index isn't built. `retriever_node` checks for `None` and returns empty results. The app runs without RAG — degraded, not broken.

---

### 🏗️ PRODUCTION IMPLEMENTATION

```python
# rag/__init__.py — public API (callers import from here, not submodules)
from rag.graph import get_rag_graph, get_retriever_node
from rag.ingest import build_index

__all__ = ["get_rag_graph", "get_retriever_node", "build_index"]
```

```python
# rag/graph.py — CRAG subgraph
from functools import lru_cache
from core.graph_builder import GraphBuilder
from state import AgentState

@lru_cache(maxsize=1)
def get_rag_graph():
    """Compiled CRAG subgraph. lru_cache: built once, reused everywhere."""
    from rag.retriever import make_retriever_node

    return (
        GraphBuilder(AgentState)
        .node("retrieve", make_retriever_node())
        .node("grade_retrieval", retrieval_grade_node)
        .node("rewrite_query", rewrite_query_node)
        .node("generate", generate_node)
        .node("hallucination_check", hallucination_check_node)
        .edge("retrieve", "grade_retrieval")
        .conditional_edge(
            "grade_retrieval",
            route_by_grade,                                    # pure function router
            {"generate": "generate", "rewrite_query": "rewrite_query"},
        )
        .edge("rewrite_query", "retrieve")                     # ← CRAG self-correction loop
        .edge("generate", "hallucination_check")
        .entry("retrieve")
        .build()                                               # no checkpointer — parent owns it
    )
```

```python
# agents/supervisor.py — kb_worker_node calls the retriever, not the full CRAG loop
@with_retry
async def kb_worker_node(state: AgentState, config: RunnableConfig) -> dict:
    query = state.get("query") or ""

    retrieved_docs = []
    if query:
        from rag.retriever import make_retriever_node
        retriever_fn = make_retriever_node()
        result = await retriever_fn(state, config)
        retrieved_docs = result.get("retrieved_docs", [])

    return {
        "retrieved_docs": retrieved_docs,
        "agent_results": [{"agent": "kb_worker", "doc_count": len(retrieved_docs), ...}],
        "audit_events": [...],
    }
```

---

### 🧪 TESTS

```python
# tests/rag/test_rag_pipeline.py

@pytest.mark.asyncio
async def test_retrieval_grade_node_sets_grade(mock_settings, sample_retrieved_docs):
    state = {"query": "refund policy", "retrieved_docs": sample_retrieved_docs}
    mock_llm = MagicMock()
    mock_llm.invoke = MagicMock(return_value=AIMessage(content="relevant\nOn topic."))

    with patch("rag.graph.get_router_llm", return_value=mock_llm):
        result = await retrieval_grade_node(state, config)

    assert result["retrieval_grade"] in ("relevant", "irrelevant", "partial")

def test_route_by_grade_irrelevant_goes_to_rewrite():
    assert route_by_grade({"retrieval_grade": "irrelevant"}) == "rewrite_query"

def test_route_by_grade_relevant_goes_to_generate():
    assert route_by_grade({"retrieval_grade": "relevant"}) == "generate"
```

---

### ✅ CONCEPT CHECKPOINT

1. What is the difference between standard RAG and CRAG?
2. Why is the RAG pipeline packaged as a subgraph instead of inline nodes?
3. Why does `get_rag_graph()` not take a checkpointer argument?

---

## 📌 CONCEPT 90: Retrieval Nodes  [Module 9]

### 🔍 WHAT

A retrieval node is a LangGraph node that queries a vector store, reranks results with a cross-encoder, and returns a structured list of `RetrievedDocument` TypedDicts. It is created via a **DI factory** (`make_retriever_node()`) that accepts optional injected dependencies — enabling testing without a real FAISS index.

**Mental model:** A retrieval node is a librarian who runs two passes:
1. **First pass** (FAISS similarity search): "Which of the 100,000 shelf locations *look* related?" — fast, imprecise, returns 20 candidates.
2. **Second pass** (cross-encoder reranking): "Of these 20, which actually *answer* the question?" — slower, precise, returns the top 5.

---

### 🤔 WHY

**Why two stages?**
- FAISS uses **bi-encoders**: query and documents are embedded independently, then compared by cosine similarity. Fast (milliseconds for 100k docs), but imprecise — it can't score a (query, document) pair jointly.
- Cross-encoders score **(query, document) pairs together** — much more accurate, but O(k) model calls. Running the cross-encoder on all 100k docs would be prohibitively slow. Running it on 20 FAISS candidates is fast.

**What breaks with only FAISS:**
```
Query: "How long does a refund take?"
FAISS rank 1: "Double charges under $200 can be refunded same-day."  ← related but wrong
FAISS rank 2: "Refunds under $500 take 3–5 business days."          ← correct answer

# Cross-encoder moves rank 2 above rank 1 — answers the question directly
```

---

### 🏗️ PRODUCTION IMPLEMENTATION

```python
# rag/retriever.py — DI factory pattern
def make_retriever_node(store=None, reranker=None):
    """
    Factory: creates a retrieval node with injected dependencies.

    Production: make_retriever_node()              # factories resolve defaults
    Test:       make_retriever_node(store=mock)    # inject mock, no real index needed
    """
    def _get_store():
        if store is not None:
            return store
        from core.factories import get_vector_store
        return get_vector_store()

    def _get_reranker():
        if reranker is not None:
            return reranker
        from llm_provider import get_reranker
        return get_reranker()

    @with_retry
    async def retriever_node(state: AgentState, config: RunnableConfig) -> dict:
        """
        Reads:  state["query"]
        Writes: state["retrieved_docs"], state["retrieval_grade"], state["query"]
        """
        query = state.get("query") or ""
        if not query.strip():
            return {"retrieved_docs": [], "retrieval_grade": "irrelevant"}

        _store = _get_store()
        if _store is None:
            return {"retrieved_docs": [], "retrieval_grade": "irrelevant"}

        # Stage 1: FAISS similarity search (k=20)
        raw_docs = _store.similarity_search(query, k=settings.rag_top_k_retrieve)

        # Stage 2: Cross-encoder reranking (top 5)
        reranked = _rerank(query, raw_docs, _get_reranker(), settings.rag_top_k_rerank)

        retrieved = [
            RetrievedDocument(content=doc.page_content, source=doc.metadata.get("source"),
                              score=score, metadata=doc.metadata)
            for doc, score in reranked
        ]
        return {"retrieved_docs": retrieved, "retrieval_grade": "relevant"}

    return retriever_node
```

**Why `@with_retry` on the inner function, not the factory?** The factory creates the closure. The inner function is what LangGraph calls. The retry wrapper belongs on the callable that LangGraph invokes.

---

### 💡 BEST PRACTICES

1. **Always inject via factory, never construct directly.** `make_retriever_node(store=mock_store)` in tests — no real FAISS index needed.
2. **Use `rag_top_k_retrieve=20`, `rag_top_k_rerank=5`.** Retrieve more than you use — FAISS recall improves with larger k, then reranking selects the best.
3. **Handle missing store gracefully.** Return empty results with `retrieval_grade="irrelevant"`, not an exception. The app degrades gracefully without a built index.
4. **Cap document content in the returned TypedDict.** The `content` field goes into LLM prompts. `content[:1000]` per chunk prevents token blowout.

---

## 📌 CONCEPT 91: Query Rewriting  [Module 9]

### 🔍 WHAT

Query rewriting is a CRAG step that rephrases the user's original question into a more retrieval-friendly form when initial retrieval returns irrelevant results. The rewriter uses a fast LLM to produce a keyword-rich, specific version of the question.

**Mental model:** A user types "my bill is messed up" — FAISS searches for "messed up bill" and finds nothing about billing procedures. The rewriter produces "invoice error billing correction policy" — keywords that match the KB documents.

---

### 🤔 WHY

User-typed questions are conversational and imprecise. FAISS vector similarity works best with specific, keyword-dense queries. The gap between how users write and how KB documents are written is a major source of retrieval failure. Query rewriting bridges that gap without user involvement.

```python
# Original:  "my bill is messed up"
# Rewritten: "invoice incorrect charge billing correction refund policy"

# Original:  "can't log in"
# Rewritten: "login authentication failure 401 unauthorized access account"
```

---

### 🏗️ PRODUCTION IMPLEMENTATION

```python
# rag/graph.py
@with_retry
async def rewrite_query_node(state: AgentState, config: RunnableConfig) -> dict:
    """
    Reads:  state["query"]
    Writes: state["query"] (rewritten)
    """
    original_query = state.get("query") or ""
    llm = get_router_llm()  # fast, cheap model — rewriting doesn't need reasoning

    prompt = [
        SystemMessage(content=(
            "You are a query optimizer for a support knowledge base. "
            "Rewrite the user's question to be more specific and keyword-rich "
            "for better document retrieval. Return ONLY the rewritten query."
        )),
        HumanMessage(content=f"Original question: {original_query}"),
    ]

    try:
        response = await llm.ainvoke(prompt, config=config)
        return {"query": response.content.strip()}
    except Exception as e:
        logger.warning("rewrite failed, keeping original. error=%s", e)
        return {"query": original_query}  # fail-safe: keep original, don't crash
```

**The CRAG loop:** `rewrite_query_node` always connects back to `retrieve` — `GraphBuilder.edge("rewrite_query", "retrieve")`. LangGraph's recursion limit (`graph_recursion_limit=50`) prevents infinite loops.

---

## 📌 CONCEPT 93: Retrieval Graders  [Module 9]

### 🔍 WHAT

A retrieval grader evaluates whether each retrieved document actually answers the user's question. The overall grade (`"relevant"`, `"irrelevant"`, `"partial"`) drives CRAG routing: relevant → answer directly; irrelevant → rewrite and retry; partial → use what we have.

**Where it lives:** `rag/grader.py` as a **pure function** — not a node. Nodes call graders; graders don't call state.

---

### 🤔 WHY

FAISS retrieves by semantic similarity, not by answer quality. A document about "billing system architecture" is semantically similar to "billing refund policy" but doesn't answer the user's question. The grader filters out false positives.

```
Query: "How long does a refund take?"
Retrieved: "Our billing system uses PostgreSQL for transaction storage."

FAISS score: 0.78 (high — "billing" matches)
Grader:      irrelevant (the doc doesn't address refund timelines)

→ CRAG rewrite: "refund processing time business days"
→ Re-retrieval: finds "Refunds take 3–5 business days"
→ Grader: relevant ✓
```

---

### 🏗️ PRODUCTION IMPLEMENTATION

```python
# rag/grader.py — pure functions, no state, no graph knowledge

def grade_document(query: str, content: str, llm) -> GradeResult:
    """Grade a single document. Pure function — easily unit-testable."""
    prompt = [
        SystemMessage(content=(
            "Grade document relevance. "
            "Respond with ONLY: relevant / irrelevant\n"
            "Then one sentence explaining why."
        )),
        HumanMessage(content=f"Question: {query}\n\nDocument:\n{content[:1000]}"),
    ]
    response = llm.invoke(prompt)
    lines = response.content.strip().split("\n", 1)
    raw_grade = lines[0].strip().lower()

    # ⚠️ CRITICAL: check "irrelevant" BEFORE "relevant"
    # "irrelevant" contains "relevant" as a substring — wrong order → all grades are "relevant"
    grade = "irrelevant" if "irrelevant" in raw_grade else "relevant"
    reason = lines[1].strip() if len(lines) > 1 else ""
    return GradeResult(grade=grade, reason=reason, scores=[1.0 if grade == "relevant" else 0.0])


def grade_retrieval(query: str, docs: list[RetrievedDocument], llm) -> str:
    """Grade all retrieved docs. Returns "relevant" | "irrelevant" | "partial"."""
    if not docs:
        return "irrelevant"

    grades = [grade_document(query, doc["content"], llm) for doc in docs]
    relevant_count = sum(1 for g in grades if g["grade"] == "relevant")
    ratio = relevant_count / len(grades)

    if ratio >= 0.7:
        return "relevant"
    elif ratio <= 0.3:
        return "irrelevant"
    return "partial"
```

**The most common mistake:** Checking `"relevant" in raw_grade` before `"irrelevant"` — the substring match causes all grades to be `"relevant"` even when the LLM says `"irrelevant"`. Always check the more specific string first.

---

### 🧪 TESTS

```python
def test_grade_document_returns_irrelevant(mock_llm):
    mock_llm.invoke.return_value = AIMessage(content="irrelevant\nOff topic.")
    result = grade_document("refund policy", "API authentication guide", mock_llm)
    assert result["grade"] == "irrelevant"  # fails if substring bug is present

def test_grade_retrieval_empty_returns_irrelevant(mock_llm):
    result = grade_retrieval("any query", [], mock_llm)
    assert result == "irrelevant"
```

---

## 📌 CONCEPT 94: Hallucination Graders  [Module 9]

### 🔍 WHAT

A hallucination grader checks whether the LLM-generated answer is supported by the retrieved documents. Returns `True` (grounded) or `False` (hallucinated). Applied after `generate_node` in the CRAG pipeline before the answer reaches the user.

**Mental model:** After the LLM writes an answer, a second "fact-checker" LLM reads both the answer and the source documents and asks: "Is every claim in this answer actually in the sources?" If not, the answer is flagged.

**Where it lives:** `rag/grader.py` — `check_hallucination()` — a pure function.

---

### 🤔 WHY

Even with good retrieval, LLMs can blend retrieved facts with training data hallucinations. A 95%-accurate answer with 5% invented policy details is worse than "I don't know" for customer support.

```python
# Retrieved doc: "Refunds under $500 take 3–5 business days."
# LLM answer:    "Refunds are processed within 24 hours."  ← hallucinated!

# Hallucination grader: NOT grounded → flag the answer
```

---

### 🏗️ PRODUCTION IMPLEMENTATION

```python
# rag/grader.py
def check_hallucination(answer: str, docs: list[RetrievedDocument], llm) -> bool:
    """
    Returns True (grounded) or False (hallucinated).
    Pure function — no state, no side effects.

    Fail-safe: returns True on LLM error (prefer showing possibly-imperfect
    answer over silently blocking the user).
    """
    if not docs:
        return True  # no docs → nothing to contradict

    context = "\n\n---\n\n".join(
        f"Source: {doc['source']}\n{doc['content'][:500]}"
        for doc in docs[:5]
    )

    prompt = [
        SystemMessage(content=(
            "Is the answer supported by the context? "
            "Respond with ONLY: grounded OR hallucinated"
        )),
        HumanMessage(content=f"Context:\n{context}\n\nAnswer:\n{answer}"),
    ]

    try:
        response = llm.invoke(prompt)
        return "grounded" in response.content.strip().lower()
    except Exception:
        return True  # fail-safe: don't block answers due to grader failure
```

**Fail-safe design:** Both `grade_document` and `check_hallucination` return safe defaults on LLM failure (`"relevant"` and `True` respectively). Grader failures must never block the user from getting an answer.

---

## 📌 CONCEPT 95: CRAG (Corrective RAG)  [Module 9]

### 🔍 WHAT

CRAG (Corrective Retrieval-Augmented Generation) extends standard RAG with a quality feedback loop: if retrieved documents are irrelevant, the query is rewritten and retrieval runs again. After generation, a hallucination check verifies the answer is grounded.

**Mental model:** Standard RAG is one trip to the library. CRAG is a persistent reference librarian: they look up your question, check if what they found actually answers it, and if not, rephrase the search and try again — until they find something useful or exhaust their attempts.

---

### 🤔 WHY

Standard RAG retrieves → generates, regardless of retrieval quality. If the first search returns irrelevant documents, the LLM either hallucinates or says "I don't know" — both are bad outcomes. CRAG adds a quality gate: bad retrieval triggers a rewrite-and-retry loop before the LLM ever sees the documents.

```
Standard RAG:     retrieve → generate  (no quality check)
CRAG:             retrieve → grade → {
                    relevant:   generate → hallucination_check
                    irrelevant: rewrite_query → retrieve (loop)
                    partial:    generate → hallucination_check
                  }
```

---

### 🌍 WHERE IT FITS

```
CRAG subgraph topology:

[retrieve]
    │
[grade_retrieval]
    │
    ├─── "relevant" / "partial" ──→ [generate]
    │                                    │
    └─── "irrelevant" ──→ [rewrite_query]  [hallucination_check]
                              │                   │
                          [retrieve]          (result → parent)
                         (loop back)
```

**Recursion guard:** The loop `rewrite_query → retrieve → grade_retrieval` can cycle if every rewrite produces irrelevant results. `settings.graph_recursion_limit = 50` (LangGraph default) terminates the loop after 50 steps.

---

### 🏗️ PRODUCTION IMPLEMENTATION

```python
# rag/graph.py — CRAG subgraph with self-correction loop
@lru_cache(maxsize=1)
def get_rag_graph():
    return (
        GraphBuilder(AgentState)
        .node("retrieve", make_retriever_node())
        .node("grade_retrieval", retrieval_grade_node)
        .node("rewrite_query", rewrite_query_node)
        .node("generate", generate_node)
        .node("hallucination_check", hallucination_check_node)
        .edge("retrieve", "grade_retrieval")
        .conditional_edge(
            "grade_retrieval",
            route_by_grade,
            {"generate": "generate", "rewrite_query": "rewrite_query"},
        )
        .edge("rewrite_query", "retrieve")      # ← self-correction loop
        .edge("generate", "hallucination_check")
        .entry("retrieve")
        .build()
    )

def route_by_grade(state: AgentState) -> str:
    grade = state.get("retrieval_grade", "irrelevant")
    return "generate" if grade in ("relevant", "partial") else "rewrite_query"
```

---

### 💡 BEST PRACTICES

1. **Use `lru_cache` on `get_rag_graph()`.** Building a `StateGraph` involves compiling the graph topology. Cache the compiled graph — build once, invoke many times.

2. **Don't reuse the hallucination check result as a loop trigger.** In the current implementation, `hallucination_check_node` sets `hallucination_check` in state but does NOT route back to `rewrite_query`. One failed check → note in state, move on. Looping on hallucination risks infinite cycles; display a warning instead.

3. **Use `get_router_llm()` for graders, `get_research_llm()` for generation.** Graders are binary classification — a fast model (gemma2-9b-it) is sufficient. Generation requires reasoning — use the powerful model (llama-3.3-70b-versatile).

4. **Cap context in prompts.** `docs[:5]` with `content[:600]` per doc. Longer contexts don't improve accuracy after ~3k tokens and increase latency significantly on Groq free tier.

---

## 📌 CONCEPT 99: Document Ingestion Pipeline  [Module 9]

### 🔍 WHAT

The ingestion pipeline loads raw KB documents, splits them into chunks, embeds them, and saves a searchable vector index. This is an **offline, one-time operation** — not part of the live request path. Run it whenever the KB changes.

**Four stages:**
1. **Load** — `load_kb_documents("data/kb/")` reads all `.md` files as `Document` objects
2. **Chunk** — `ChunkingStrategy.split(docs)` splits large documents into retrieval-sized chunks
3. **Embed + Index** — `FAISSVectorStore.add_documents(chunks)` embeds and indexes all chunks
4. **Persist** — `store.save("indexes/kb.faiss")` writes the index to disk

**Entry point:** `python -m rag.ingest`

---

### 🤔 WHY

**Why chunk at all?**
- An embedding model embeds a fixed-size window of text (typically 512 tokens for `all-MiniLM-L6-v2`).
- A 5-page policy document has 2,000+ tokens — it can't be embedded as a single unit.
- Chunking produces ~512-token pieces that fit the embedding window and can be compared individually to a query.

**Why overlap?**
- A key sentence might be split across two chunks without overlap.
- `chunk_overlap=50` repeats 50 characters at the start of each new chunk, preventing context loss at boundaries.

---

### 🏗️ PRODUCTION IMPLEMENTATION

```python
# rag/chunking/strategies.py — Strategy Pattern

class FixedSizeChunker:
    """Strategy: fixed-size chunks by character count. Fast, deterministic."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split(self, documents: list[Document]) -> list[Document]:
        chunks = self._splitter.split_documents(documents)
        # Add chunk_index to metadata for traceability
        source_counts: dict[str, int] = {}
        for chunk in chunks:
            src = chunk.metadata.get("source", "unknown")
            source_counts[src] = source_counts.get(src, 0) + 1
            chunk.metadata["chunk_index"] = source_counts[src] - 1
        return chunks
```

```python
# rag/ingest.py — build_index() entry point
def build_index(docs=None, strategy=None, index_path=None) -> None:
    settings = get_settings()
    docs = docs or load_kb_documents()

    if not docs:
        raise ValueError("build_index: no documents to index.")

    strategy = strategy or FixedSizeChunker(settings.rag_chunk_size, settings.rag_chunk_overlap)
    chunks = strategy.split(docs)

    from llm_provider import get_embeddings
    from rag.stores.faiss_store import FAISSVectorStore

    store = FAISSVectorStore(get_embeddings())
    store.add_documents(chunks)
    store.save(index_path or settings.rag_index_path)

# Run with: python -m rag.ingest
if __name__ == "__main__":
    build_index()
```

**Strategy Pattern in action:**
- Caller passes a `ChunkingStrategy` — doesn't know if it's `FixedSizeChunker` or `SemanticChunker`
- Adding a new chunker requires only implementing `.split(documents)` — no changes to `build_index()`
- This is Open/Closed Principle applied to the ingestion pipeline

---

### 💡 BEST PRACTICES

1. **Run ingestion as a CI/CD step, not lazily.** A missing index means 0 KB results, not a crash. But users get worse answers. Always build before deploy.

2. **Add `chunk_index` to metadata.** `chunk.metadata["chunk_index"] = N` lets you trace any retrieved chunk back to its position in the source document — invaluable for debugging retrieval quality.

3. **Use `FixedSizeChunker` as your default.** `SemanticChunker` produces better chunks but requires an embedding model call per document at ingest time. Only switch if retrieval quality is measurably poor.

4. **Version your index alongside your KB.** When the KB changes, rebuild the index. A stale index serving obsolete policies is worse than no index.

---

## 📌 CONCEPT 100: Vector Stores  [Module 9]

### 🔍 WHAT

A vector store is a database that stores document embeddings (dense float vectors) and supports efficient similarity search: given a query vector, find the k most similar document vectors. In our stack, FAISS runs locally with zero network calls.

**Mental model:** A vector store is a filing system where documents are sorted not alphabetically but by semantic meaning. "How long for a refund?" sits next to "Refunds take 3–5 days" because they mean the same thing — not because the words match.

---

### 🤔 WHY

Keyword search finds documents that share words with the query. Vector similarity finds documents that share *meaning*. A user who types "invoice is wrong" should find the "billing error correction" policy — keyword search won't match, vector search will.

---

### 🏗️ PRODUCTION IMPLEMENTATION

```python
# rag/protocols.py — VectorStoreProtocol (Liskov Substitution Principle)
@runtime_checkable
class VectorStoreProtocol(Protocol):
    def similarity_search(self, query: str, k: int) -> list[Document]: ...
    def add_documents(self, docs: list[Document]) -> None: ...
    def save(self, path: str) -> None: ...
```

```python
# rag/stores/faiss_store.py — satisfies VectorStoreProtocol
class FAISSVectorStore:
    """
    FAISS-backed vector store. No network calls, no cost.
    Single .faiss directory — gitignore it, rebuild from data/kb/.
    """
    def similarity_search(self, query: str, k: int = 20) -> list[Document]:
        if self._store is None:
            raise RuntimeError("No index loaded. Run python -m rag.ingest first.")
        return self._store.similarity_search(query, k=k)

    def add_documents(self, docs: list[Document]) -> None:
        if self._store is None:
            self._store = FAISS.from_documents(docs, self._embeddings)
        else:
            self._store.merge_from(FAISS.from_documents(docs, self._embeddings))

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self._store.save_local(path)

    @classmethod
    def load(cls, path: str, embeddings) -> "FAISSVectorStore":
        if not os.path.exists(path):
            raise FileNotFoundError(f"Index not found: {path}")
        instance = cls(embeddings)
        instance._store = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
        return instance
```

```python
# core/factories.py — get_vector_store() factory
def get_vector_store():
    """Returns FAISSVectorStore, or None if index not yet built."""
    settings = get_settings()
    if not os.path.exists(settings.rag_index_path):
        return None  # not an error — degrade gracefully

    from rag.stores.faiss_store import FAISSVectorStore
    from llm_provider import get_embeddings
    return FAISSVectorStore.load(settings.rag_index_path, get_embeddings())
```

**Why Protocol + factory (not direct instantiation):**
- Callers depend on `VectorStoreProtocol` — not `FAISSVectorStore` specifically
- Swapping to ChromaDB changes only `get_vector_store()` — zero changes to retriever, grader, or graph
- This is Dependency Inversion + Liskov Substitution in the same design

---

## 📌 CONCEPT 101: Embeddings  [Module 9]

### 🔍 WHAT

Embeddings convert text into dense numerical vectors (e.g. 384 floats for `all-MiniLM-L6-v2`). Semantically similar texts produce vectors that are close in embedding space. In RAG, embeddings enable the similarity search that retrieves relevant documents.

**Rule:** Always use `get_embeddings()` from `llm_provider.py`. Never instantiate `HuggingFaceEmbeddings(...)` directly in node or service code. This is the same provider abstraction rule that applies to LLMs.

---

### 🏗️ PRODUCTION IMPLEMENTATION

```python
# llm_provider.py — get_embeddings() (excerpt)
@lru_cache(maxsize=1)
def get_embeddings():
    """
    Returns the HuggingFace sentence-transformer for embeddings.
    Cached: model is loaded once (300MB+), reused for all ingest + retrieval calls.
    Model: sentence-transformers/all-MiniLM-L6-v2 (384-dimensional, fast, free).
    """
    from langchain_huggingface import HuggingFaceEmbeddings
    settings = get_settings()
    return HuggingFaceEmbeddings(model_name=settings.hf_embeddings_model)
```

**Why `lru_cache` on `get_embeddings()`?** Loading a sentence-transformer downloads and initialises a 300MB+ model. Without caching, every call to `get_vector_store()` would reload the model from disk — catastrophic latency. With `lru_cache`, it loads once per process.

**Embedding consistency rule:** The same model must be used at ingest time (when building the index) and at query time (when searching). Mixing models produces meaningless similarity scores. `get_embeddings()` as the single source guarantees this.

---

## 📌 CONCEPT 102: Reranking  [Module 9]

### 🔍 WHAT

Reranking is a second-stage retrieval step that uses a **cross-encoder** model to re-score the top-k FAISS candidates. Unlike bi-encoders (used in FAISS), cross-encoders read the query and document together, producing more accurate relevance scores.

**Bi-encoder vs cross-encoder:**
- **Bi-encoder** (FAISS): embeds query and documents *separately*, compares by cosine similarity. Fast. O(1) per query against pre-built index. Lower precision.
- **Cross-encoder** (reranker): takes (query, document) *pairs*, runs a forward pass through the model for each. Slower. O(k) per query. Higher precision.

**Rule:** Always use `get_reranker()` from `llm_provider.py`. Never instantiate `CrossEncoder(...)` directly.

---

### 🏗️ PRODUCTION IMPLEMENTATION

```python
# rag/retriever.py — _rerank() pure helper
def _rerank(query: str, docs: list[Document], reranker, top_k: int) -> list[tuple[Document, float]]:
    """
    Rerank documents using a cross-encoder. Pure function — no state.

    Args:
        query:    Search query.
        docs:     Candidates from FAISS (typically k=20).
        reranker: CrossEncoder from get_reranker().
        top_k:    Number to return after reranking (typically 5).

    Returns:
        List of (document, score) tuples, sorted by score descending.
        Falls back to original similarity order on error.
    """
    if not docs:
        return []

    try:
        pairs = [[query, doc.page_content] for doc in docs]
        scores = reranker.predict(pairs)  # O(k) cross-encoder passes
        scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return scored[:top_k]
    except Exception as e:
        logger.warning("reranker failed, using similarity order. error=%s", e)
        return [(doc, 1.0) for doc in docs[:top_k]]  # graceful fallback
```

**Note on FAISS scores:** FAISS returns L2 distance — lower is MORE similar. Cross-encoder scores are logits — higher is MORE relevant. The reranker sorts by score descending (highest = most relevant), replacing the FAISS ordering entirely.

---

## 📌 CONCEPT 106: RAG Evaluation  [Module 9]

### 🔍 WHAT

RAG evaluation measures the quality of the full retrieval pipeline offline: did we retrieve the right documents (recall)? Is the answer grounded in the documents (faithfulness)? Does the answer address the question (relevance)?

**Three metrics:**
- **Recall@k**: fraction of known-relevant sources found in top-k retrieved docs
- **Faithfulness**: is every claim in the answer supported by the retrieved docs?
- **Answer Relevance**: does the answer actually address the question asked?

**Where it lives:** `rag/evaluator.py` — offline evaluation, not part of the live graph.

---

### 🤔 WHY

Type checking and unit tests verify code correctness. RAG evaluation verifies **retrieval quality**. Your code can be perfectly correct but still retrieve irrelevant documents if:
- The chunking strategy produces chunks that don't match query semantics
- The embedding model doesn't capture domain-specific terminology well
- The FAISS index was built from outdated documents

Run `run_eval_suite()` after any change to: KB documents, chunking settings, embedding model, or retriever configuration.

---

### 🏗️ PRODUCTION IMPLEMENTATION

```python
# rag/evaluator.py — pure evaluation functions

def evaluate_retrieval_recall(query, retrieved, relevant_sources) -> float:
    """Recall@k: fraction of known-relevant sources that were retrieved."""
    if not relevant_sources:
        return 0.0
    retrieved_sources = {doc["source"] for doc in retrieved}
    return sum(1 for src in relevant_sources if src in retrieved_sources) / len(relevant_sources)

def run_eval_suite(test_cases: list[dict], llm) -> dict:
    """
    Each test case: {"query", "expected_sources", "retrieved", "answer"}
    Returns: {"recall_mean", "faithfulness_mean", "relevance_mean", "n", "by_case"}
    """
    results = [
        {
            "query": c["query"],
            "recall":       evaluate_retrieval_recall(c["query"], c["retrieved"], c["expected_sources"]),
            "faithfulness": evaluate_faithfulness(c["answer"], c["retrieved"], llm),
            "relevance":    evaluate_answer_relevance(c["query"], c["answer"], llm),
        }
        for c in test_cases
    ]
    n = len(results) or 1
    return {
        "recall_mean":       sum(r["recall"] for r in results) / n,
        "faithfulness_mean": sum(r["faithfulness"] for r in results) / n,
        "relevance_mean":    sum(r["relevance"] for r in results) / n,
        "n": n,
        "by_case": results,
    }
```

---

## Module 9: Topology Changes

No changes to the main graph topology. The RAG pipeline integrates via `kb_worker_node`:

```
classify_ticket
    │
    ├── "billing" ──→ [billing_node] ──→ [approval_check]? ──→ [respond]
    │
    └── others ──→ [supervisor_node]
                        │
                   route_after_supervisor
                    /              \
              [kb_worker]        [web_worker]
                  │                   │
          ← RAG retrieval         Tavily search
          (FAISS + reranker)
                  └──────────────────┘
                              │
                    [research_collect]
                              │
                          [respond]
                              │
                       [memory_write] → END
```

**What changed in `kb_worker_node` (Module 8 stub → Module 9 real):**

```python
# Module 8 stub
return {
    "agent_results": [{"agent": "kb_worker", "content": "", "doc_count": 0}],
    ...
}

# Module 9 real
retriever_fn = make_retriever_node()
result = await retriever_fn(state, config)
retrieved_docs = result.get("retrieved_docs", [])

return {
    "retrieved_docs": retrieved_docs,  # ← now populated
    "agent_results": [{"agent": "kb_worker", "doc_count": len(retrieved_docs), ...}],
    ...
}
```

**`get_vector_store()` in `core/factories.py` — fully implemented:**

```python
def get_vector_store():
    settings = get_settings()
    if not os.path.exists(settings.rag_index_path):
        return None  # graceful degradation — no index yet

    from rag.stores.faiss_store import FAISSVectorStore
    from llm_provider import get_embeddings
    return FAISSVectorStore.load(settings.rag_index_path, get_embeddings())
```

---

## Module 9: Key Lessons

| Lesson | Detail |
|--------|--------|
| Graders are pure functions, not nodes | `grade_document`, `grade_retrieval`, `check_hallucination` live in `rag/grader.py` with zero state dependencies. Any node can call them. Unit tests mock only the LLM, nothing else. |
| Check "irrelevant" before "relevant" | `"irrelevant"` contains `"relevant"` as a substring. Checking `"relevant" in raw_grade` first matches both → all grades are "relevant". Always check the longer string first. |
| CRAG loop needs a recursion limit | `rewrite_query → retrieve → grade` can loop forever. `settings.graph_recursion_limit = 50` is the guard. Design CRAG to converge: if the KB doesn't have the answer after 2–3 rewrites, move to web fallback. |
| `lru_cache` on `get_rag_graph()` | Building a `StateGraph` compiles the topology. Cache the compiled graph — one build per process, not one per request. Same principle as `get_settings()`. |
| Subgraph = no checkpointer | The CRAG subgraph is compiled without `checkpointer=`. It shares the parent graph's checkpoint context. Adding a separate checkpointer to a subgraph creates checkpoint isolation — state changes don't appear in the parent's history. |
| `make_retriever_node()` for testability | The DI factory lets tests inject `mock_store` and `mock_reranker` — no real FAISS index, no model downloads, no ingest step needed. This is why production nodes use factory functions, not `store = FAISSVectorStore(...)`. |
| Embedding model consistency | Same `get_embeddings()` at ingest time and query time. Different models produce different vector spaces — mixing them makes similarity scores meaningless. |
| Build the index before starting the app | `get_vector_store()` returns `None` gracefully, but KB retrieval then returns empty results. Build with `python -m rag.ingest` as part of your deploy pipeline, not lazily. |
| Module 9 test count | Module 9 added 44 RAG tests. All 480 tests pass. |

---

---

# Module 10: Enterprise Security, Governance & Observability

**Session 7 · Concepts 109–120 · ~2.5 hours**
*Unlocks: JWT auth, RBAC, LangSmith tracing, in-process metrics, FastAPI delivery layer*

---

## Concept 109: Authentication

### What
Authentication is the process of verifying WHO the caller is before they interact with your system. In our app, a JWT (JSON Web Token) is used: a signed, self-contained token that carries the caller's identity (`user_id`, `tenant_id`, `roles`). The signature is verified with a shared secret — if the token has been tampered with, verification fails.

### Why
Without authentication, anyone can submit tickets, approve refunds, or access tenant data. Authentication is the first gate in the security pipeline.

**Where it lives:** Authentication is a pre-graph guard, NOT a node inside the graph.

```
HTTP request → validate JWT (in API layer) → graph.ainvoke(authenticated state)
```

The graph assumes callers are already authenticated. `auth_node` is optional — it only logs the auth context to the audit trail.

### Key design: python-jose

```python
# security/auth.py
from jose import jwt as jose_jwt, JWTError

def validate_jwt(token: str, secret: str, algorithm: str = "HS256") -> dict:
    try:
        payload = jose_jwt.decode(token, secret, algorithms=[algorithm])
    except JWTError as e:
        raise AuthenticationError(f"invalid token: {e}")

    if "sub" not in payload:
        raise AuthenticationError("invalid token: missing 'sub'")
    if "tenant_id" not in payload:
        raise AuthenticationError("invalid token: missing 'tenant_id'")
    if "roles" not in payload:
        raise AuthenticationError("invalid token: missing 'roles'")

    return payload
```

### Dev mode: no JWT_SECRET = open access

```python
def build_auth_context(token=None, settings=None) -> AuthContext:
    settings = settings or get_settings()
    if settings.jwt_secret is None:
        # Dev mode — accept any/no token
        return AuthContext(user_id="dev-user", tenant_id=settings.default_tenant_id,
                           roles=["agent", "billing_admin", "supervisor"], authenticated=True)
    # Production mode — token required
    if token is None:
        raise AuthenticationError("no token provided")
    payload = validate_jwt(token, settings.jwt_secret.get_secret_value(), settings.jwt_algorithm)
    return AuthContext(user_id=payload["sub"], tenant_id=payload["tenant_id"],
                       roles=payload["roles"], authenticated=True)
```

**Production best practice:** Use `SecretStr` for all secrets. `settings.jwt_secret.get_secret_value()` extracts the raw value — `settings.jwt_secret` itself never logs the key.

---

## Concept 110: Authorization

### What
Authorization answers WHAT an authenticated user is allowed to do. Authentication = "who are you?"; Authorization = "what are you allowed to do?".

### Custom exception hierarchy

Every security failure in our app raises a typed exception from `core/exceptions.py`:

```python
class SupportAgentError(Exception): ...   # base — never raise directly

class AuthenticationError(SupportAgentError):
    def __init__(self, reason: str): ...

class AuthorizationError(SupportAgentError):
    def __init__(self, user_id: str, action: str, required_role: str): ...
    # Carries: user_id, action, required_role — for structured logging
```

**Why typed exceptions?** Three reasons:
1. Callers can catch by type: `except AuthorizationError` — not `except Exception`
2. Structured context in `__init__` — no string parsing to extract user_id
3. Custom hierarchy means `except SupportAgentError` catches all app errors; `except Exception` catches unexpected ones separately

### Three-layer error handling

```
Layer 1: @with_retry  →  transient failures (rate limits, timeouts) → auto-retry
Layer 2: specific except  →  expected business errors (auth, retrieval) → graceful handling
Layer 3: let propagate  →  unexpected bugs → surface, don't swallow
```

**NEVER** `except Exception` in nodes. If you don't know what went wrong, don't pretend you handled it.

---

## Concept 111: RBAC

### What
Role-Based Access Control (RBAC) maps roles to permissions. A user has roles (strings). Each role grants a set of permissions (actions). You check permissions, never roles directly.

### Policy as code

```python
# security/rbac.py — the entire policy in one dict
ROLE_PERMISSIONS: dict[str, set[str]] = {
    "agent":         {"tickets:read", "tickets:respond", "kb:search", "web:search"},
    "billing_admin": {"tickets:read", "tickets:respond", "kb:search", "web:search",
                      "refunds:approve", "refunds:reject", "billing:view"},
    "supervisor":    {"tickets:read", "tickets:respond", "tickets:escalate",
                      "kb:search", "web:search", "tenants:list", "agents:manage"},
    "readonly":      {"tickets:read", "kb:search"},
}
```

**Why a dict instead of a DB table?** The policy changes on every deploy, not on every user action. Version-controlling it gives you a full audit trail: `git log security/rbac.py` shows every permission change, by whom, when.

```python
def has_permission(roles: list[str], permission: str) -> bool:
    return permission in get_permissions(roles)

def require_permission(roles, user_id, action, required_permission) -> None:
    if not has_permission(roles, required_permission):
        raise AuthorizationError(user_id, action, required_permission)
```

### RBAC node

```python
async def rbac_check_node(state, config, required_permission="tickets:respond") -> dict:
    roles = state.get("rbac_roles", [])
    user_id = config.get("configurable", {}).get("user_id", "unknown")
    allowed = has_permission(roles, required_permission)
    audit = create_audit_event(
        node="rbac_check_node", action="rbac_check",
        user_id=user_id, tenant_id=...,
        metadata={"permission_checked": required_permission, "allowed": allowed}
    )
    if not allowed:
        return {"error": f"Access denied: missing permission '{required_permission}'",
                "audit_events": [audit]}
    return {"audit_events": [audit]}
```

---

## Concept 112: Multi-Tenant Graphs

### What
Multi-tenancy means one app instance serves multiple customers (tenants), each isolated from the others. In our app, `tenant_id` flows through every log entry, audit event, and LangSmith trace.

### Tenant isolation checklist

```
✅ tenant_id in every logger.info/warning/error call
✅ tenant_id in every audit_event dict
✅ tenant_id in every LangSmith metadata dict (get_run_metadata)
✅ FAISS index per tenant if KB is tenant-scoped (use path: indexes/{tenant_id}.faiss)
✅ thread_id prefixed with tenant_id for checkpoint isolation
❌ NEVER mix tenant documents in a shared FAISS index
❌ NEVER return data from a different tenant (check tenant_id in retrieved doc metadata)
```

---

## Concept 113-114: Governance and Audit Trails

### What
Governance = a verifiable record of every sensitive action. Audit trails = the append-only log that makes governance possible.

### The append-only reducer pattern

```python
# state.py — audit_events uses operator.add (append-only)
audit_events: Annotated[list[dict], operator.add]
```

`operator.add` appends new events. Events already in the list are NEVER modified. This is immutable by construction — no node can overwrite a previous audit entry, because returning `{"audit_events": [new_event]}` causes `operator.add([existing...], [new_event])` = `[existing..., new_event]`.

### Creating audit events

```python
from observability.audit import create_audit_event

event = create_audit_event(
    node="billing_node",
    action="refund_requested",
    user_id=user_id,
    tenant_id=tenant_id,
    metadata={"amount": 149.99, "ticket_id": ticket_id},
)
# Returns: {node, action, user_id, tenant_id, metadata, timestamp}
return {"audit_events": [event], ...}
```

---

## Concept 115: Compliance

### What
Compliance gates are nodes that check tickets against legal/regulatory requirements before processing. Our `compliance_gate_node` detects PII (Personally Identifiable Information) and flags tickets that contain it.

### PII detection (pure function)

```python
# security/compliance.py
_PII_PATTERNS: dict[str, re.Pattern] = {
    "email":  re.compile(r'\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b', re.I),
    "phone":  re.compile(r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b'),
    "ssn":    re.compile(r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b'),
    "credit_card": re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
}

def detect_pii(text: str) -> list[str]:
    """Pure function — returns detected PII type names. Empty = clean."""
    return [name for name, pattern in _PII_PATTERNS.items() if pattern.search(text)]
```

**Pure function = trivially testable:** `detect_pii("call me at 555-123-4567")` returns `["phone"]` — no LLM, no graph, no state.

---

## Concept 116: LangSmith Tracing

### What
LangSmith is Anthropic's observability platform for LLM applications. It records every LLM call, its inputs/outputs, latency, token counts, and custom metadata. You can filter traces by tenant, user, ticket type.

### Configuration

```python
# observability/tracing.py
def configure_tracing(settings=None) -> None:
    settings = settings or get_settings()
    if settings.langsmith_api_key is None:
        return  # tracing disabled — no-op

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key.get_secret_value()
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
```

### Per-request metadata tagging

```python
def get_run_metadata(tenant_id, user_id, ticket_type=None, thread_id=None) -> dict:
    meta = {"tenant_id": tenant_id, "user_id": user_id, "app": "support-research-agent"}
    if ticket_type:
        meta["ticket_type"] = ticket_type
    if thread_id:
        meta["thread_id"] = thread_id
    return meta

# In routes.py — config passed to every graph run
config = {
    "configurable": {...},
    "metadata": get_run_metadata(tenant_id=..., user_id=..., thread_id=thread_id),
}
```

**What this unlocks:** In LangSmith, filter by `tenant_id=acme-corp` to see all traces for that tenant. Filter by `ticket_type=billing` to debug billing agent performance.

---

## Concept 117: Graph Observability

### What
In-process metrics: latency per node, token usage per node, retrieval quality events. Collected via the Observer pattern (pre/post hooks) — zero business logic change.

### In-process metrics (no external service required)

```python
# observability/metrics.py
_node_latencies: dict[str, list[dict]] = defaultdict(list)

def record_node_latency(node_name: str, duration_ms: float, thread_id: str) -> None:
    _node_latencies[thread_id].append({"node": node_name, "duration_ms": duration_ms})

def get_session_metrics(thread_id: str) -> dict:
    latencies = _node_latencies[thread_id]
    total = sum(e["duration_ms"] for e in latencies)
    slowest = max(latencies, key=lambda e: e["duration_ms"], default=None)
    return {
        "node_latencies": latencies,
        "total_duration_ms": total,
        "slowest_node": (slowest["node"], slowest["duration_ms"]) if slowest else None,
        ...
    }
```

### Wired via the Observer post-hook

```python
def configure_metrics() -> None:
    def _latency_post_hook(name, state, config):
        if "duration_ms" in state:
            thread_id = config.get("configurable", {}).get("thread_id", "global")
            record_node_latency(name, state["duration_ms"], thread_id)
    register_post_hook(_latency_post_hook)
```

No node changes needed. Every node that's wrapped with `node_with_hooks()` automatically contributes latency metrics.

---

## Concept 118: Graph Testing

### Test pyramid for LangGraph apps

```
           ┌─────────────────────┐
           │   e2e (real LLMs)   │  tests/e2e/ — slow, run sparingly
           ├─────────────────────┤
           │    integration      │  tests/integration/ — compiled graph, mocked LLMs
           ├─────────────────────┤
           │       unit          │  tests/unit/ — pure functions, no I/O, fast
           └─────────────────────┘
```

**Unit tests (fastest):** test `_build_classification_prompt()`, `grade_document()`, `detect_pii()` — pure functions, zero mocking needed.

**Integration tests:** compile a real `StateGraph` with `InMemorySaver`, inject `AsyncMock` LLMs, call `graph.ainvoke()`, assert on state.

**Security tests (new in Module 10):** `tests/security/` — test RBAC policy, JWT validation, exception hierarchy, FastAPI endpoints via `TestClient`.

### Module 10 test count

| File | Tests |
|------|-------|
| `tests/security/test_security.py` | 46 |
| `tests/security/test_api.py` | 15 |
| **Total new** | **61** |

---

## Concept 119: Graph Debugging

### What
`GRAPH_DEBUG=true` enables verbose execution tracing inside LangGraph. This shows: node inputs, node outputs, state diffs after each node, routing decisions, checkpoint writes.

```python
# core/settings.py
graph_debug: bool = Field(default=False)

# graph.py
def build_graph(checkpointer=None, debug: bool = False):
    ...
    return builder.build(checkpointer=checkpointer, debug=debug)

# main.py
graph = build_graph(checkpointer=checkpointer, debug=settings.graph_debug)
```

**In FastAPI:** debug mode also enables the `/metrics/{thread_id}` endpoint and the `/docs` Swagger UI:

```python
app = FastAPI(
    docs_url="/docs" if settings.graph_debug else None,  # hide Swagger in prod
)
```

**Never enable `GRAPH_DEBUG=true` in production** — it logs full state content including messages, which may contain PII.

---

## Concept 120: Enterprise Deployment

### What
The FastAPI delivery layer wraps the LangGraph graph in an HTTP API. It's the outermost layer — handles HTTP I/O, auth, RBAC, then delegates to the graph.

### Factory pattern for the app

```python
# api/routes.py
def create_app() -> FastAPI:
    """Factory — called once per process (or per test)."""
    configure_logging()
    configure_tracing()
    configure_metrics()

    app = FastAPI(...)

    @app.on_event("startup")
    async def startup():
        checkpointer = get_checkpointer()
        app.state.graph = build_graph(checkpointer=checkpointer)
        # Graph compiled ONCE at startup — reused for all requests

    return app
```

**Why a factory function?** Same reason as `get_checkpointer()` — tests call `create_app()` to get a fresh, isolated instance. Module-level `app = FastAPI()` can't be reset between tests.

### Streaming endpoint (SSE)

```python
@app.post("/tickets")
async def submit_ticket(request: TicketRequest, auth=Depends(get_current_user)):
    require_permission(auth["roles"], auth["user_id"], "submit_ticket", "tickets:respond")
    ...
    return StreamingResponse(
        _stream_tokens(app.state.graph, initial_state, config),
        media_type="text/event-stream",
    )

async def _stream_tokens(graph, initial_state, config):
    async for _ in graph.astream(initial_state, config=config, stream_mode="updates"):
        pass  # Run the graph
    async for chunk, metadata in graph.astream({}, config=config, stream_mode="messages"):
        if metadata.get("langgraph_node") == "respond" and chunk.content:
            yield f"data: {chunk.content}\n\n"
    yield "data: [DONE]\n\n"
```

**SSE format:** Each line is `data: <content>\n\n`. The `[DONE]` sentinel tells the client the stream is complete. This is the same pattern used by OpenAI's streaming API.

---

## Module 10: Key Lessons

| Lesson | Detail |
|--------|--------|
| Auth is a pre-graph guard, not a node | JWT validation happens in `main.py` and `api/routes.py` BEFORE the graph runs. The graph assumes authenticated callers. `auth_node` only logs the context to the audit trail. |
| Policy as code > DB rows | RBAC permissions in `security/rbac.py` are version-controlled. `git log security/rbac.py` is your permission change history. DB-backed RBAC needs migrations, rollbacks, and a separate audit trail. |
| Custom exception hierarchy enables structured catch | `except AuthorizationError` catches only auth denials. `except SupportAgentError` catches all app errors. `except Exception` catches unexpected bugs. Never conflate these. |
| `SecretStr` masks secrets in logs | `settings.jwt_secret` repr is `SecretStr('**********')`. `.get_secret_value()` exposes the raw value. This prevents accidental secret logging in repr/str calls. |
| `audit_events` is append-only by construction | `operator.add` reducer + partial dict return means no node can overwrite a previous event. The audit log is immutable at the architectural level. |
| `configure_tracing()` is a no-op without LANGSMITH_API_KEY | Safe to call in all environments. Returns immediately if not configured. No crashes, no warnings. |
| In-process metrics need reset per test | `reset_metrics(thread_id)` in `setup_method`/`teardown_method`. Metrics are module-level globals — they persist across tests without explicit reset. |
| `create_app()` factory enables test isolation | Each test fixture calls `create_app()` fresh — separate settings, separate graph instance, no shared state. |
| FastAPI `TestClient` needs `raise_server_exceptions=False` for 4xx tests | `raise_server_exceptions=True` converts `HTTPException` into a Python exception before you can inspect the status code. Set `False` for tests that assert on 4xx responses. |
| Module 10 test count | 61 new tests. Full suite: 541 tests, all passing. |

---
