# llm_provider.py
"""
LLM Provider abstraction layer for support-research-agent.

Providers:
  groq        → Groq API (ultra-fast, free tier) — DEFAULT
  huggingface → HuggingFace Hub inference API (free)
  hf_local    → HuggingFace local download (fully offline)

Switch provider:
  export LLM_PROVIDER=groq           # default
  export LLM_PROVIDER=huggingface    # HF Hub API
  export LLM_PROVIDER=hf_local       # fully offline, no API key

All LangGraph nodes import ONLY from this file. Never instantiate
ChatGroq, HuggingFaceEndpoint, etc. directly in node code.
"""
import os

from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "groq")


# ── CORE FACTORY ─────────────────────────────────────────────────────────────

def get_llm(model: str = None, temperature: float = 0.0, **kwargs):
    """
    Returns a LangChain-compatible ChatModel.

    Args:
        model: Model name. Provider default used if None.
        temperature: 0.0 = deterministic, 0.7 = creative.

    Usage:
        from llm_provider import get_llm
        llm = get_llm()
        llm = get_llm("gpt-oss-20b", temperature=0.0)
    """
    if PROVIDER == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=model or "llama-3.3-70b-versatile",
            temperature=temperature,
            api_key=os.getenv("GROQ_API_KEY"),
            **kwargs,
        )
    elif PROVIDER == "huggingface":
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
        endpoint = HuggingFaceEndpoint(
            repo_id=model or "HuggingFaceH4/zephyr-7b-beta",
            temperature=max(temperature, 0.01),
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
            **kwargs,
        )
        return ChatHuggingFace(llm=endpoint)
    elif PROVIDER == "hf_local":
        from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
        pipeline = HuggingFacePipeline.from_model_id(
            model_id=model or "microsoft/phi-3-mini-4k-instruct",
            task="text-generation",
            pipeline_kwargs={
                "temperature": max(temperature, 0.01),
                "max_new_tokens": 1024,
                "do_sample": temperature > 0,
            },
        )
        return ChatHuggingFace(llm=pipeline)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER='{PROVIDER}'. Use: groq | huggingface | hf_local")


# ── ROLE-BASED FACTORIES ──────────────────────────────────────────────────────

def get_router_llm():
    """Fast, deterministic model for routing and classification nodes."""
    if PROVIDER == "groq":
        return get_llm("gemma2-9b-it", temperature=0.0)
    elif PROVIDER in ("huggingface", "hf_local"):
        return get_llm("microsoft/phi-3-mini-4k-instruct", temperature=0.0)
    return get_llm(temperature=0.0)


def get_research_llm():
    """Most capable model for research, reasoning, and response synthesis."""
    if PROVIDER == "groq":
        return get_llm("llama-3.3-70b-versatile", temperature=0.1)
    elif PROVIDER == "huggingface":
        return get_llm("HuggingFaceH4/zephyr-7b-beta", temperature=0.1)
    return get_llm(temperature=0.1)


def get_fast_llm():
    """Fastest model for tool calls, extraction, and grading tasks."""
    if PROVIDER == "groq":
        return get_llm("openai/gpt-oss-20b", temperature=0.0)
    elif PROVIDER in ("huggingface", "hf_local"):
        return get_llm("microsoft/phi-3-mini-4k-instruct", temperature=0.0)
    return get_llm(temperature=0.0)


# ── HF LOCAL PIPELINES (always local, no API key, zero cost) ─────────────────

def get_embeddings():
    """
    HuggingFace sentence-transformers embeddings — runs fully locally.
    Used for: RAG retrieval, semantic memory, vector search.

    Usage:
        from llm_provider import get_embeddings
        embeddings = get_embeddings()
        vector = embeddings.embed_query("billing issue")
    """
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_classifier():
    """
    HuggingFace zero-shot classification — runs fully locally.
    Used for: ticket pre-classification before LLM routing (zero tokens spent).

    Usage:
        from llm_provider import get_classifier
        clf = get_classifier()
        result = clf("My invoice is wrong", ["billing", "technical", "general"])
        label = result["labels"][0]  # → "billing"
    """
    from transformers import pipeline
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)


def get_reranker():
    """
    HuggingFace cross-encoder reranker — runs fully locally.
    Used for: RAG result reranking (retrieve top-20, rerank to top-5).

    Pattern: FAISS similarity search (fast, approximate recall)
             → cross-encoder rerank (slow, precise ordering)
             → top-5 to LLM

    Usage:
        from llm_provider import get_reranker
        reranker = get_reranker()
        scores = reranker.predict([("query text", "document text"), ...])
    """
    from sentence_transformers import CrossEncoder
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")


# ── RETRY DECORATOR ───────────────────────────────────────────────────────────

def with_retry(func):
    """
    Decorator: retries async functions up to 4 times with exponential backoff.
    Apply to EVERY node that calls an LLM to survive Groq free tier rate limits.

    IMPORTANT — functools.update_wrapper:
        LangGraph uses inspect.signature() to decide whether to pass `config`
        to a node. Without update_wrapper, the tenacity wrapper's signature is
        `(*args, **kwargs)` — no explicit `config` param — so LangGraph calls
        the node with only `(state,)`, causing TypeError at runtime.
        update_wrapper sets __wrapped__ = func, which causes inspect.signature
        to follow the chain and find the original (state, config) signature.

    Usage:
        from llm_provider import with_retry

        @with_retry
        async def my_node(state: AgentState, config: RunnableConfig) -> dict:
            llm = get_research_llm()
            ...
    """
    import functools

    from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

    # NodeInterrupt is how interrupt() pauses the graph — it must NEVER be retried.
    # If tenacity catches it and retries, the second attempt runs outside LangGraph's
    # Pregel context and raises "Called get_config outside of a runnable context".
    # We import lazily to avoid a hard dependency on langgraph.errors at module load.
    try:
        from langgraph.errors import NodeInterrupt as _NodeInterrupt
    except ImportError:
        _NodeInterrupt = None  # type: ignore[assignment]

    def _is_retryable(exc: BaseException) -> bool:
        """
        Returns True if tenacity should retry after this exception.

        NodeInterrupt must NEVER be retried — it is LangGraph's graph control signal,
        not a transient error. All other Exception subclasses are retried.

        WHY retry_if_exception and not retry=callable directly?
          tenacity's `retry=callable` passes a RetryCallState object, NOT the
          exception itself. retry_if_exception(predicate) is the correct API —
          it extracts the exception and passes it to the predicate.
        """
        if _NodeInterrupt is not None and isinstance(exc, _NodeInterrupt):
            return False
        return isinstance(exc, Exception)

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception(_is_retryable),
        reraise=True,
    )
    async def wrapper(*args, **kwargs):
        return await func(*args, **kwargs)

    # Preserve the original function's signature so LangGraph's inspect.signature
    # check sees (state: AgentState, config: RunnableConfig) and passes both args.
    functools.update_wrapper(wrapper, func)
    return wrapper


# ── PROVIDER METADATA ─────────────────────────────────────────────────────────

def get_provider_info() -> dict:
    """Returns metadata about the current provider for logging and debugging."""
    return {
        "provider": PROVIDER,
        "default_model": {
            "groq": "llama-3.3-70b-versatile",
            "huggingface": "HuggingFaceH4/zephyr-7b-beta",
            "hf_local": "microsoft/phi-3-mini-4k-instruct",
        }.get(PROVIDER, "unknown"),
        "tool_calling_supported": PROVIDER == "groq",
        "embeddings_model": "sentence-transformers/all-MiniLM-L6-v2 (local)",
        "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2 (local)",
    }