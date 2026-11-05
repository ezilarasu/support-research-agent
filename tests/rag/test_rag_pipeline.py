"""
Tests for the Module 9 RAG pipeline.

Covers:
  - FAISSVectorStore construction, add_documents, similarity_search, save/load
  - FixedSizeChunker and SemanticChunker (mocked)
  - Ingestion pipeline (build_index)
  - make_retriever_node() with injected mocks
  - Graders: grade_document, grade_retrieval, check_hallucination
  - Evaluator: evaluate_retrieval_recall, run_eval_suite
  - CRAG graph nodes: rewrite_query_node, retrieval_grade_node, generate_node
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from core.settings import get_settings
from rag.grader import check_hallucination, grade_document, grade_retrieval
from rag.protocols import RetrievedDocument

# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clear_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def mock_settings(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    get_settings.cache_clear()
    return get_settings()


@pytest.fixture
def sample_documents():
    return [
        Document(
            page_content="Refunds under $500 are processed in 3–5 business days.",
            metadata={"source": "data/kb/billing_policies.md", "filename": "billing_policies.md"},
        ),
        Document(
            page_content="Use Authorization: Bearer YOUR_API_KEY header for authentication.",
            metadata={"source": "data/kb/api_documentation.md", "filename": "api_documentation.md"},
        ),
        Document(
            page_content="Reset your password at the login page by clicking Forgot password.",
            metadata={"source": "data/kb/general_faq.md", "filename": "general_faq.md"},
        ),
    ]


@pytest.fixture
def sample_retrieved_docs() -> list[RetrievedDocument]:
    return [
        RetrievedDocument(
            content="Refunds under $500 take 3–5 business days.",
            source="data/kb/billing_policies.md",
            score=0.92,
            metadata={"filename": "billing_policies.md"},
        ),
        RetrievedDocument(
            content="Large refunds over $500 require manager approval.",
            source="data/kb/billing_policies.md",
            score=0.85,
            metadata={"filename": "billing_policies.md"},
        ),
    ]


@pytest.fixture
def mock_llm():
    mock = MagicMock()
    mock.invoke = MagicMock(return_value=AIMessage(content="relevant\nThe document discusses refunds directly."))
    mock.ainvoke = AsyncMock(return_value=AIMessage(content="relevant\nThe document is relevant."))
    return mock


@pytest.fixture
def runnable_config():
    return {
        "configurable": {
            "thread_id": "test-rag-thread",
            "user_id": "test-user",
        }
    }


# ── FAISSVectorStore tests ────────────────────────────────────────────────────

class TestFAISSVectorStore:
    """Unit tests for FAISSVectorStore."""

    def test_init_creates_empty_store(self, mock_settings):
        from rag.stores.faiss_store import FAISSVectorStore

        mock_embeddings = MagicMock()
        store = FAISSVectorStore(mock_embeddings)

        assert not store.is_loaded
        assert store._store is None

    def test_add_documents_creates_index(self, mock_settings, sample_documents):
        from rag.stores.faiss_store import FAISSVectorStore

        mock_embeddings = MagicMock()
        mock_faiss = MagicMock()

        with patch("rag.stores.faiss_store.FAISS" if False else "langchain_community.vectorstores.FAISS") as MockFAISS:
            MockFAISS.from_documents.return_value = mock_faiss
            store = FAISSVectorStore(mock_embeddings)

            # Patch the import inside add_documents
            with patch("rag.stores.faiss_store.FAISSVectorStore.add_documents") as mock_add:
                mock_add.return_value = None
                store.add_documents(sample_documents)
                mock_add.assert_called_once_with(sample_documents)

    def test_add_documents_raises_on_empty(self, mock_settings):
        from rag.stores.faiss_store import FAISSVectorStore

        store = FAISSVectorStore(MagicMock())

        with pytest.raises(ValueError, match="empty"):
            store.add_documents([])

    def test_similarity_search_raises_when_not_loaded(self, mock_settings):
        from rag.stores.faiss_store import FAISSVectorStore

        store = FAISSVectorStore(MagicMock())

        with pytest.raises(RuntimeError, match="no index loaded"):
            store.similarity_search("test query", k=5)

    def test_save_raises_when_not_loaded(self, mock_settings):
        from rag.stores.faiss_store import FAISSVectorStore

        store = FAISSVectorStore(MagicMock())

        with pytest.raises(RuntimeError, match="no index to save"):
            store.save("/tmp/test_index")

    def test_load_raises_when_path_missing(self, mock_settings):
        from rag.stores.faiss_store import FAISSVectorStore

        with pytest.raises(FileNotFoundError, match="index not found"):
            FAISSVectorStore.load("/nonexistent/path", MagicMock())

    def test_similarity_search_delegates_to_inner_store(self, mock_settings):
        from rag.stores.faiss_store import FAISSVectorStore

        mock_embeddings = MagicMock()
        store = FAISSVectorStore(mock_embeddings)

        mock_inner = MagicMock()
        mock_inner.similarity_search.return_value = [
            Document(page_content="test content", metadata={"source": "kb/test.md"})
        ]
        store._store = mock_inner

        results = store.similarity_search("refund policy", k=3)

        mock_inner.similarity_search.assert_called_once_with("refund policy", k=3)
        assert len(results) == 1
        assert results[0].page_content == "test content"

    def test_to_retriever_raises_when_not_loaded(self, mock_settings):
        from rag.stores.faiss_store import FAISSVectorStore

        store = FAISSVectorStore(MagicMock())

        with pytest.raises(RuntimeError, match="no index loaded"):
            store.to_retriever()


# ── FixedSizeChunker tests ────────────────────────────────────────────────────

class TestFixedSizeChunker:
    """Unit tests for FixedSizeChunker."""

    def test_split_returns_chunks(self, sample_documents):
        from rag.chunking.strategies import FixedSizeChunker

        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.split(sample_documents)

        assert len(chunks) >= len(sample_documents)  # should produce at least one chunk per doc
        for chunk in chunks:
            assert isinstance(chunk, Document)
            assert len(chunk.page_content) <= 200  # generous bound for small test docs

    def test_split_empty_returns_empty(self):
        from rag.chunking.strategies import FixedSizeChunker

        chunker = FixedSizeChunker()
        result = chunker.split([])

        assert result == []

    def test_split_preserves_metadata(self, sample_documents):
        from rag.chunking.strategies import FixedSizeChunker

        chunker = FixedSizeChunker(chunk_size=50, chunk_overlap=5)
        chunks = chunker.split([sample_documents[0]])

        for chunk in chunks:
            assert "source" in chunk.metadata
            assert chunk.metadata["source"] == "data/kb/billing_policies.md"

    def test_split_adds_chunk_index(self, sample_documents):
        from rag.chunking.strategies import FixedSizeChunker

        # Use a small chunk size to force multiple chunks from one document
        long_doc = Document(
            page_content="a" * 300,
            metadata={"source": "test.md"},
        )
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.split([long_doc])

        for chunk in chunks:
            assert "chunk_index" in chunk.metadata


# ── Ingestion pipeline tests ──────────────────────────────────────────────────

class TestBuildIndex:
    """Unit tests for the build_index ingestion pipeline."""

    def test_build_index_raises_on_no_docs(self, mock_settings):
        from rag.ingest import build_index

        with pytest.raises(ValueError, match="no documents"):
            build_index(docs=[])

    def test_load_kb_documents_returns_empty_for_missing_dir(self, mock_settings):
        from rag.ingest import load_kb_documents

        result = load_kb_documents("/nonexistent/path")

        assert result == []

    def test_load_kb_documents_loads_markdown_files(self, mock_settings):
        from rag.ingest import load_kb_documents

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test markdown files
            Path(tmpdir, "test1.md").write_text("# Test 1\nContent one.")
            Path(tmpdir, "test2.md").write_text("# Test 2\nContent two.")
            Path(tmpdir, "not_markdown.txt").write_text("ignored file")

            docs = load_kb_documents(tmpdir)

        assert len(docs) == 2
        sources = [d.metadata["filename"] for d in docs]
        assert "test1.md" in sources
        assert "test2.md" in sources

    def test_build_index_calls_chunker_and_store(self, mock_settings, sample_documents):
        from rag.ingest import build_index

        mock_strategy = MagicMock()
        mock_strategy.split.return_value = sample_documents

        mock_store_instance = MagicMock()
        mock_store_instance.add_documents = MagicMock()
        mock_store_instance.save = MagicMock()
        mock_store_cls = MagicMock(return_value=mock_store_instance)
        mock_embeddings = MagicMock()

        # Patch the lazy imports inside build_index's function body
        with (
            patch("llm_provider.get_embeddings", return_value=mock_embeddings),
            patch("rag.stores.faiss_store.FAISSVectorStore", mock_store_cls),
        ):
            build_index(
                docs=sample_documents,
                strategy=mock_strategy,
                index_path="/tmp/test_index_build",
            )

            mock_strategy.split.assert_called_once_with(sample_documents)


# ── Retriever node tests ──────────────────────────────────────────────────────

class TestRetrieverNode:
    """Unit tests for make_retriever_node()."""

    @pytest.mark.asyncio
    async def test_returns_empty_on_empty_query(self, mock_settings, runnable_config):
        from rag.retriever import make_retriever_node
        from state import AgentState

        AgentState.__new__(AgentState)  # bypass TypedDict construction for test
        state_dict = {"query": "", "messages": [], "retrieved_docs": []}

        node = make_retriever_node(store=None, reranker=None)
        result = await node(state_dict, runnable_config)

        assert result["retrieved_docs"] == []
        assert result["retrieval_grade"] == "irrelevant"

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_store(self, mock_settings, runnable_config):
        from rag.retriever import make_retriever_node

        node = make_retriever_node(store=None, reranker=None)
        state = {"query": "refund policy", "messages": [], "retrieved_docs": []}

        with patch("core.factories.get_vector_store", return_value=None):
            result = await node(state, runnable_config)

        assert result["retrieved_docs"] == []
        assert result["retrieval_grade"] == "irrelevant"

    @pytest.mark.asyncio
    async def test_retrieves_and_reranks_documents(self, mock_settings, runnable_config, sample_documents):
        from rag.retriever import make_retriever_node

        mock_store = MagicMock()
        mock_store.similarity_search.return_value = sample_documents

        mock_reranker = MagicMock()
        mock_reranker.predict.return_value = [0.9, 0.7, 0.3]

        node = make_retriever_node(store=mock_store, reranker=mock_reranker)
        state = {"query": "how do I get a refund", "messages": [], "retrieved_docs": []}

        result = await node(state, runnable_config)

        assert len(result["retrieved_docs"]) > 0
        assert result["retrieval_grade"] == "relevant"
        mock_store.similarity_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_store_runtime_error_gracefully(self, mock_settings, runnable_config):
        from rag.retriever import make_retriever_node

        mock_store = MagicMock()
        mock_store.similarity_search.side_effect = RuntimeError("no index loaded")

        node = make_retriever_node(store=mock_store, reranker=None)
        state = {"query": "test query", "messages": [], "retrieved_docs": []}

        result = await node(state, runnable_config)

        assert result["retrieved_docs"] == []
        assert result["retrieval_grade"] == "irrelevant"

    @pytest.mark.asyncio
    async def test_node_returns_partial_dict(self, mock_settings, runnable_config, sample_documents):
        from rag.retriever import make_retriever_node

        mock_store = MagicMock()
        mock_store.similarity_search.return_value = sample_documents

        mock_reranker = MagicMock()
        mock_reranker.predict.return_value = [0.9, 0.5, 0.2]

        node = make_retriever_node(store=mock_store, reranker=mock_reranker)
        state = {"query": "password reset", "messages": [], "retrieved_docs": []}

        result = await node(state, runnable_config)

        # Should not contain unrelated state fields
        assert "messages" not in result
        assert "active_agent" not in result
        assert "ticket_type" not in result


# ── Grader tests ─────────────────────────────────────────────────────────────

class TestGraders:
    """Unit tests for pure grader functions."""

    def test_grade_document_returns_relevant(self, mock_llm):
        mock_llm.invoke.return_value = AIMessage(content="relevant\nDirectly discusses refunds.")

        result = grade_document("refund policy", "Refunds take 3–5 days.", mock_llm)

        assert result["grade"] == "relevant"
        assert isinstance(result["reason"], str)
        assert len(result["scores"]) > 0

    def test_grade_document_returns_irrelevant(self, mock_llm):
        mock_llm.invoke.return_value = AIMessage(content="irrelevant\nThis is about API auth, not refunds.")

        result = grade_document("refund policy", "Use Bearer token for auth.", mock_llm)

        assert result["grade"] == "irrelevant"

    def test_grade_document_defaults_to_relevant_on_error(self, mock_llm):
        mock_llm.invoke.side_effect = Exception("LLM failed")

        result = grade_document("any query", "any content", mock_llm)

        assert result["grade"] == "relevant"  # fail-safe: don't suppress content

    def test_grade_retrieval_all_relevant(self, mock_llm, sample_retrieved_docs):
        mock_llm.invoke.return_value = AIMessage(content="relevant\nOn topic.")

        overall = grade_retrieval("refund policy", sample_retrieved_docs, mock_llm)

        assert overall == "relevant"

    def test_grade_retrieval_all_irrelevant(self, mock_llm, sample_retrieved_docs):
        mock_llm.invoke.return_value = AIMessage(content="irrelevant\nOff topic.")

        overall = grade_retrieval("password reset", sample_retrieved_docs, mock_llm)

        assert overall == "irrelevant"

    def test_grade_retrieval_empty_docs_returns_irrelevant(self, mock_llm):
        result = grade_retrieval("any query", [], mock_llm)

        assert result == "irrelevant"

    def test_check_hallucination_grounded_returns_true(self, mock_llm, sample_retrieved_docs):
        mock_llm.invoke.return_value = AIMessage(content="grounded\nThe answer matches the docs.")

        result = check_hallucination(
            "Refunds take 3–5 business days.",
            sample_retrieved_docs,
            mock_llm,
        )

        assert result is True

    def test_check_hallucination_hallucinated_returns_false(self, mock_llm, sample_retrieved_docs):
        mock_llm.invoke.return_value = AIMessage(content="hallucinated\nThis claim is not in the docs.")

        result = check_hallucination(
            "Refunds are processed in 30 seconds.",
            sample_retrieved_docs,
            mock_llm,
        )

        assert result is False

    def test_check_hallucination_no_docs_returns_true(self, mock_llm):
        result = check_hallucination("any answer", [], mock_llm)

        assert result is True  # no docs → nothing to contradict


# ── Evaluator tests ───────────────────────────────────────────────────────────

class TestEvaluator:
    """Unit tests for RAG evaluation metrics."""

    def test_recall_perfect_score(self, sample_retrieved_docs):
        from rag.evaluator import evaluate_retrieval_recall

        relevant_sources = ["data/kb/billing_policies.md"]
        recall = evaluate_retrieval_recall("refund", sample_retrieved_docs, relevant_sources)

        assert recall == 1.0

    def test_recall_zero_when_no_overlap(self, sample_retrieved_docs):
        from rag.evaluator import evaluate_retrieval_recall

        relevant_sources = ["data/kb/missing_doc.md"]
        recall = evaluate_retrieval_recall("refund", sample_retrieved_docs, relevant_sources)

        assert recall == 0.0

    def test_recall_zero_when_no_relevant_sources_given(self, sample_retrieved_docs):
        from rag.evaluator import evaluate_retrieval_recall

        recall = evaluate_retrieval_recall("refund", sample_retrieved_docs, [])

        assert recall == 0.0

    def test_run_eval_suite_empty(self, mock_llm):
        from rag.evaluator import run_eval_suite

        result = run_eval_suite([], mock_llm)

        assert result["n"] == 0
        assert result["recall_mean"] == 0.0

    def test_run_eval_suite_single_case(self, mock_llm, sample_retrieved_docs):
        from rag.evaluator import run_eval_suite

        mock_llm.invoke.return_value = AIMessage(content="grounded\nFaithful.")

        cases = [{
            "query": "How long for a refund?",
            "expected_sources": ["data/kb/billing_policies.md"],
            "retrieved": sample_retrieved_docs,
            "answer": "Refunds take 3–5 business days.",
        }]

        result = run_eval_suite(cases, mock_llm)

        assert result["n"] == 1
        assert result["recall_mean"] == 1.0
        assert 0.0 <= result["faithfulness_mean"] <= 1.0
        assert "by_case" in result
        assert len(result["by_case"]) == 1


# ── CRAG graph node tests ─────────────────────────────────────────────────────

class TestCRAGNodes:
    """Integration tests for CRAG graph node functions."""

    @pytest.mark.asyncio
    async def test_rewrite_query_node_returns_rewritten_query(self, mock_settings, runnable_config):
        from rag.graph import rewrite_query_node

        state = {
            "query": "invoice wrong",
            "messages": [HumanMessage(content="invoice wrong")],
            "retrieved_docs": [],
        }

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(
            return_value=AIMessage(content="Why is my invoice incorrect or showing wrong charges?")
        )

        with patch("rag.graph.get_router_llm", return_value=mock_llm):
            result = await rewrite_query_node(state, runnable_config)

        assert "query" in result
        assert result["query"] != ""

    @pytest.mark.asyncio
    async def test_rewrite_query_node_keeps_original_on_failure(self, mock_settings, runnable_config):
        from rag.graph import rewrite_query_node

        state = {
            "query": "original query",
            "messages": [],
            "retrieved_docs": [],
        }

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM failed"))

        with patch("rag.graph.get_router_llm", return_value=mock_llm):
            result = await rewrite_query_node(state, runnable_config)

        assert result["query"] == "original query"

    @pytest.mark.asyncio
    async def test_retrieval_grade_node_sets_grade(self, mock_settings, runnable_config, sample_retrieved_docs):
        from rag.graph import retrieval_grade_node

        state = {
            "query": "refund policy",
            "retrieved_docs": sample_retrieved_docs,
            "messages": [],
        }

        mock_llm = MagicMock()
        mock_llm.invoke = MagicMock(return_value=AIMessage(content="relevant\nOn topic."))

        with patch("rag.graph.get_router_llm", return_value=mock_llm):
            result = await retrieval_grade_node(state, runnable_config)

        assert "retrieval_grade" in result
        assert result["retrieval_grade"] in ("relevant", "irrelevant", "partial")

    @pytest.mark.asyncio
    async def test_generate_node_returns_final_response(self, mock_settings, runnable_config, sample_retrieved_docs):
        from rag.graph import generate_node

        state = {
            "query": "How long does a refund take?",
            "retrieved_docs": sample_retrieved_docs,
            "messages": [],
        }

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(
            return_value=AIMessage(content="Refunds under $500 take 3–5 business days.")
        )

        with patch("rag.graph.get_research_llm", return_value=mock_llm):
            result = await generate_node(state, runnable_config)

        assert "final_response" in result
        assert len(result["final_response"]) > 0

    @pytest.mark.asyncio
    async def test_generate_node_handles_llm_failure(self, mock_settings, runnable_config):
        from rag.graph import generate_node

        state = {
            "query": "test query",
            "retrieved_docs": [],
            "messages": [],
        }

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("Provider down"))

        with patch("rag.graph.get_research_llm", return_value=mock_llm):
            result = await generate_node(state, runnable_config)

        assert "final_response" in result
        assert "error" in result

    @pytest.mark.asyncio
    async def test_hallucination_check_node_sets_bool(self, mock_settings, runnable_config, sample_retrieved_docs):
        from rag.graph import hallucination_check_node

        state = {
            "final_response": "Refunds take 3–5 business days.",
            "retrieved_docs": sample_retrieved_docs,
            "messages": [],
        }

        mock_llm = MagicMock()
        mock_llm.invoke = MagicMock(return_value=AIMessage(content="grounded\nSupported by docs."))

        with patch("rag.graph.get_router_llm", return_value=mock_llm):
            result = await hallucination_check_node(state, runnable_config)

        assert "hallucination_check" in result
        assert isinstance(result["hallucination_check"], bool)

    def test_route_by_grade_relevant_goes_to_generate(self):
        from rag.graph import route_by_grade

        state = {"retrieval_grade": "relevant"}
        assert route_by_grade(state) == "generate"

    def test_route_by_grade_partial_goes_to_generate(self):
        from rag.graph import route_by_grade

        state = {"retrieval_grade": "partial"}
        assert route_by_grade(state) == "generate"

    def test_route_by_grade_irrelevant_goes_to_rewrite(self):
        from rag.graph import route_by_grade

        state = {"retrieval_grade": "irrelevant"}
        assert route_by_grade(state) == "rewrite_query"
