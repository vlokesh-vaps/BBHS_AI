"""Dedicated unit tests for pluggable LLM and embedding providers."""

from langchain_core.messages import HumanMessage

from src.providers.embeddings import (
    BaseEmbeddingProvider,
    MockEmbeddingProvider,
    check_embedding_reachability,
    get_embedding_provider,
)
from src.providers.llm import (
    BaseLLMProvider,
    MockLLMProvider,
    get_llm_provider,
)


def test_mock_llm_provider():
    provider = MockLLMProvider("Grounded test answer")
    assert isinstance(provider, BaseLLMProvider)
    assert provider.model_name == "mock-llm"

    response = provider.invoke([HumanMessage(content="Hello")])
    assert response.content == "Grounded test answer"


def test_mock_embedding_provider():
    provider = MockEmbeddingProvider(dimension=64)
    assert isinstance(provider, BaseEmbeddingProvider)
    assert provider.dimension == 64
    assert provider.is_available()

    vec = provider.embed_query("test query")
    assert len(vec) == 64
    assert isinstance(vec[0], float)

    batch = provider.embed_documents(["first", "second"])
    assert len(batch) == 2
    assert len(batch[0]) == 64


def test_provider_factory():
    llm = get_llm_provider("mock")
    assert isinstance(llm, MockLLMProvider)

    embedder = get_embedding_provider("mock")
    assert isinstance(embedder, MockEmbeddingProvider)


def test_check_embedding_reachability_with_invalid_host():
    reachable, msg = check_embedding_reachability("http://127.0.0.1:59999", timeout_seconds=0.5)
    assert not reachable
    assert "unreachable" in msg.lower() or "error" in msg.lower()

