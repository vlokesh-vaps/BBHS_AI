"""Pluggable LLM provider protocol and implementations."""

import os
from typing import Any, Protocol, runtime_checkable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq

from src.config import CHAT_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE
from src.utils.logging import log_error, log_step


@runtime_checkable
class BaseLLMProvider(Protocol):
    """Protocol for interchangeable LLM inference engines."""

    @property
    def model_name(self) -> str:
        """Active model identifier."""
        ...

    def invoke(self, messages: Any, **kwargs: Any) -> Any:
        """Synchronously execute inference."""
        ...

    async def ainvoke(self, messages: Any, **kwargs: Any) -> Any:
        """Asynchronously execute inference."""
        ...

    @property
    def langchain_chat_model(self) -> BaseChatModel:
        """Underlying LangChain chat model for chain composition."""
        ...


class GroqProvider:
    """Primary production LLM provider using Groq Cloud API."""

    def __init__(
        self,
        model_name: str = CHAT_MODEL,
        api_key: str | None = None,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
        timeout: int = 60,
    ) -> None:
        self._model_name = model_name
        self._api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout

        self._llm = ChatGroq(
            groq_api_key=self._api_key,
            model=self._model_name,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            timeout=self._timeout,
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def langchain_chat_model(self) -> BaseChatModel:
        return self._llm

    def invoke(self, messages: Any, **kwargs: Any) -> Any:
        return self._llm.invoke(messages, **kwargs)

    async def ainvoke(self, messages: Any, **kwargs: Any) -> Any:
        return await self._llm.ainvoke(messages, **kwargs)


class OpenAIProvider:
    """OpenAI or compatible provider adapter."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: str | None = None,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
        timeout: int = 60,
        base_url: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._base_url = base_url

        try:
            from langchain_openai import ChatOpenAI

            self._llm = ChatOpenAI(
                model=self._model_name,
                api_key=self._api_key,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                timeout=self._timeout,
                base_url=self._base_url,
            )
        except ImportError:
            raise ImportError("langchain-openai is required to use OpenAIProvider. Install with `pip install langchain-openai`.")

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def langchain_chat_model(self) -> BaseChatModel:
        return self._llm

    def invoke(self, messages: Any, **kwargs: Any) -> Any:
        return self._llm.invoke(messages, **kwargs)

    async def ainvoke(self, messages: Any, **kwargs: Any) -> Any:
        return await self._llm.ainvoke(messages, **kwargs)


class OllamaLLMProvider:
    """Local Ollama LLM provider adapter."""

    def __init__(
        self,
        model_name: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        temperature: float = LLM_TEMPERATURE,
        timeout: int = 60,
    ) -> None:
        self._model_name = model_name
        self._base_url = base_url
        self._temperature = temperature
        self._timeout = timeout

        from langchain_ollama import ChatOllama

        self._llm = ChatOllama(
            model=self._model_name,
            base_url=self._base_url,
            temperature=self._temperature,
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def langchain_chat_model(self) -> BaseChatModel:
        return self._llm

    def invoke(self, messages: Any, **kwargs: Any) -> Any:
        return self._llm.invoke(messages, **kwargs)

    async def ainvoke(self, messages: Any, **kwargs: Any) -> Any:
        return await self._llm.ainvoke(messages, **kwargs)


class MockLLMProvider:
    """Mock LLM provider returning deterministic responses for offline unit tests."""

    def __init__(self, response_text: str = "Grounded response from test assistant.") -> None:
        self._model_name = "mock-llm"
        self._response_text = response_text

        from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
        from langchain_core.messages import AIMessage

        self._llm = FakeMessagesListChatModel(responses=[AIMessage(content=self._response_text)])

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def langchain_chat_model(self) -> BaseChatModel:
        return self._llm

    def invoke(self, messages: Any, **kwargs: Any) -> Any:
        from langchain_core.messages import AIMessage
        return AIMessage(content=self._response_text)

    async def ainvoke(self, messages: Any, **kwargs: Any) -> Any:
        from langchain_core.messages import AIMessage
        return AIMessage(content=self._response_text)


def get_llm_provider(
    provider_name: str = "groq",
    model_name: str = CHAT_MODEL,
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int = LLM_MAX_TOKENS,
    timeout: int = 60,
    api_key: str | None = None,
) -> BaseLLMProvider:
    """Factory function for LLM providers."""
    normalized = provider_name.lower().strip()
    if normalized == "mock":
        return MockLLMProvider()
    if normalized == "openai":
        return OpenAIProvider(model_name=model_name, temperature=temperature, max_tokens=max_tokens, timeout=timeout, api_key=api_key)
    if normalized == "ollama":
        return OllamaLLMProvider(model_name=model_name, temperature=temperature, timeout=timeout)
    # Default to Groq
    return GroqProvider(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        api_key=api_key,
    )

