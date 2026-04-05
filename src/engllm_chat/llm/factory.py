"""Chat LLM provider factory."""

from __future__ import annotations

from engllm_chat.domain.errors import LLMError
from engllm_chat.domain.models import ChatLLMConfig, ChatProvider
from engllm_chat.llm.base import ChatLLMClient
from engllm_chat.llm.mock import MockLLMClient
from engllm_chat.llm.openai_compatible import (
    OpenAICompatibleChatLLMClient,
    normalize_ollama_base_url,
)


def create_chat_llm_client(
    config: ChatLLMConfig,
    provider: ChatProvider | None = None,
    model_name: str | None = None,
    api_base_url: str | None = None,
    timeout_seconds: float | None = None,
    api_key: str | None = None,
    verbose_llm_logging: bool | None = None,
) -> ChatLLMClient:
    """Create a chat-capable provider client from chat config and overrides."""

    resolved_provider = provider or config.provider
    resolved_model = model_name or config.model_name
    resolved_timeout = timeout_seconds or config.timeout_seconds
    resolved_verbose_logging = (
        config.verbose_llm_logging
        if verbose_llm_logging is None
        else verbose_llm_logging
    )

    if resolved_provider == "mock":
        return MockLLMClient(model_name=resolved_model)

    if resolved_provider == "ollama":
        resolved_base_url = api_base_url or config.resolved_api_base_url()
        return OpenAICompatibleChatLLMClient(
            model_name=resolved_model,
            provider_name="ollama",
            api_key_env_var=None,
            base_url=normalize_ollama_base_url(resolved_base_url),
            timeout_seconds=resolved_timeout,
            api_key=api_key,
            verbose_logging=resolved_verbose_logging,
        )

    if resolved_provider in {"openai", "xai", "anthropic", "gemini"}:
        resolved_api_base_url = api_base_url or config.resolved_api_base_url()
        return OpenAICompatibleChatLLMClient(
            model_name=resolved_model,
            provider_name=resolved_provider,
            api_key_env_var=config.resolved_api_key_env_var(),
            base_url=resolved_api_base_url,
            timeout_seconds=resolved_timeout,
            api_key=api_key,
            verbose_logging=resolved_verbose_logging,
        )

    raise LLMError(f"Unsupported chat LLM provider '{resolved_provider}'")
