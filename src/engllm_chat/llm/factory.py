"""Chat LLM client factory."""

from __future__ import annotations

from engllm_chat.domain.errors import LLMError
from engllm_chat.domain.models import ChatLLMConfig
from engllm_chat.llm.base import ChatLLMClient
from engllm_chat.llm.mock import MockLLMClient
from engllm_chat.llm.openai_compatible import OpenAICompatibleChatLLMClient

_DEFAULT_API_KEY_ENV_VAR = "ENGLLM_CHAT_API_KEY"


def create_chat_llm_client(
    config: ChatLLMConfig,
    *,
    use_mock: bool = False,
    model_name: str | None = None,
    api_base_url: str | None = None,
    timeout_seconds: float | None = None,
    api_key: str | None = None,
    verbose_llm_logging: bool | None = None,
) -> ChatLLMClient:
    """Create a chat-capable client from config and overrides."""

    resolved_model = model_name or config.model_name
    resolved_timeout = timeout_seconds or config.timeout_seconds
    resolved_verbose_logging = (
        config.verbose_llm_logging
        if verbose_llm_logging is None
        else verbose_llm_logging
    )

    if use_mock:
        return MockLLMClient(model_name=resolved_model)

    resolved_api_base_url = api_base_url or config.api_base_url
    if resolved_api_base_url is None:
        raise LLMError(
            "api_base_url is required for non-mock chat runs. "
            "Set llm.api_base_url in the config or pass --api-base-url."
        )

    return OpenAICompatibleChatLLMClient(
        model_name=resolved_model,
        provider_name="openai-compatible",
        api_key_env_var=_DEFAULT_API_KEY_ENV_VAR,
        base_url=resolved_api_base_url,
        timeout_seconds=resolved_timeout,
        api_key=api_key,
        verbose_logging=resolved_verbose_logging,
        use_beta_parse=config.use_beta_parse,
    )
