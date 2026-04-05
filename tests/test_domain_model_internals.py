"""Focused tests for internal domain model groupings."""

from __future__ import annotations

from pathlib import Path

import pytest

from engllm_chat.domain._models.chat_protocol import ChatMessage, ChatToolResult
from engllm_chat.domain._models.common import (
    _PROVIDER_DEFAULT_API_BASE_URLS,
    _PROVIDER_DEFAULT_API_KEY_ENV_VARS,
)
from engllm_chat.domain._models.config import ChatLLMConfig
from engllm_chat.domain._models.responses import ChatFinalResponse, ChatTokenUsage


def test_common_provider_defaults_are_reflected_by_chat_llm_config() -> None:
    assert _PROVIDER_DEFAULT_API_BASE_URLS["gemini"] is not None
    assert _PROVIDER_DEFAULT_API_KEY_ENV_VARS["openai"] == "OPENAI_API_KEY"

    config = ChatLLMConfig(provider="gemini", model_name="gemini-2.5-flash")
    assert config.resolved_api_base_url() == _PROVIDER_DEFAULT_API_BASE_URLS["gemini"]
    assert (
        config.resolved_api_key_env_var()
        == _PROVIDER_DEFAULT_API_KEY_ENV_VARS["gemini"]
    )


def test_config_model_retains_legacy_ollama_base_url_and_prompt_metadata() -> None:
    config = ChatLLMConfig(
        provider="ollama",
        model_name="qwen",
        ollama_base_url="http://localhost:11434",  # type: ignore[call-arg]
    )

    metadata = config.credential_prompt_metadata()
    assert config.api_base_url == "http://localhost:11434"
    assert metadata.provider == "ollama"
    assert metadata.expects_api_key is False


def test_chat_protocol_model_retains_tool_result_and_message_validation() -> None:
    with pytest.raises(
        ValueError,
        match="error_message is required when tool result status=error",
    ):
        ChatToolResult(
            call_id="call-1",
            tool_name="read_file",
            status="error",
            payload={},
        )

    with pytest.raises(ValueError, match="tool messages must include a tool result"):
        ChatMessage(role="tool")


def test_response_models_retain_validation_and_token_total_derivation() -> None:
    usage = ChatTokenUsage(input_tokens=3, output_tokens=4)
    assert usage.total_tokens == 7

    with pytest.raises(
        ValueError, match="final response list entries must not be empty"
    ):
        ChatFinalResponse(answer="ok", missing_information=[" "])

    citation_response = ChatFinalResponse(
        answer="Done",
        citations=[{"source_path": Path("src/app.py")}],
    )
    assert citation_response.citations[0].source_path == Path("src/app.py")
