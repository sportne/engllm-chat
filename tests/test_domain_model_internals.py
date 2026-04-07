"""Focused tests for internal domain model groupings."""

from __future__ import annotations

from pathlib import Path

import pytest

from engllm_chat.domain._models.chat_protocol import ChatMessage, ChatToolResult
from engllm_chat.domain._models.common import DEFAULT_CHAT_API_KEY_ENV_VAR
from engllm_chat.domain._models.config import ChatLLMConfig
from engllm_chat.domain._models.responses import ChatFinalResponse, ChatTokenUsage


def test_common_runtime_defaults_use_shared_api_key_env_var() -> None:
    config = ChatLLMConfig(model_name="gemini-2.5-flash")
    assert DEFAULT_CHAT_API_KEY_ENV_VAR == "ENGLLM_CHAT_API_KEY"
    assert (
        config.credential_prompt_metadata().api_key_env_var
        == DEFAULT_CHAT_API_KEY_ENV_VAR
    )


def test_config_model_can_mark_mock_runs_as_not_requiring_credentials() -> None:
    config = ChatLLMConfig(model_name="mock-chat")
    metadata = config.credential_prompt_metadata(mock_mode=True)
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
