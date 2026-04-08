"""Tests for standalone chat config loading and chat domain models."""

from __future__ import annotations

from pathlib import Path

import pytest

from engllm_chat.config import load_chat_config
from engllm_chat.domain.errors import ConfigError, ValidationError
from engllm_chat.domain.models import (
    ChatCitation,
    ChatFinalResponse,
    ChatLLMConfig,
    ChatMessage,
    ChatSourceFilters,
    ChatTokenUsage,
    ChatToolCall,
    ChatToolLimits,
    ChatToolResult,
)


def test_load_chat_config_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "chat.yaml"
    config_path.write_text("{}", encoding="utf-8")

    config = load_chat_config(config_path)

    assert config.llm.model_name == "qwen2.5-coder:7b-instruct-q4_K_M"
    assert config.llm.api_base_url is None
    assert config.llm.verbose_llm_logging is False
    assert config.llm.use_beta_parse is True
    assert config.source_filters.include == []
    assert config.source_filters.exclude == []
    assert config.source_filters.include_hidden is False
    assert config.session.max_context_tokens == 24000
    assert config.tool_limits.max_file_size_characters == 262144
    assert config.tool_limits.max_read_file_chars is None
    assert config.tool_limits.max_tool_result_chars == 24000
    assert config.ui.show_token_usage is True
    assert config.ui.show_footer_help is True


def test_load_chat_config_preserves_credential_prompt_metadata_without_secret(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "chat.yaml"
    config_path.write_text(
        """
llm:
  model_name: mock-chat
  temperature: 0.2
  timeout_seconds: 30.0
  prompt_for_api_key_if_missing: false
session:
  max_context_tokens: 1234
""".strip(),
        encoding="utf-8",
    )

    config = load_chat_config(config_path)
    metadata = config.llm.credential_prompt_metadata(mock_mode=True)

    assert config.session.max_context_tokens == 1234
    assert metadata.api_key_env_var == "ENGLLM_CHAT_API_KEY"
    assert metadata.prompt_for_api_key_if_missing is False
    assert metadata.expects_api_key is False
    assert metadata.mask_input is True
    assert metadata.allow_empty_secret is True
    assert metadata.persist_secret is False
    assert "api_key" not in config.llm.model_dump()


def test_chat_llm_config_uses_shared_credential_prompt_metadata() -> None:
    config = ChatLLMConfig(model_name="hosted-model")

    metadata = config.credential_prompt_metadata()
    assert metadata.api_key_env_var == "ENGLLM_CHAT_API_KEY"
    assert metadata.expects_api_key is True


def test_chat_llm_config_keeps_explicit_api_base_url_and_logging_flag() -> None:
    config = ChatLLMConfig(
        model_name="grok-4",
        api_base_url="https://example.test/v1",
        verbose_llm_logging=True,
    )

    assert config.api_base_url == "https://example.test/v1"
    assert config.credential_prompt_metadata().api_key_env_var == "ENGLLM_CHAT_API_KEY"
    assert config.verbose_llm_logging is True


def test_chat_llm_config_accepts_use_beta_parse_false() -> None:
    config = ChatLLMConfig(
        model_name="qwen",
        use_beta_parse=False,
    )

    assert config.use_beta_parse is False


def test_load_chat_config_rejects_validation_errors(tmp_path: Path) -> None:
    config_path = tmp_path / "chat.yaml"
    config_path.write_text(
        """
source_filters:
  include: ["   "]
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(
        ValidationError, match="source filter patterns must not be empty"
    ):
        load_chat_config(config_path)


def test_load_chat_config_rejects_non_mapping_sections(tmp_path: Path) -> None:
    config_path = tmp_path / "chat.yaml"
    config_path.write_text(
        """
llm:
  - not
  - a
  - mapping
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="chat config 'llm' must be a mapping"):
        load_chat_config(config_path)


def test_load_chat_config_rejects_invalid_limits(tmp_path: Path) -> None:
    config_path = tmp_path / "chat.yaml"
    config_path.write_text(
        """
session:
  max_tool_round_trips: 0
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(
        ValidationError, match="chat session limits must be positive integers"
    ):
        load_chat_config(config_path)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"model_name": "   "}, "chat LLM string settings must not be empty"),
        (
            {"api_base_url": "   "},
            "api_base_url must not be empty when provided",
        ),
        ({"temperature": -0.1}, "temperature must be between 0.0 and 1.0"),
        ({"timeout_seconds": 0}, "timeout_seconds must be greater than 0"),
    ],
)
def test_chat_llm_config_rejects_invalid_values(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ChatLLMConfig(**kwargs)


def test_chat_source_filters_strip_patterns() -> None:
    filters = ChatSourceFilters(include=[" *.py "], exclude=[" build/** "])

    assert filters.include == ["*.py"]
    assert filters.exclude == ["build/**"]


def test_chat_tool_limits_reject_non_positive_values() -> None:
    with pytest.raises(ValueError, match="chat tool limits must be positive integers"):
        ChatToolLimits(max_read_lines=0)

    with pytest.raises(ValueError, match="chat tool limits must be positive integers"):
        ChatToolLimits(max_file_size_characters=0)

    with pytest.raises(ValueError, match="chat tool limits must be positive integers"):
        ChatToolLimits(max_read_file_chars=0)


def test_chat_message_tool_round_trip_validates() -> None:
    tool_call = ChatToolCall(
        call_id="call-1",
        tool_name="list_directory",
        arguments={"path": "."},
    )
    assistant_message = ChatMessage(role="assistant", tool_calls=[tool_call])
    tool_message = ChatMessage(
        role="tool",
        tool_result=ChatToolResult(
            call_id="call-1",
            tool_name="list_directory",
            payload={"entries": ["src", "tests"]},
        ),
    )

    assert assistant_message.tool_calls[0].tool_name == "list_directory"
    assert tool_message.tool_result is not None
    assert tool_message.tool_result.payload["entries"] == ["src", "tests"]


def test_chat_tool_identifiers_reject_blank_values() -> None:
    with pytest.raises(ValueError, match="tool call identifiers must not be empty"):
        ChatToolCall(call_id=" ", tool_name="read_file")

    with pytest.raises(ValueError, match="tool result identifiers must not be empty"):
        ChatToolResult(call_id="call-1", tool_name=" ", payload={})


def test_chat_tool_result_rejects_invalid_error_field_combinations() -> None:
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

    with pytest.raises(
        ValueError,
        match="error_message is only allowed when tool result status=error",
    ):
        ChatToolResult(
            call_id="call-1",
            tool_name="read_file",
            status="ok",
            payload={},
            error_message="boom",
        )


def test_chat_message_rejects_empty_assistant_message() -> None:
    with pytest.raises(
        ValueError,
        match="assistant messages must include content or at least one tool call",
    ):
        ChatMessage(role="assistant")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"role": "system"}, "system messages must include content"),
        ({"role": "user"}, "user messages must include content"),
        (
            {
                "role": "user",
                "content": "hi",
                "tool_calls": [
                    ChatToolCall(call_id="call-1", tool_name="list_directory")
                ],
            },
            "user messages cannot include tool calls",
        ),
        (
            {
                "role": "user",
                "content": "hi",
                "tool_result": ChatToolResult(
                    call_id="call-1",
                    tool_name="list_directory",
                    payload={},
                ),
            },
            "user messages cannot include tool results",
        ),
        (
            {
                "role": "assistant",
                "content": "done",
                "tool_result": ChatToolResult(
                    call_id="call-1",
                    tool_name="list_directory",
                    payload={},
                ),
            },
            "assistant messages cannot include tool results",
        ),
        (
            {
                "role": "tool",
                "tool_calls": [
                    ChatToolCall(call_id="call-1", tool_name="list_directory")
                ],
                "tool_result": ChatToolResult(
                    call_id="call-1",
                    tool_name="list_directory",
                    payload={},
                ),
            },
            "tool messages cannot include tool calls",
        ),
        ({"role": "tool"}, "tool messages must include a tool result"),
    ],
)
def test_chat_message_rejects_invalid_role_shapes(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ChatMessage(**kwargs)


def test_chat_citation_rejects_inverted_line_range() -> None:
    with pytest.raises(
        ValueError, match="line_end must be greater than or equal to line_start"
    ):
        ChatCitation(source_path=Path("src/app.py"), line_start=5, line_end=4)


def test_chat_final_response_requires_non_empty_answer() -> None:
    with pytest.raises(ValueError, match="answer must not be empty"):
        ChatFinalResponse(answer="   ")


def test_chat_final_response_rejects_blank_list_entries() -> None:
    with pytest.raises(
        ValueError, match="final response list entries must not be empty"
    ):
        ChatFinalResponse(answer="ok", missing_information=[" "])


def test_chat_token_usage_derives_total_tokens() -> None:
    usage = ChatTokenUsage(input_tokens=12, output_tokens=5)

    assert usage.total_tokens == 17


def test_chat_token_usage_rejects_total_below_sum() -> None:
    with pytest.raises(
        ValueError,
        match="total_tokens cannot be less than input_tokens \\+ output_tokens",
    ):
        ChatTokenUsage(input_tokens=12, output_tokens=5, total_tokens=10)
