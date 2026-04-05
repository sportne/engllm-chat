"""Focused tests for the internal chat-tool registry."""

from __future__ import annotations

from pathlib import Path

from engllm_chat.domain.models import ChatConfig, ChatToolCall
from engllm_chat.tools.chat.registry import (
    build_chat_tool_definitions,
    execute_chat_tool_call,
    get_chat_tool_spec,
    get_chat_tool_specs,
)


def test_chat_tool_registry_exposes_expected_order_and_metadata() -> None:
    specs = get_chat_tool_specs()

    assert [spec.name for spec in specs] == [
        "list_directory",
        "list_directory_recursive",
        "find_files",
        "search_text",
        "get_file_info",
        "read_file",
    ]
    assert specs[0].description == "List immediate children of one directory."
    assert get_chat_tool_spec("search_text") is not None
    assert get_chat_tool_spec("missing_tool") is None


def test_build_chat_tool_definitions_match_registry_order_and_schemas() -> None:
    definitions = build_chat_tool_definitions()

    assert [definition.name for definition in definitions] == [
        "list_directory",
        "list_directory_recursive",
        "find_files",
        "search_text",
        "get_file_info",
        "read_file",
    ]
    assert definitions[0].description == "List immediate children of one directory."
    assert definitions[0].input_schema["type"] == "object"
    assert "path" in definitions[0].input_schema["properties"]
    assert "pattern" in definitions[2].input_schema["properties"]


def test_execute_chat_tool_call_runs_tool_and_wraps_success_payload(
    tmp_path: Path,
) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('hi')\n", encoding="utf-8")

    result = execute_chat_tool_call(
        ChatToolCall(
            call_id="call-1",
            tool_name="find_files",
            arguments={"path": "src", "pattern": "**/*.py"},
        ),
        root_path=tmp_path,
        config=ChatConfig(),
    )

    assert result.status == "ok"
    assert result.error_message is None
    assert result.payload["pattern"] == "**/*.py"
    assert result.payload["matches"][0]["path"] == "src/app.py"


def test_execute_chat_tool_call_wraps_unknown_tool_and_validation_errors(
    tmp_path: Path,
) -> None:
    unknown_result = execute_chat_tool_call(
        ChatToolCall(call_id="call-1", tool_name="missing_tool", arguments={}),
        root_path=tmp_path,
        config=ChatConfig(),
    )
    invalid_result = execute_chat_tool_call(
        ChatToolCall(
            call_id="call-2",
            tool_name="read_file",
            arguments={"path": "   "},
        ),
        root_path=tmp_path,
        config=ChatConfig(),
    )

    assert unknown_result.status == "error"
    assert "Unknown chat tool 'missing_tool'" == unknown_result.error_message
    assert invalid_result.status == "error"
    assert invalid_result.error_message is not None
    assert "path" in invalid_result.error_message
