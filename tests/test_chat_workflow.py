"""Prompt-builder and workflow tests for the interactive chat tool."""

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

import pytest

from engllm_chat.core.tokenize import tokenize
from engllm_chat.domain.models import (
    ChatConfig,
    ChatFinalResponse,
    ChatMessage,
    ChatTokenUsage,
    ChatToolCall,
    ChatToolLimits,
)
from engllm_chat.llm.base import (
    ChatTurnRequest,
    ChatTurnResponse,
)
from engllm_chat.llm.mock import MockLLMClient
from engllm_chat.prompts import __all__ as prompt_namespaces
from engllm_chat.prompts.chat import build_chat_system_prompt
from engllm_chat.tools.chat.models import (
    ChatSessionState,
    ChatSessionTurnRecord,
    ChatWorkflowResultEvent,
    ChatWorkflowStatusEvent,
    ChatWorkflowTurnResult,
    FindFilesArgs,
    ReadFileArgs,
    SearchTextArgs,
)
from engllm_chat.tools.chat.workflow import (
    run_chat_session_turn,
    run_chat_turn,
    run_streaming_chat_session_turn,
)


def _write(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _estimate_message_tokens(message: ChatMessage) -> int:
    token_count = 0
    if message.content:
        token_count += len(tokenize(message.content))
    if message.tool_calls:
        token_count += len(
            tokenize(
                json.dumps(
                    [
                        tool_call.model_dump(mode="json")
                        for tool_call in message.tool_calls
                    ],
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        )
    if message.tool_result is not None:
        token_count += len(
            tokenize(
                json.dumps(
                    message.tool_result.model_dump(mode="json"),
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        )
    return token_count


def _estimate_messages_tokens(messages: list[ChatMessage]) -> int:
    return sum(_estimate_message_tokens(message) for message in messages)


class _RecordingChatClient:
    def __init__(self, responses: list[ChatTurnResponse]) -> None:
        self._responses = list(responses)
        self.requests: list[ChatTurnRequest] = []

    def generate_chat_turn(self, request: ChatTurnRequest) -> ChatTurnResponse:
        self.requests.append(
            ChatTurnRequest(
                messages=list(request.messages),
                response_model=request.response_model,
                model_name=request.model_name,
                tools=list(request.tools),
                temperature=request.temperature,
            )
        )
        if not self._responses:
            raise AssertionError("No canned chat responses remain")
        return self._responses.pop(0)


def test_build_chat_system_prompt_includes_tool_catalog_and_examples() -> None:
    prompt = build_chat_system_prompt(tool_limits=ChatToolLimits())

    for tool_name in (
        "list_directory",
        "list_directory_recursive",
        "find_files",
        "search_text",
        "get_file_info",
        "read_file",
    ):
        assert tool_name in prompt

    assert "find_files(path='src', pattern='**/*.py')" in prompt
    assert "get_file_info(path='src/app.py')" in prompt
    assert "read_file(path='src/app.py', start_char=0, end_char=4000)" in prompt


def test_build_chat_system_prompt_mentions_ground_rules_and_final_response() -> None:
    prompt = build_chat_system_prompt(tool_limits=ChatToolLimits())

    assert "All tool paths are relative to the configured root." in prompt
    assert "Tool outputs are authoritative over guesses." in prompt
    assert "Answer conservatively and cite the evidence you actually have." in prompt

    for field_name in (
        "answer",
        "citations",
        "confidence",
        "uncertainty",
        "missing_information",
        "follow_up_suggestions",
    ):
        assert f"- {field_name}:" in prompt


def test_build_chat_system_prompt_is_deterministic_and_includes_limits() -> None:
    limits = ChatToolLimits(
        max_search_matches=12,
        max_file_size_characters=3456,
        max_read_file_chars=789,
        max_tool_result_chars=4321,
    )

    first = build_chat_system_prompt(
        tool_limits=limits,
        response_model=ChatFinalResponse,
    )
    second = build_chat_system_prompt(
        tool_limits=limits,
        response_model=ChatFinalResponse,
    )

    assert first == second
    assert "12 matches per call" in first
    assert "3456 characters" in first
    assert "max_read_file_chars" in first
    assert "4321 characters" in first


def test_chat_prompt_exports_are_available() -> None:
    assert "chat" in prompt_namespaces
    assert callable(build_chat_system_prompt)


def test_run_chat_turn_returns_single_turn_final_response(tmp_path: Path) -> None:
    client = MockLLMClient(
        chat_canned_turns=[
            ChatTurnResponse(
                assistant_message=ChatMessage(
                    role="assistant", content='{"answer":"Done"}'
                ),
                final_response=ChatFinalResponse(answer="Done"),
                token_usage=ChatTokenUsage(input_tokens=10, output_tokens=5),
                raw_text='{"answer":"Done"}',
                finish_reason="final_response",
            )
        ]
    )

    result = run_chat_turn(
        user_message="What is here?",
        prior_messages=[],
        root_path=tmp_path,
        config=ChatConfig(),
        llm_client=client,
    )

    assert result.status == "completed"
    assert result.final_response is not None
    assert result.final_response.answer == "Done"
    assert [message.role for message in result.new_messages] == ["user", "assistant"]
    assert result.token_usage is not None
    assert result.token_usage.total_tokens == 15
    assert result.tool_results == []


def test_run_chat_turn_executes_one_tool_round_then_returns_final_response(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "src" / "app.py", "print('hi')\n")
    client = MockLLMClient(
        chat_canned_turns=[
            ChatTurnResponse(
                assistant_message=ChatMessage(
                    role="assistant",
                    tool_calls=[
                        ChatToolCall(
                            call_id="call-1",
                            tool_name="find_files",
                            arguments={"path": "src", "pattern": "**/*.py"},
                        )
                    ],
                ),
                tool_calls=[
                    ChatToolCall(
                        call_id="call-1",
                        tool_name="find_files",
                        arguments={"path": "src", "pattern": "**/*.py"},
                    )
                ],
                raw_text="",
                finish_reason="tool_calls",
            ),
            ChatTurnResponse(
                assistant_message=ChatMessage(
                    role="assistant", content='{"answer":"Found src/app.py"}'
                ),
                final_response=ChatFinalResponse(answer="Found src/app.py"),
                token_usage=ChatTokenUsage(input_tokens=5, output_tokens=7),
                raw_text='{"answer":"Found src/app.py"}',
                finish_reason="final_response",
            ),
        ]
    )

    result = run_chat_turn(
        user_message="Find python files",
        prior_messages=[],
        root_path=tmp_path,
        config=ChatConfig(),
        llm_client=client,
    )

    assert result.status == "completed"
    assert result.final_response is not None
    assert result.final_response.answer == "Found src/app.py"
    assert [message.role for message in result.new_messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert [tool_result.tool_name for tool_result in result.tool_results] == [
        "find_files"
    ]
    assert result.new_messages[2].tool_result is not None
    assert (
        result.new_messages[2].tool_result.payload["matches"][0]["path"] == "src/app.py"
    )


def test_run_chat_turn_executes_multiple_tool_calls_in_order(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "app.py", "print('hi')\n")
    client = MockLLMClient(
        chat_canned_turns=[
            ChatTurnResponse(
                assistant_message=ChatMessage(
                    role="assistant",
                    tool_calls=[
                        ChatToolCall(
                            call_id="call-1",
                            tool_name="list_directory",
                            arguments={"path": "src"},
                        ),
                        ChatToolCall(
                            call_id="call-2",
                            tool_name="get_file_info",
                            arguments={"path": "src/app.py"},
                        ),
                    ],
                ),
                tool_calls=[
                    ChatToolCall(
                        call_id="call-1",
                        tool_name="list_directory",
                        arguments={"path": "src"},
                    ),
                    ChatToolCall(
                        call_id="call-2",
                        tool_name="get_file_info",
                        arguments={"path": "src/app.py"},
                    ),
                ],
                raw_text="",
                finish_reason="tool_calls",
            ),
            ChatTurnResponse(
                assistant_message=ChatMessage(
                    role="assistant",
                    content='{"answer":"Directory and file inspected"}',
                ),
                final_response=ChatFinalResponse(answer="Directory and file inspected"),
                raw_text='{"answer":"Directory and file inspected"}',
                finish_reason="final_response",
            ),
        ]
    )

    result = run_chat_turn(
        user_message="Inspect src",
        prior_messages=[],
        root_path=tmp_path,
        config=ChatConfig(),
        llm_client=client,
    )

    assert [tool_result.tool_name for tool_result in result.tool_results] == [
        "list_directory",
        "get_file_info",
    ]
    assert result.new_messages[2].tool_result is not None
    assert result.new_messages[3].tool_result is not None
    assert (
        result.new_messages[2].tool_result.payload["entries"][0]["path"] == "src/app.py"
    )
    assert result.new_messages[3].tool_result.payload["resolved_path"] == "src/app.py"


def test_run_chat_turn_unknown_tool_becomes_error_result_and_model_recovers(
    tmp_path: Path,
) -> None:
    client = MockLLMClient(
        chat_canned_turns=[
            ChatTurnResponse(
                assistant_message=ChatMessage(
                    role="assistant",
                    tool_calls=[
                        ChatToolCall(
                            call_id="call-1",
                            tool_name="unknown_tool",
                            arguments={},
                        )
                    ],
                ),
                tool_calls=[
                    ChatToolCall(
                        call_id="call-1",
                        tool_name="unknown_tool",
                        arguments={},
                    )
                ],
                raw_text="",
                finish_reason="tool_calls",
            ),
            ChatTurnResponse(
                assistant_message=ChatMessage(
                    role="assistant", content='{"answer":"Recovered"}'
                ),
                final_response=ChatFinalResponse(answer="Recovered"),
                raw_text='{"answer":"Recovered"}',
                finish_reason="final_response",
            ),
        ]
    )

    result = run_chat_turn(
        user_message="Try a tool",
        prior_messages=[],
        root_path=tmp_path,
        config=ChatConfig(),
        llm_client=client,
    )

    assert result.status == "completed"
    assert result.tool_results[0].status == "error"
    assert "Unknown chat tool" in (result.tool_results[0].error_message or "")
    assert result.final_response is not None
    assert result.final_response.answer == "Recovered"


def test_run_chat_turn_invalid_tool_arguments_become_error_results(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "notes.txt", "abcdef")
    client = MockLLMClient(
        chat_canned_turns=[
            ChatTurnResponse(
                assistant_message=ChatMessage(
                    role="assistant",
                    tool_calls=[
                        ChatToolCall(
                            call_id="call-1",
                            tool_name="read_file",
                            arguments={
                                "path": "notes.txt",
                                "start_char": 4,
                                "end_char": 4,
                            },
                        )
                    ],
                ),
                tool_calls=[
                    ChatToolCall(
                        call_id="call-1",
                        tool_name="read_file",
                        arguments={
                            "path": "notes.txt",
                            "start_char": 4,
                            "end_char": 4,
                        },
                    )
                ],
                raw_text="",
                finish_reason="tool_calls",
            ),
            ChatTurnResponse(
                assistant_message=ChatMessage(
                    role="assistant", content='{"answer":"Recovered after arg error"}'
                ),
                final_response=ChatFinalResponse(answer="Recovered after arg error"),
                raw_text='{"answer":"Recovered after arg error"}',
                finish_reason="final_response",
            ),
        ]
    )

    result = run_chat_turn(
        user_message="Read file badly",
        prior_messages=[],
        root_path=tmp_path,
        config=ChatConfig(),
        llm_client=client,
    )

    assert result.status == "completed"
    assert result.tool_results[0].status == "error"
    assert "end_char must be greater than start_char" in (
        result.tool_results[0].error_message or ""
    )
    assert result.final_response is not None
    assert result.final_response.answer == "Recovered after arg error"


def test_run_chat_turn_returns_needs_continuation_at_round_boundary(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "src" / "app.py", "print('hi')\n")
    client = MockLLMClient(
        chat_canned_turns=[
            ChatTurnResponse(
                assistant_message=ChatMessage(
                    role="assistant",
                    tool_calls=[
                        ChatToolCall(
                            call_id="call-1",
                            tool_name="find_files",
                            arguments={"path": "src", "pattern": "**/*.py"},
                        )
                    ],
                ),
                tool_calls=[
                    ChatToolCall(
                        call_id="call-1",
                        tool_name="find_files",
                        arguments={"path": "src", "pattern": "**/*.py"},
                    )
                ],
                raw_text="",
                finish_reason="tool_calls",
            )
        ]
    )
    config = ChatConfig.model_validate({"session": {"max_tool_round_trips": 1}})

    result = run_chat_turn(
        user_message="Need more than one round",
        prior_messages=[],
        root_path=tmp_path,
        config=config,
        llm_client=client,
    )

    assert result.status == "needs_continuation"
    assert result.final_response is None
    assert result.continuation_reason is not None
    assert "more tool rounds" in result.continuation_reason
    assert len(result.tool_results) == 1


def test_run_chat_turn_pauses_when_one_round_exceeds_tool_call_limit(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "src" / "app.py", "print('hi')\n")
    client = MockLLMClient(
        chat_canned_turns=[
            ChatTurnResponse(
                assistant_message=ChatMessage(
                    role="assistant",
                    tool_calls=[
                        ChatToolCall(
                            call_id="call-1",
                            tool_name="list_directory",
                            arguments={"path": "src"},
                        ),
                        ChatToolCall(
                            call_id="call-2",
                            tool_name="find_files",
                            arguments={"path": "src", "pattern": "**/*.py"},
                        ),
                    ],
                ),
                tool_calls=[
                    ChatToolCall(
                        call_id="call-1",
                        tool_name="list_directory",
                        arguments={"path": "src"},
                    ),
                    ChatToolCall(
                        call_id="call-2",
                        tool_name="find_files",
                        arguments={"path": "src", "pattern": "**/*.py"},
                    ),
                ],
                raw_text="",
                finish_reason="tool_calls",
            )
        ]
    )
    config = ChatConfig.model_validate({"session": {"max_tool_calls_per_round": 1}})

    result = run_chat_turn(
        user_message="Inspect src with too many calls",
        prior_messages=[],
        root_path=tmp_path,
        config=config,
        llm_client=client,
    )

    assert result.status == "needs_continuation"
    assert result.final_response is None
    assert result.continuation_reason is not None
    assert "more tool calls in one round than allowed" in result.continuation_reason
    assert result.tool_results == []
    assert [message.role for message in result.new_messages] == ["user", "assistant"]


def test_run_chat_turn_allows_exact_per_round_tool_call_limit(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "app.py", "print('hi')\n")
    client = MockLLMClient(
        chat_canned_turns=[
            ChatTurnResponse(
                assistant_message=ChatMessage(
                    role="assistant",
                    tool_calls=[
                        ChatToolCall(
                            call_id="call-1",
                            tool_name="list_directory",
                            arguments={"path": "src"},
                        ),
                        ChatToolCall(
                            call_id="call-2",
                            tool_name="get_file_info",
                            arguments={"path": "src/app.py"},
                        ),
                    ],
                ),
                tool_calls=[
                    ChatToolCall(
                        call_id="call-1",
                        tool_name="list_directory",
                        arguments={"path": "src"},
                    ),
                    ChatToolCall(
                        call_id="call-2",
                        tool_name="get_file_info",
                        arguments={"path": "src/app.py"},
                    ),
                ],
                raw_text="",
                finish_reason="tool_calls",
            ),
            ChatTurnResponse(
                assistant_message=ChatMessage(
                    role="assistant",
                    content='{"answer":"Exactly at the per-round limit"}',
                ),
                final_response=ChatFinalResponse(
                    answer="Exactly at the per-round limit"
                ),
                raw_text='{"answer":"Exactly at the per-round limit"}',
                finish_reason="final_response",
            ),
        ]
    )
    config = ChatConfig.model_validate({"session": {"max_tool_calls_per_round": 2}})

    result = run_chat_turn(
        user_message="Use exactly two tools",
        prior_messages=[],
        root_path=tmp_path,
        config=config,
        llm_client=client,
    )

    assert result.status == "completed"
    assert [tool_result.tool_name for tool_result in result.tool_results] == [
        "list_directory",
        "get_file_info",
    ]


def test_run_chat_turn_pauses_when_total_tool_call_budget_would_be_exceeded(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "src" / "app.py", "print('hi')\n")
    client = MockLLMClient(
        chat_canned_turns=[
            ChatTurnResponse(
                assistant_message=ChatMessage(
                    role="assistant",
                    tool_calls=[
                        ChatToolCall(
                            call_id="call-1",
                            tool_name="list_directory",
                            arguments={"path": "src"},
                        ),
                        ChatToolCall(
                            call_id="call-2",
                            tool_name="get_file_info",
                            arguments={"path": "src/app.py"},
                        ),
                    ],
                ),
                tool_calls=[
                    ChatToolCall(
                        call_id="call-1",
                        tool_name="list_directory",
                        arguments={"path": "src"},
                    ),
                    ChatToolCall(
                        call_id="call-2",
                        tool_name="get_file_info",
                        arguments={"path": "src/app.py"},
                    ),
                ],
                raw_text="",
                finish_reason="tool_calls",
            ),
            ChatTurnResponse(
                assistant_message=ChatMessage(
                    role="assistant",
                    tool_calls=[
                        ChatToolCall(
                            call_id="call-3",
                            tool_name="find_files",
                            arguments={"path": "src", "pattern": "**/*.py"},
                        )
                    ],
                ),
                tool_calls=[
                    ChatToolCall(
                        call_id="call-3",
                        tool_name="find_files",
                        arguments={"path": "src", "pattern": "**/*.py"},
                    )
                ],
                raw_text="",
                finish_reason="tool_calls",
            ),
        ]
    )
    config = ChatConfig.model_validate(
        {
            "session": {
                "max_tool_calls_per_round": 2,
                "max_total_tool_calls_per_turn": 2,
            }
        }
    )

    result = run_chat_turn(
        user_message="Run out of total tool budget",
        prior_messages=[],
        root_path=tmp_path,
        config=config,
        llm_client=client,
    )

    assert result.status == "needs_continuation"
    assert result.final_response is None
    assert result.continuation_reason is not None
    assert "more total tool-call budget" in result.continuation_reason
    assert [tool_result.tool_name for tool_result in result.tool_results] == [
        "list_directory",
        "get_file_info",
    ]
    assert [message.role for message in result.new_messages] == [
        "user",
        "assistant",
        "tool",
        "tool",
        "assistant",
    ]


def test_run_chat_turn_allows_exact_total_tool_call_limit(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "app.py", "print('hi')\n")
    client = MockLLMClient(
        chat_canned_turns=[
            ChatTurnResponse(
                assistant_message=ChatMessage(
                    role="assistant",
                    tool_calls=[
                        ChatToolCall(
                            call_id="call-1",
                            tool_name="list_directory",
                            arguments={"path": "src"},
                        ),
                        ChatToolCall(
                            call_id="call-2",
                            tool_name="get_file_info",
                            arguments={"path": "src/app.py"},
                        ),
                    ],
                ),
                tool_calls=[
                    ChatToolCall(
                        call_id="call-1",
                        tool_name="list_directory",
                        arguments={"path": "src"},
                    ),
                    ChatToolCall(
                        call_id="call-2",
                        tool_name="get_file_info",
                        arguments={"path": "src/app.py"},
                    ),
                ],
                raw_text="",
                finish_reason="tool_calls",
            ),
            ChatTurnResponse(
                assistant_message=ChatMessage(
                    role="assistant",
                    content='{"answer":"Exactly at the total-call limit"}',
                ),
                final_response=ChatFinalResponse(
                    answer="Exactly at the total-call limit"
                ),
                raw_text='{"answer":"Exactly at the total-call limit"}',
                finish_reason="final_response",
            ),
        ]
    )
    config = ChatConfig.model_validate(
        {
            "session": {
                "max_tool_calls_per_round": 2,
                "max_total_tool_calls_per_turn": 2,
            }
        }
    )

    result = run_chat_turn(
        user_message="Use the whole total tool budget",
        prior_messages=[],
        root_path=tmp_path,
        config=config,
        llm_client=client,
    )

    assert result.status == "completed"
    assert result.final_response is not None
    assert result.final_response.answer == "Exactly at the total-call limit"
    assert len(result.tool_results) == 2


def test_run_chat_session_turn_stores_turns_and_reuses_follow_up_context(
    tmp_path: Path,
) -> None:
    client = _RecordingChatClient(
        responses=[
            ChatTurnResponse(
                assistant_message=ChatMessage(
                    role="assistant", content='{"answer":"First answer"}'
                ),
                final_response=ChatFinalResponse(answer="First answer"),
                token_usage=ChatTokenUsage(total_tokens=9),
                raw_text='{"answer":"First answer"}',
                finish_reason="final_response",
            ),
            ChatTurnResponse(
                assistant_message=ChatMessage(
                    role="assistant", content='{"answer":"Second answer"}'
                ),
                final_response=ChatFinalResponse(answer="Second answer"),
                raw_text='{"answer":"Second answer"}',
                finish_reason="final_response",
            ),
        ]
    )

    first_result = run_chat_session_turn(
        user_message="First question",
        session_state=ChatSessionState(),
        root_path=tmp_path,
        config=ChatConfig(),
        llm_client=client,
    )

    assert first_result.session_state is not None
    assert len(first_result.session_state.turns) == 1
    assert first_result.session_state.active_context_start_turn == 0

    second_result = run_chat_session_turn(
        user_message="Follow up question",
        session_state=first_result.session_state,
        root_path=tmp_path,
        config=ChatConfig(),
        llm_client=client,
    )

    assert second_result.session_state is not None
    assert len(second_result.session_state.turns) == 2
    assert [message.role for message in client.requests[1].messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert client.requests[1].messages[1].content == "First question"
    assert client.requests[1].messages[2].content == '{"answer":"First answer"}'


def test_run_chat_session_turn_populates_session_and_active_context_tokens(
    tmp_path: Path,
) -> None:
    client = _RecordingChatClient(
        responses=[
            ChatTurnResponse(
                assistant_message=ChatMessage(
                    role="assistant", content='{"answer":"First answer"}'
                ),
                final_response=ChatFinalResponse(answer="First answer"),
                token_usage=ChatTokenUsage(total_tokens=9),
                raw_text='{"answer":"First answer"}',
                finish_reason="final_response",
            ),
            ChatTurnResponse(
                assistant_message=ChatMessage(
                    role="assistant", content='{"answer":"Second answer"}'
                ),
                final_response=ChatFinalResponse(answer="Second answer"),
                raw_text='{"answer":"Second answer"}',
                finish_reason="final_response",
            ),
        ]
    )

    first_result = run_chat_session_turn(
        user_message="First question",
        session_state=ChatSessionState(),
        root_path=tmp_path,
        config=ChatConfig(),
        llm_client=client,
    )
    assert first_result.token_usage is not None
    assert first_result.token_usage.session_tokens == 9
    assert first_result.token_usage.total_tokens == 9
    assert first_result.token_usage.active_context_tokens is not None
    assert first_result.token_usage.active_context_tokens > 0

    second_result = run_chat_session_turn(
        user_message="Second question",
        session_state=first_result.session_state or ChatSessionState(),
        root_path=tmp_path,
        config=ChatConfig(),
        llm_client=client,
    )

    assert second_result.token_usage is not None
    assert second_result.session_state is not None
    assert second_result.token_usage.session_tokens is not None
    assert second_result.token_usage.session_tokens > 9
    active_context_messages = [
        ChatMessage(
            role="system",
            content=build_chat_system_prompt(tool_limits=ChatToolLimits()),
        ),
        *(
            message
            for turn in second_result.session_state.turns[
                second_result.session_state.active_context_start_turn :
            ]
            for message in turn.new_messages
        ),
    ]
    assert second_result.token_usage.active_context_tokens == _estimate_messages_tokens(
        active_context_messages
    )


def test_run_chat_session_turn_truncates_oldest_whole_turns_when_context_is_too_large(
    tmp_path: Path,
) -> None:
    client = _RecordingChatClient(
        responses=[
            ChatTurnResponse(
                assistant_message=ChatMessage(
                    role="assistant", content='{"answer":"First answer"}'
                ),
                final_response=ChatFinalResponse(answer="First answer"),
                raw_text='{"answer":"First answer"}',
                finish_reason="final_response",
            ),
            ChatTurnResponse(
                assistant_message=ChatMessage(
                    role="assistant", content='{"answer":"Second answer"}'
                ),
                final_response=ChatFinalResponse(answer="Second answer"),
                raw_text='{"answer":"Second answer"}',
                finish_reason="final_response",
            ),
        ]
    )

    first_result = run_chat_session_turn(
        user_message="alpha",
        session_state=ChatSessionState(),
        root_path=tmp_path,
        config=ChatConfig(),
        llm_client=client,
    )
    system_tokens = len(
        tokenize(build_chat_system_prompt(tool_limits=ChatToolLimits()))
    )
    config = ChatConfig.model_validate(
        {"session": {"max_context_tokens": system_tokens + 1}}
    )

    second_result = run_chat_session_turn(
        user_message="beta",
        session_state=first_result.session_state or ChatSessionState(),
        root_path=tmp_path,
        config=config,
        llm_client=client,
    )

    assert second_result.session_state is not None
    assert len(second_result.session_state.turns) == 2
    assert second_result.session_state.active_context_start_turn == 1
    assert second_result.context_warning is not None
    assert "removed from active context" in second_result.context_warning
    assert [message.role for message in client.requests[1].messages] == [
        "system",
        "user",
    ]


def test_run_chat_session_turn_does_not_truncate_when_candidate_context_exactly_fits(
    tmp_path: Path,
) -> None:
    client = _RecordingChatClient(
        responses=[
            ChatTurnResponse(
                assistant_message=ChatMessage(
                    role="assistant", content='{"answer":"First answer"}'
                ),
                final_response=ChatFinalResponse(answer="First answer"),
                raw_text='{"answer":"First answer"}',
                finish_reason="final_response",
            ),
            ChatTurnResponse(
                assistant_message=ChatMessage(
                    role="assistant", content='{"answer":"Second answer"}'
                ),
                final_response=ChatFinalResponse(answer="Second answer"),
                raw_text='{"answer":"Second answer"}',
                finish_reason="final_response",
            ),
        ]
    )

    first_result = run_chat_session_turn(
        user_message="alpha",
        session_state=ChatSessionState(),
        root_path=tmp_path,
        config=ChatConfig(),
        llm_client=client,
    )
    retained_messages = [
        message
        for turn in (first_result.session_state or ChatSessionState()).turns
        for message in turn.new_messages
    ]
    candidate_messages = [
        ChatMessage(
            role="system",
            content=build_chat_system_prompt(tool_limits=ChatToolLimits()),
        ),
        *retained_messages,
        ChatMessage(role="user", content="beta"),
    ]
    config = ChatConfig.model_validate(
        {
            "session": {
                "max_context_tokens": _estimate_messages_tokens(candidate_messages)
            }
        }
    )

    second_result = run_chat_session_turn(
        user_message="beta",
        session_state=first_result.session_state or ChatSessionState(),
        root_path=tmp_path,
        config=config,
        llm_client=client,
    )

    assert second_result.session_state is not None
    assert second_result.session_state.active_context_start_turn == 0
    assert second_result.context_warning is None
    assert [message.role for message in client.requests[1].messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]


def test_run_chat_session_turn_stores_needs_continuation_turns_in_session_state(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "src" / "app.py", "print('hi')\n")
    client = MockLLMClient(
        chat_canned_turns=[
            ChatTurnResponse(
                assistant_message=ChatMessage(
                    role="assistant",
                    tool_calls=[
                        ChatToolCall(
                            call_id="call-1",
                            tool_name="find_files",
                            arguments={"path": "src", "pattern": "**/*.py"},
                        )
                    ],
                ),
                tool_calls=[
                    ChatToolCall(
                        call_id="call-1",
                        tool_name="find_files",
                        arguments={"path": "src", "pattern": "**/*.py"},
                    )
                ],
                raw_text="",
                finish_reason="tool_calls",
            )
        ]
    )
    config = ChatConfig.model_validate({"session": {"max_tool_round_trips": 1}})

    result = run_chat_session_turn(
        user_message="Need more than one round",
        session_state=ChatSessionState(),
        root_path=tmp_path,
        config=config,
        llm_client=client,
    )

    assert result.status == "needs_continuation"
    assert result.session_state is not None
    assert len(result.session_state.turns) == 1
    assert result.session_state.turns[0].status == "needs_continuation"
    assert result.session_state.turns[0].continuation_reason is not None


def test_chat_tool_package_export_is_available() -> None:
    chat_package = import_module("engllm_chat.tools.chat")

    assert callable(chat_package.run_chat_turn)
    assert callable(chat_package.run_chat_session_turn)
    assert callable(chat_package.run_streaming_chat_session_turn)


def test_chat_workflow_models_validate_required_fields() -> None:
    assert FindFilesArgs(pattern="  **/*.py  ").pattern == "**/*.py"

    with pytest.raises(ValueError, match="query must not be empty"):
        SearchTextArgs(query="   ")

    with pytest.raises(ValueError, match="end_char must be greater than start_char"):
        ReadFileArgs(path="notes.txt", start_char=3, end_char=3)

    with pytest.raises(ValueError, match="completed chat turns require final_response"):
        ChatSessionTurnRecord(status="completed")

    with pytest.raises(
        ValueError, match="needs_continuation chat turns require continuation_reason"
    ):
        ChatWorkflowTurnResult(status="needs_continuation")

    with pytest.raises(ValueError, match="active_context_start_turn cannot be greater"):
        ChatSessionState(
            turns=[
                ChatSessionTurnRecord(
                    status="interrupted",
                    interruption_reason="Stopped.",
                )
            ],
            active_context_start_turn=2,
        )


def test_run_streaming_chat_session_turn_emits_status_and_completed_result(
    tmp_path: Path,
) -> None:
    client = _RecordingChatClient(
        responses=[
            ChatTurnResponse(
                assistant_message=ChatMessage(
                    role="assistant",
                    content='{"answer":"Known"}',
                ),
                final_response=ChatFinalResponse(answer="Known"),
                token_usage=ChatTokenUsage(total_tokens=9),
                raw_text='{"answer":"Known"}',
            )
        ]
    )

    events = list(
        run_streaming_chat_session_turn(
            user_message="What is here?",
            session_state=ChatSessionState(),
            root_path=tmp_path,
            config=ChatConfig(),
            llm_client=client,
        )
    )

    assert [
        event.status for event in events if isinstance(event, ChatWorkflowStatusEvent)
    ] == [
        "thinking",
        "drafting answer",
    ]
    result_event = next(
        event for event in events if isinstance(event, ChatWorkflowResultEvent)
    )
    assert result_event.result.status == "completed"
    assert result_event.result.final_response is not None
    assert result_event.result.final_response.answer == "Known"
    assert result_event.result.session_state is not None
    assert result_event.result.session_state.turns[0].status == "completed"
    assert len(client.requests) == 1


def test_run_streaming_chat_session_turn_executes_tool_rounds_and_statuses(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "src" / "app.py", "print('hi')\n")
    tool_call = ChatToolCall(
        call_id="call-1",
        tool_name="find_files",
        arguments={"path": "src", "pattern": "**/*.py"},
    )
    client = _RecordingChatClient(
        responses=[
            ChatTurnResponse(
                assistant_message=ChatMessage(
                    role="assistant",
                    tool_calls=[tool_call],
                ),
                tool_calls=[tool_call],
                token_usage=ChatTokenUsage(total_tokens=3),
                finish_reason="tool_calls",
            ),
            ChatTurnResponse(
                assistant_message=ChatMessage(
                    role="assistant",
                    content='{"answer":"Found src/app.py"}',
                ),
                final_response=ChatFinalResponse(answer="Found src/app.py"),
                token_usage=ChatTokenUsage(total_tokens=8),
                raw_text='{"answer":"Found src/app.py"}',
            ),
        ]
    )

    stream = run_streaming_chat_session_turn(
        user_message="Find python files",
        session_state=ChatSessionState(),
        root_path=tmp_path,
        config=ChatConfig(),
        llm_client=client,
    )
    events = list(stream)
    status_labels = [
        event.status for event in events if isinstance(event, ChatWorkflowStatusEvent)
    ]
    result_events = [
        event for event in events if isinstance(event, ChatWorkflowResultEvent)
    ]

    assert "thinking" in status_labels
    assert "listing files" in status_labels
    assert "drafting answer" in status_labels
    assert len(result_events) == 1
    assert result_events[0].result.final_response is not None
    assert result_events[0].result.final_response.answer == "Found src/app.py"


def test_run_streaming_chat_session_turn_cancel_stops_before_tool_execution(
    tmp_path: Path,
) -> None:
    tool_call = ChatToolCall(
        call_id="call-1",
        tool_name="find_files",
        arguments={"path": ".", "pattern": "**/*.py"},
    )
    client = _RecordingChatClient(
        responses=[
            ChatTurnResponse(
                assistant_message=ChatMessage(
                    role="assistant",
                    content="Searching",
                    tool_calls=[tool_call],
                ),
                tool_calls=[tool_call],
                finish_reason="tool_calls",
            )
        ]
    )

    stream = run_streaming_chat_session_turn(
        user_message="Question",
        session_state=ChatSessionState(),
        root_path=tmp_path,
        config=ChatConfig(),
        llm_client=client,
    )

    iterator = iter(stream)
    assert isinstance(next(iterator), ChatWorkflowStatusEvent)
    tool_status = next(iterator)
    assert isinstance(tool_status, ChatWorkflowStatusEvent)
    assert tool_status.status == "listing files"
    stream.cancel()
    remaining = list(iterator)

    assert len(remaining) == 1
    assert isinstance(remaining[0], ChatWorkflowResultEvent)
    assert remaining[0].result.status == "interrupted"
    assert remaining[0].result.interruption_reason == "Interrupted by user."
    assert remaining[0].result.tool_results == []
    assistant_message = next(
        message
        for message in remaining[0].result.new_messages
        if message.role == "assistant"
    )
    assert assistant_message.content == "Searching"
    assert assistant_message.completion_state == "complete"


def test_run_streaming_chat_session_turn_cancel_before_iteration_returns_interrupted(
    tmp_path: Path,
) -> None:
    client = _RecordingChatClient(responses=[])
    stream = run_streaming_chat_session_turn(
        user_message="Question",
        session_state=ChatSessionState(),
        root_path=tmp_path,
        config=ChatConfig(),
        llm_client=client,
    )

    stream.cancel()
    events = list(stream)

    assert len(client.requests) == 0
    assert isinstance(events[0], ChatWorkflowStatusEvent)
    assert events[0].status == "thinking"
    assert isinstance(events[1], ChatWorkflowResultEvent)
    assert events[1].result.status == "interrupted"
    assert events[1].result.interruption_reason == "Interrupted by user."


def test_run_streaming_chat_session_turn_needs_continuation_for_stream_limits(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "src" / "app.py", "print('hi')\n")
    tool_call = ChatToolCall(
        call_id="call-1",
        tool_name="find_files",
        arguments={"path": "src", "pattern": "**/*.py"},
    )

    per_round_client = _RecordingChatClient(
        responses=[
            ChatTurnResponse(
                assistant_message=ChatMessage(
                    role="assistant",
                    tool_calls=[
                        tool_call,
                        tool_call.model_copy(update={"call_id": "call-2"}),
                    ],
                ),
                tool_calls=[
                    tool_call,
                    tool_call.model_copy(update={"call_id": "call-2"}),
                ],
                finish_reason="tool_calls",
            )
        ]
    )
    per_round_config = ChatConfig.model_validate(
        {"session": {"max_tool_calls_per_round": 1}}
    )
    per_round_events = list(
        run_streaming_chat_session_turn(
            user_message="Question",
            session_state=ChatSessionState(),
            root_path=tmp_path,
            config=per_round_config,
            llm_client=per_round_client,
        )
    )
    per_round_result = next(
        event.result
        for event in per_round_events
        if isinstance(event, ChatWorkflowResultEvent)
    )
    assert per_round_result.status == "needs_continuation"
    assert "one round than allowed" in (per_round_result.continuation_reason or "")

    total_budget_client = _RecordingChatClient(
        responses=[
            ChatTurnResponse(
                assistant_message=ChatMessage(
                    role="assistant",
                    tool_calls=[tool_call],
                ),
                tool_calls=[tool_call],
                finish_reason="tool_calls",
            ),
            ChatTurnResponse(
                assistant_message=ChatMessage(
                    role="assistant",
                    tool_calls=[tool_call.model_copy(update={"call_id": "call-2"})],
                ),
                tool_calls=[tool_call.model_copy(update={"call_id": "call-2"})],
                finish_reason="tool_calls",
            ),
        ]
    )
    total_budget_config = ChatConfig.model_validate(
        {"session": {"max_total_tool_calls_per_turn": 1, "max_tool_round_trips": 3}}
    )
    total_budget_events = list(
        run_streaming_chat_session_turn(
            user_message="Question",
            session_state=ChatSessionState(),
            root_path=tmp_path,
            config=total_budget_config,
            llm_client=total_budget_client,
        )
    )
    total_budget_result = next(
        event.result
        for event in total_budget_events
        if isinstance(event, ChatWorkflowResultEvent)
    )
    assert total_budget_result.status == "needs_continuation"
    assert "total tool-call budget" in (total_budget_result.continuation_reason or "")

    round_trip_client = _RecordingChatClient(
        responses=[
            ChatTurnResponse(
                assistant_message=ChatMessage(
                    role="assistant",
                    tool_calls=[tool_call],
                ),
                tool_calls=[tool_call],
                finish_reason="tool_calls",
            )
        ]
    )
    round_trip_config = ChatConfig.model_validate(
        {"session": {"max_tool_round_trips": 1}}
    )
    round_trip_events = list(
        run_streaming_chat_session_turn(
            user_message="Question",
            session_state=ChatSessionState(),
            root_path=tmp_path,
            config=round_trip_config,
            llm_client=round_trip_client,
        )
    )
    round_trip_result = next(
        event.result
        for event in round_trip_events
        if isinstance(event, ChatWorkflowResultEvent)
    )
    assert round_trip_result.status == "needs_continuation"
    assert "more tool rounds" in (round_trip_result.continuation_reason or "")


def test_run_streaming_chat_session_turn_status_labels_cover_search_and_read(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "src" / "app.py", "alpha\nbeta\n")
    search_call = ChatToolCall(
        call_id="call-1",
        tool_name="search_text",
        arguments={"path": "src", "query": "alpha"},
    )
    read_call = ChatToolCall(
        call_id="call-2",
        tool_name="read_file",
        arguments={"path": "src/app.py", "start_char": 0, "end_char": 5},
    )
    client = _RecordingChatClient(
        responses=[
            ChatTurnResponse(
                assistant_message=ChatMessage(
                    role="assistant",
                    tool_calls=[search_call, read_call],
                ),
                tool_calls=[search_call, read_call],
                finish_reason="tool_calls",
            ),
            ChatTurnResponse(
                assistant_message=ChatMessage(
                    role="assistant",
                    content='{"answer":"Done"}',
                ),
                final_response=ChatFinalResponse(answer="Done"),
                raw_text='{"answer":"Done"}',
            ),
        ]
    )

    events = list(
        run_streaming_chat_session_turn(
            user_message="Question",
            session_state=ChatSessionState(),
            root_path=tmp_path,
            config=ChatConfig(),
            llm_client=client,
        )
    )
    statuses = [
        event.status for event in events if isinstance(event, ChatWorkflowStatusEvent)
    ]
    assert "searching text" in statuses
    assert "reading file" in statuses
