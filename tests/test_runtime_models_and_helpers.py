"""Focused validation tests for runtime helper models and provider primitives."""

from __future__ import annotations

import json
import urllib.error
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from engllm_chat.config.loader import load_chat_config
from engllm_chat.domain.models import (
    ChatFinalResponse,
    ChatMessage,
    ChatToolCall,
    ChatToolResult,
    QueryAnswer,
    SectionDraft,
    SectionUpdateProposal,
)
from engllm_chat.llm import ollama as ollama_module
from engllm_chat.llm import openai_compatible as openai_compatible_module
from engllm_chat.llm.base import (
    ChatAssistantDeltaEvent,
    ChatFinalResponseEvent,
    ChatInterruptedEvent,
    ChatToolCallsEvent,
    ChatTurnRequest,
    ChatTurnResponse,
    StructuredGenerationRequest,
)
from engllm_chat.llm.mock import MockLLMClient
from engllm_chat.tools.chat.models import (
    ChatSessionState,
    ChatSessionTurnRecord,
    ChatWorkflowResultEvent,
    ChatWorkflowStatusEvent,
    ChatWorkflowTurnResult,
    FindFilesArgs,
    GetFileInfoArgs,
    ListDirectoryArgs,
    ListDirectoryRecursiveArgs,
    ReadFileArgs,
    SearchTextArgs,
)


class _FakeStructuredModel(BaseModel):
    name: str
    count: int
    enabled: bool
    tags: list[str]
    metadata: dict[str, str]
    maybe: str | None = None


class _FakeURLResponse:
    def __init__(self, *, status: int = 200, body: str) -> None:
        self.status = status
        self._body = body

    def __enter__(self) -> _FakeURLResponse:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return self._body.encode("utf-8")


def test_chat_turn_and_stream_event_validators_reject_invalid_combinations() -> None:
    tool_call = ChatToolCall(call_id="call-1", tool_name="read_file", arguments={})
    assistant = ChatMessage(role="assistant", content="hello")

    with pytest.raises(ValueError, match="role='assistant'"):
        ChatTurnResponse(
            assistant_message=ChatMessage(role="user", content="bad"),
            final_response=ChatFinalResponse(answer="done"),
        )

    with pytest.raises(ValueError, match="requires at least one tool call"):
        ChatTurnResponse(assistant_message=assistant, finish_reason="tool_calls")

    with pytest.raises(ValueError, match="does not allow final_response"):
        ChatTurnResponse(
            assistant_message=ChatMessage(
                role="assistant",
                content="tool",
                tool_calls=[tool_call],
            ),
            tool_calls=[tool_call],
            final_response=ChatFinalResponse(answer="bad"),
            finish_reason="tool_calls",
        )

    with pytest.raises(ValueError, match="does not allow tool_calls"):
        ChatTurnResponse(
            assistant_message=assistant,
            tool_calls=[tool_call],
            final_response=ChatFinalResponse(answer="done"),
            finish_reason="final_response",
        )

    with pytest.raises(ValueError, match="requires final_response"):
        ChatTurnResponse(assistant_message=assistant, finish_reason="final_response")

    with pytest.raises(ValueError, match="does not allow tool_calls"):
        ChatTurnResponse(
            assistant_message=assistant,
            tool_calls=[tool_call],
            finish_reason="interrupted",
        )

    with pytest.raises(ValueError, match="does not allow final_response"):
        ChatTurnResponse(
            assistant_message=assistant,
            final_response=ChatFinalResponse(answer="done"),
            finish_reason="interrupted",
        )

    with pytest.raises(ValueError, match="non-empty delta_text"):
        ChatAssistantDeltaEvent(delta_text="", accumulated_text="")

    with pytest.raises(ValueError, match="role='assistant'"):
        ChatToolCallsEvent(
            assistant_message=ChatMessage(role="user", content="bad"),
            tool_calls=[tool_call],
        )

    with pytest.raises(ValueError, match="at least one tool call"):
        ChatToolCallsEvent(
            assistant_message=assistant,
            tool_calls=[],
        )

    with pytest.raises(ValueError, match="role='assistant'"):
        ChatFinalResponseEvent(
            assistant_message=ChatMessage(role="user", content="bad"),
            final_response=ChatFinalResponse(answer="done"),
        )

    with pytest.raises(ValueError, match="role='assistant'"):
        ChatInterruptedEvent(
            assistant_message=ChatMessage(role="user", content="bad"),
        )


def test_runtime_argument_and_turn_models_validate_expected_states() -> None:
    assert ListDirectoryArgs(path=" src ").path == "src"
    assert ListDirectoryRecursiveArgs(path=".", max_depth=2).max_depth == 2
    assert FindFilesArgs(path=".", pattern=" *.py ").pattern == "*.py"
    assert SearchTextArgs(path=".", query=" TODO ").query == "TODO"
    assert GetFileInfoArgs(path=" README.md ").path == "README.md"
    assert ReadFileArgs(path="README.md", start_char=1, end_char=4).end_char == 4

    with pytest.raises(ValueError, match="path must not be empty"):
        ListDirectoryArgs(path="   ")
    with pytest.raises(ValueError, match="pattern must not be empty"):
        FindFilesArgs(path=".", pattern="   ")
    with pytest.raises(ValueError, match="query must not be empty"):
        SearchTextArgs(path=".", query="   ")
    with pytest.raises(ValueError, match="end_char must be greater than start_char"):
        ReadFileArgs(path="README.md", start_char=5, end_char=5)

    final = ChatFinalResponse(answer="done")
    completed = ChatSessionTurnRecord(status="completed", final_response=final)
    assert completed.final_response == final

    with pytest.raises(ValueError, match="require final_response"):
        ChatSessionTurnRecord(status="completed")
    with pytest.raises(ValueError, match="require continuation_reason"):
        ChatSessionTurnRecord(status="needs_continuation")
    with pytest.raises(ValueError, match="require interruption_reason"):
        ChatSessionTurnRecord(status="interrupted")

    state = ChatSessionState(turns=[completed], active_context_start_turn=1)
    assert state.active_context_start_turn == 1
    with pytest.raises(ValueError, match="cannot be greater than the number of turns"):
        ChatSessionState(turns=[completed], active_context_start_turn=2)

    result = ChatWorkflowTurnResult(status="completed", final_response=final)
    assert result.final_response == final
    with pytest.raises(ValueError, match="require continuation_reason"):
        ChatWorkflowTurnResult(status="needs_continuation")
    with pytest.raises(ValueError, match="require interruption_reason"):
        ChatWorkflowTurnResult(status="interrupted")

    status_event = ChatWorkflowStatusEvent(status="thinking")
    result_event = ChatWorkflowResultEvent(result=result)
    assert status_event.event_type == "status"
    assert result_event.event_type == "result"


def test_config_loader_rejects_missing_path_directory_invalid_yaml_and_root_shape(
    tmp_path,
) -> None:
    missing = tmp_path / "missing.yaml"
    with pytest.raises(Exception, match="Configuration file not found"):
        load_chat_config(missing)

    directory_path = tmp_path / "config-dir"
    directory_path.mkdir()
    with pytest.raises(Exception, match="Configuration path is not a file"):
        load_chat_config(directory_path)

    invalid_yaml = tmp_path / "invalid.yaml"
    invalid_yaml.write_text(":\n  - bad", encoding="utf-8")
    with pytest.raises(Exception, match="Invalid YAML"):
        load_chat_config(invalid_yaml)

    wrong_root = tmp_path / "wrong-root.yaml"
    wrong_root.write_text("- bad\n- root", encoding="utf-8")
    with pytest.raises(Exception, match="Expected mapping at root of YAML file"):
        load_chat_config(wrong_root)


def test_mock_client_covers_section_payloads_default_payloads_and_interrupted_stream() -> (
    None
):
    client = MockLLMClient()

    section_draft = client.generate_structured(
        StructuredGenerationRequest(
            system_prompt="",
            user_prompt='{"id":"S-1","title":"Overview"}',
            response_model=SectionDraft,
            model_name="mock-chat",
        )
    ).content
    assert section_draft.section_id == "S-1"
    assert section_draft.title == "Overview"

    update = client.generate_structured(
        StructuredGenerationRequest(
            system_prompt="",
            user_prompt='{"id":"S-2","title":"Design"}',
            response_model=SectionUpdateProposal,
            model_name="mock-chat",
        )
    ).content
    assert update.section_id == "S-2"
    assert update.review_priority == "medium"

    answer = client.generate_structured(
        StructuredGenerationRequest(
            system_prompt="",
            user_prompt="question",
            response_model=QueryAnswer,
            model_name="mock-chat",
        )
    ).content
    assert "mock answer" in answer.answer.lower()

    default_payload = client.generate_structured(
        StructuredGenerationRequest(
            system_prompt="",
            user_prompt="question",
            response_model=_FakeStructuredModel,
            model_name="mock-chat",
        )
    ).content
    assert default_payload.name == "TBD"
    assert default_payload.count == 0
    assert default_payload.enabled is False
    assert default_payload.tags == []
    assert default_payload.metadata == {}
    assert default_payload.maybe is None

    tool_call = ChatToolCall(call_id="call-1", tool_name="search_text", arguments={})
    interrupted_turn = ChatTurnResponse(
        assistant_message=ChatMessage(role="assistant", content="partial"),
        raw_text="partial",
        finish_reason="interrupted",
    )
    tool_turn = ChatTurnResponse(
        assistant_message=ChatMessage(
            role="assistant",
            content="working",
            tool_calls=[tool_call],
        ),
        tool_calls=[tool_call],
        finish_reason="tool_calls",
    )
    scripted_client = MockLLMClient(chat_canned_turns=[tool_turn, interrupted_turn])

    tool_stream_events = list(
        scripted_client.stream_chat_turn(
            ChatTurnRequest(
                messages=[ChatMessage(role="user", content="hello")],
                response_model=ChatFinalResponse,
                model_name="mock-chat",
            )
        )
    )
    assert isinstance(tool_stream_events[-1], ChatToolCallsEvent)

    interrupted_events = list(
        scripted_client.stream_chat_turn(
            ChatTurnRequest(
                messages=[ChatMessage(role="user", content="hello")],
                response_model=ChatFinalResponse,
                model_name="mock-chat",
            )
        )
    )
    assert isinstance(interrupted_events[-1], ChatInterruptedEvent)
    assert interrupted_events[-1].reason == "Interrupted by mock provider."

    fallback_turn = client.generate_chat_turn(
        ChatTurnRequest(
            messages=[ChatMessage(role="user", content="hello")],
            response_model=ChatFinalResponse,
            model_name="mock-chat",
        )
    )
    assert fallback_turn.finish_reason == "final_response"

    fallback_stream_events = list(
        client.stream_chat_turn(
            ChatTurnRequest(
                messages=[ChatMessage(role="user", content="hello")],
                response_model=ChatFinalResponse,
                model_name="mock-chat",
            )
        )
    )
    assert isinstance(fallback_stream_events[0], ChatAssistantDeltaEvent)
    assert isinstance(fallback_stream_events[-1], ChatFinalResponseEvent)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("127.0.0.1:11434", "http://127.0.0.1:11434/api/chat"),
        ("http://localhost:11434", "http://localhost:11434/api/chat"),
        ("http://localhost:11434/api", "http://localhost:11434/api/chat"),
        ("http://localhost:11434/custom", "http://localhost:11434/custom/api/chat"),
    ],
)
def test_ollama_helper_functions_cover_normalization_and_serialization(
    raw: str,
    expected: str,
) -> None:
    assert ollama_module._normalize_chat_url(raw) == expected

    tool_result = ChatToolResult(
        call_id="call-1", tool_name="read_file", payload={"path": "README.md"}
    )
    serialized_tool = ollama_module._serialize_chat_message(
        ChatMessage(role="tool", tool_result=tool_result)
    )
    assert serialized_tool["role"] == "tool"
    assert json.loads(serialized_tool["content"])["tool_name"] == "read_file"

    serialized_user = ollama_module._serialize_chat_message(
        ChatMessage(role="user", content=" hello ")
    )
    assert serialized_user == {"role": "user", "content": "hello"}
    assert (
        ollama_module._serialize_tool_definition(
            SimpleNamespace(
                name="read_file",
                description="Read one file",
                input_schema={"type": "object"},
            )
        )["function"]["name"]
        == "read_file"
    )
    token_usage = ollama_module._extract_token_usage(
        {"prompt_eval_count": "bad", "eval_count": "also-bad"}
    )
    assert token_usage.input_tokens == 0
    assert token_usage.output_tokens == 0


def test_ollama_helpers_and_generate_structured_error_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(Exception, match="Tool chat message missing tool_result"):
        ollama_module._serialize_chat_message(ChatMessage.model_construct(role="tool"))

    with pytest.raises(Exception, match="malformed"):
        ollama_module._extract_tool_calls({"tool_calls": ["bad"]})
    with pytest.raises(Exception, match="missing function payload"):
        ollama_module._extract_tool_calls({"tool_calls": [{}]})
    with pytest.raises(Exception, match="missing function name"):
        ollama_module._extract_tool_calls(
            {"tool_calls": [{"function": {"arguments": {}}}]}
        )
    with pytest.raises(Exception, match="missing structured arguments"):
        ollama_module._extract_tool_calls(
            {"tool_calls": [{"function": {"name": "read_file", "arguments": "bad"}}]}
        )

    success_payload = {
        "message": {"content": '{"answer":"Done"}'},
    }
    monkeypatch.setattr(
        "engllm_chat.llm.ollama.request_lib.urlopen",
        lambda *_args, **_kwargs: _FakeURLResponse(body=json.dumps(success_payload)),
    )
    client = ollama_module.OllamaLLMClient(model_name="qwen")
    response = client.generate_structured(
        StructuredGenerationRequest(
            system_prompt="system",
            user_prompt="user",
            response_model=ChatFinalResponse,
            model_name="qwen",
        )
    )
    assert response.content == ChatFinalResponse(answer="Done")

    monkeypatch.setattr(
        "engllm_chat.llm.ollama.request_lib.urlopen",
        lambda *_args, **_kwargs: _FakeURLResponse(status=500, body="server error"),
    )
    with pytest.raises(Exception, match="status 500"):
        client.generate_structured(
            StructuredGenerationRequest(
                system_prompt="system",
                user_prompt="user",
                response_model=ChatFinalResponse,
                model_name="qwen",
            )
        )

    http_error = urllib.error.HTTPError(
        url="http://localhost",
        code=404,
        msg="not found",
        hdrs=None,
        fp=None,
    )
    monkeypatch.setattr(
        http_error,
        "read",
        lambda: b"missing",
    )
    monkeypatch.setattr(
        "engllm_chat.llm.ollama.request_lib.urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(http_error),
    )
    with pytest.raises(Exception, match="status 404: missing"):
        client.generate_structured(
            StructuredGenerationRequest(
                system_prompt="system",
                user_prompt="user",
                response_model=ChatFinalResponse,
                model_name="qwen",
            )
        )

    monkeypatch.setattr(
        "engllm_chat.llm.ollama.request_lib.urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(urllib.error.URLError("down")),
    )
    with pytest.raises(Exception, match="Cannot connect to Ollama"):
        client.generate_structured(
            StructuredGenerationRequest(
                system_prompt="system",
                user_prompt="user",
                response_model=ChatFinalResponse,
                model_name="qwen",
            )
        )


def test_ollama_stream_handles_error_and_edge_cases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _StreamingResponse:
        def __init__(
            self, *, status: int = 200, lines: list[bytes], close_raises: bool = False
        ):
            self.status = status
            self._lines = list(lines)
            self._close_raises = close_raises

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.close()

        def readline(self) -> bytes:
            if not self._lines:
                return b""
            return self._lines.pop(0)

        def read(self) -> bytes:
            return b"server error"

        def close(self) -> None:
            if self._close_raises:
                raise OSError("close failed")

    client = ollama_module.OllamaLLMClient(model_name="qwen")
    request = ChatTurnRequest(
        messages=[ChatMessage(role="user", content="hello")],
        response_model=ChatFinalResponse,
        model_name="qwen",
    )

    monkeypatch.setattr(
        "engllm_chat.llm.ollama.request_lib.urlopen",
        lambda *_args, **_kwargs: _StreamingResponse(status=500, lines=[]),
    )
    with pytest.raises(Exception, match="status 500"):
        list(client.stream_chat_turn(request))

    monkeypatch.setattr(
        "engllm_chat.llm.ollama.request_lib.urlopen",
        lambda *_args, **_kwargs: _StreamingResponse(lines=[b"not-json"]),
    )
    with pytest.raises(Exception, match="malformed JSON stream chunk"):
        list(client.stream_chat_turn(request))

    monkeypatch.setattr(
        "engllm_chat.llm.ollama.request_lib.urlopen",
        lambda *_args, **_kwargs: _StreamingResponse(
            lines=[
                json.dumps({"message": {"content": 3}, "done": False}).encode("utf-8")
            ]
        ),
    )
    with pytest.raises(Exception, match="missing assistant content"):
        list(client.stream_chat_turn(request))

    monkeypatch.setattr(
        "engllm_chat.llm.ollama.request_lib.urlopen",
        lambda *_args, **_kwargs: _StreamingResponse(lines=[]),
    )
    with pytest.raises(Exception, match="ended without a final response or tool calls"):
        list(client.stream_chat_turn(request))

    monkeypatch.setattr(
        "engllm_chat.llm.ollama.request_lib.urlopen",
        lambda *_args, **_kwargs: _StreamingResponse(lines=[], close_raises=True),
    )
    cancelled = client.stream_chat_turn(request)
    cancelled.cancel()
    cancelled_events = list(cancelled)
    assert isinstance(cancelled_events[-1], ChatInterruptedEvent)


def test_openai_compatible_helpers_cover_serialization_and_stream_fallbacks() -> None:
    tool_result = ChatToolResult(
        call_id="call-1", tool_name="search_text", payload={"query": "TODO"}
    )
    serialized_tool = openai_compatible_module._serialize_chat_message(
        ChatMessage(role="tool", tool_result=tool_result)
    )
    assert serialized_tool["tool_call_id"] == "call-1"

    assistant_with_tool = openai_compatible_module._serialize_chat_message(
        ChatMessage(
            role="assistant",
            content="working",
            tool_calls=[
                ChatToolCall(
                    call_id="call-2",
                    tool_name="read_file",
                    arguments={"path": "README.md"},
                )
            ],
        )
    )
    assert assistant_with_tool["tool_calls"][0]["function"]["name"] == "read_file"

    with pytest.raises(Exception, match="missing tool_result"):
        openai_compatible_module._serialize_chat_message(
            ChatMessage.model_construct(role="tool")
        )

    assert (
        openai_compatible_module._extract_message_text(SimpleNamespace(content="hello"))
        == "hello"
    )
    assert (
        openai_compatible_module._extract_message_text(
            SimpleNamespace(content=[{"text": "a"}, SimpleNamespace(text="b")])
        )
        == "ab"
    )

    assert (
        openai_compatible_module._extract_token_usage(SimpleNamespace(usage=None))
        is None
    )
    usage = openai_compatible_module._extract_token_usage(
        SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens="bad", completion_tokens=2, total_tokens="bad"
            )
        )
    )
    assert usage is not None
    assert usage.input_tokens == 0
    assert usage.output_tokens == 2
    assert usage.total_tokens == 2

    assert openai_compatible_module._parse_tool_arguments({"path": "README.md"}) == {
        "path": "README.md"
    }
    with pytest.raises(Exception, match="malformed tool-call arguments"):
        openai_compatible_module._parse_tool_arguments("not-json")
    with pytest.raises(Exception, match="non-object tool-call arguments"):
        openai_compatible_module._parse_tool_arguments('["bad"]')

    with pytest.raises(Exception, match="missing function name"):
        openai_compatible_module._extract_tool_calls(
            SimpleNamespace(
                tool_calls=[
                    SimpleNamespace(function=SimpleNamespace(name="", arguments="{}"))
                ]
            )
        )

    class _Stream:
        def __init__(self, chunks):
            self._chunks = chunks
            self.closed = False

        def __iter__(self):
            yield from self._chunks

        def close(self):
            self.closed = True

    final_stream = openai_compatible_module._OpenAICompatibleChatTurnStream(
        provider_name="openai",
        stream=_Stream(
            [
                SimpleNamespace(choices=[]),
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content='{"answer":"Hi"}'),
                            finish_reason=None,
                        )
                    ],
                    usage=None,
                ),
            ]
        ),
        response_model=ChatFinalResponse,
    )
    final_events = list(final_stream)
    assert isinstance(final_events[-1], ChatFinalResponseEvent)

    tool_stream = openai_compatible_module._OpenAICompatibleChatTurnStream(
        provider_name="openai",
        stream=_Stream(
            [
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(
                                content=None,
                                tool_calls=[
                                    SimpleNamespace(
                                        index=0,
                                        id="call-1",
                                        function=SimpleNamespace(
                                            name="search_text",
                                            arguments='{"query":"TODO"}',
                                        ),
                                    )
                                ],
                            ),
                            finish_reason=None,
                        )
                    ],
                    usage=None,
                )
            ]
        ),
        response_model=ChatFinalResponse,
    )
    tool_events = list(tool_stream)
    assert isinstance(tool_events[-1], ChatToolCallsEvent)

    empty_stream = openai_compatible_module._OpenAICompatibleChatTurnStream(
        provider_name="openai",
        stream=_Stream([]),
        response_model=ChatFinalResponse,
    )
    with pytest.raises(Exception, match="ended without a final response or tool calls"):
        list(empty_stream)
