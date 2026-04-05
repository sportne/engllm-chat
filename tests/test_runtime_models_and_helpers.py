"""Focused validation tests for runtime helper models and provider primitives."""

from __future__ import annotations

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
        ("127.0.0.1:11434", "http://127.0.0.1:11434/v1"),
        ("http://localhost:11434", "http://localhost:11434/v1"),
        ("http://localhost:11434/api", "http://localhost:11434/api/v1"),
        ("http://localhost:11434/custom", "http://localhost:11434/custom/v1"),
    ],
)
def test_ollama_helper_functions_cover_normalization_and_serialization(
    raw: str,
    expected: str,
) -> None:
    assert openai_compatible_module.normalize_ollama_base_url(raw) == expected


def test_ollama_client_uses_openai_compatible_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeOpenAIClient:
        last_init_kwargs: dict[str, object] | None = None
        captured_payloads: list[dict[str, object]] = []

        def __init__(self, **kwargs: object) -> None:
            type(self).last_init_kwargs = dict(kwargs)
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._create)
            )

        def _create(self, **payload: object) -> object:
            type(self).captured_payloads.append(dict(payload))
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='{"answer":"Done"}', tool_calls=[]
                        )
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=1,
                    completion_tokens=2,
                    total_tokens=3,
                ),
            )

    monkeypatch.setattr(openai_compatible_module, "OpenAI", _FakeOpenAIClient)

    client = openai_compatible_module.OpenAICompatibleChatLLMClient(
        model_name="qwen",
        provider_name="ollama",
        api_key_env_var=None,
        api_key="ollama",
        base_url=openai_compatible_module.normalize_ollama_base_url(
            "http://localhost:11434"
        ),
    )
    response = client.generate_chat_turn(
        ChatTurnRequest(
            messages=[ChatMessage(role="user", content="hello")],
            response_model=ChatFinalResponse,
            model_name="qwen",
        )
    )

    assert response.final_response == ChatFinalResponse(answer="Done")
    assert _FakeOpenAIClient.last_init_kwargs == {
        "api_key": "ollama",
        "base_url": "http://localhost:11434/v1",
        "timeout": 60.0,
    }
    assert (
        _FakeOpenAIClient.captured_payloads[-1]["response_format"]["type"]
        == "json_schema"
    )


def test_ollama_stream_cancel_closes_openai_compatible_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeStreamingResponse:
        def __init__(self) -> None:
            self.closed = False

        def __iter__(self):
            yield SimpleNamespace(
                usage=None,
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="partial", tool_calls=[]),
                        finish_reason=None,
                    )
                ],
            )

        def close(self) -> None:
            self.closed = True

    class _FakeOpenAIClient:
        last_stream: _FakeStreamingResponse | None = None

        def __init__(self, **kwargs: object) -> None:
            del kwargs
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._create)
            )

        def _create(self, **payload: object) -> object:
            del payload
            type(self).last_stream = _FakeStreamingResponse()
            return type(self).last_stream

    monkeypatch.setattr(openai_compatible_module, "OpenAI", _FakeOpenAIClient)

    client = openai_compatible_module.OpenAICompatibleChatLLMClient(
        model_name="qwen",
        provider_name="ollama",
        api_key_env_var=None,
        api_key="ollama",
        base_url=openai_compatible_module.normalize_ollama_base_url(None),
    )
    cancelled = client.stream_chat_turn(
        ChatTurnRequest(
            messages=[ChatMessage(role="user", content="hello")],
            response_model=ChatFinalResponse,
            model_name="qwen",
        )
    )
    cancelled.cancel()
    cancelled_events = list(cancelled)
    assert isinstance(cancelled_events[-1], ChatInterruptedEvent)
    assert _FakeOpenAIClient.last_stream is not None
    assert _FakeOpenAIClient.last_stream.closed is True


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
