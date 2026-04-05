"""Focused tests for standalone chat LLM components."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from engllm_chat.domain.errors import LLMError, ValidationError
from engllm_chat.domain.models import (
    ChatFinalResponse,
    ChatLLMConfig,
    ChatMessage,
    ChatToolCall,
)
from engllm_chat.llm.base import (
    ChatAssistantDeltaEvent,
    ChatFinalResponseEvent,
    ChatInterruptedEvent,
    ChatToolCallsEvent,
    ChatToolDefinition,
    ChatTurnRequest,
    ChatTurnResponse,
    StructuredGenerationRequest,
    validate_json_text,
    validate_payload,
)
from engllm_chat.llm.factory import create_chat_llm_client
from engllm_chat.llm.mock import MockLLMClient
from engllm_chat.llm.openai_compatible import (
    OpenAICompatibleChatLLMClient,
    normalize_ollama_base_url,
)
from engllm_chat.probe_openai_api import main as probe_main


class _SimpleModel(BaseModel):
    value: int


class _ReadFileArgs(BaseModel):
    path: str


class _FakeFunction:
    def __init__(
        self,
        name: str,
        arguments: str | dict[str, object],
        *,
        parsed_arguments: object | None = None,
    ) -> None:
        self.name = name
        self.arguments = arguments
        self.parsed_arguments = parsed_arguments


class _FakeToolCall:
    def __init__(
        self,
        call_id: str,
        name: str,
        arguments: str | dict[str, object],
        *,
        parsed_arguments: object | None = None,
    ) -> None:
        self.id = call_id
        self.function = _FakeFunction(
            name,
            arguments,
            parsed_arguments=parsed_arguments,
        )


class _FakeMessage:
    def __init__(
        self,
        *,
        content: str | None = None,
        tool_calls: list[_FakeToolCall] | None = None,
        parsed: object | None = None,
    ) -> None:
        self.content = content
        self.tool_calls = tool_calls or []
        self.parsed = parsed


class _FakeChoice:
    def __init__(self, message: _FakeMessage) -> None:
        self.message = message


class _FakeChatCompletionResponse:
    def __init__(
        self,
        *,
        message: _FakeMessage,
        usage: object | None = None,
    ) -> None:
        self.choices = [_FakeChoice(message)]
        self.usage = usage


class _FakeUsage:
    def __init__(
        self,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int | None = None,
    ) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


class _FakeStreamingEvent:
    def __init__(
        self,
        event_type: str,
        *,
        delta: str | None = None,
        chunk: object | None = None,
    ) -> None:
        self.type = event_type
        self.delta = delta
        self.chunk = chunk


class _FakeStreamingResponse:
    def __init__(
        self,
        *,
        events: list[_FakeStreamingEvent],
        final_completion: object,
    ) -> None:
        self._events = events
        self._final_completion = final_completion
        self.closed = False

    def __iter__(self):
        yield from self._events

    def close(self) -> None:
        self.closed = True

    def get_final_completion(self) -> object:
        return self._final_completion


class _FakeStreamingManager:
    def __init__(self, stream: _FakeStreamingResponse) -> None:
        self.stream = stream
        self.entered = False
        self.exited = False

    def __enter__(self) -> _FakeStreamingResponse:
        self.entered = True
        return self.stream

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        del exc_type, exc, exc_tb
        self.exited = True
        self.stream.close()


class _FakeOpenAIClient:
    last_init_kwargs: dict[str, object] | None = None
    queued_parse_responses: list[object] = []
    queued_stream_managers: list[_FakeStreamingManager] = []
    captured_parse_payloads: list[dict[str, object]] = []
    captured_stream_payloads: list[dict[str, object]] = []

    def __init__(self, **kwargs: object) -> None:
        type(self).last_init_kwargs = dict(kwargs)
        completions = type(
            "_BetaCompletionsNamespace",
            (),
            {
                "parse": self._parse,
                "stream": self._stream,
            },
        )()
        self.beta = type(
            "_BetaNamespace",
            (),
            {
                "chat": type(
                    "_BetaChatNamespace",
                    (),
                    {"completions": completions},
                )()
            },
        )()
        self.chat = type(
            "_ChatNamespace",
            (),
            {
                "completions": type(
                    "_CompletionsNamespace",
                    (),
                    {"create": self._unexpected_create},
                )()
            },
        )()

    @classmethod
    def reset(cls) -> None:
        cls.last_init_kwargs = None
        cls.queued_parse_responses = []
        cls.queued_stream_managers = []
        cls.captured_parse_payloads = []
        cls.captured_stream_payloads = []

    def _unexpected_create(self, **payload: object) -> object:
        del payload
        raise AssertionError("Raw chat.completions.create should not be used")

    def _parse(self, **payload: object) -> object:
        type(self).captured_parse_payloads.append(dict(payload))
        if not type(self).queued_parse_responses:
            raise AssertionError("No queued fake parsed response")
        return type(self).queued_parse_responses.pop(0)

    def _stream(self, **payload: object) -> _FakeStreamingManager:
        type(self).captured_stream_payloads.append(dict(payload))
        if not type(self).queued_stream_managers:
            raise AssertionError("No queued fake streaming manager")
        return type(self).queued_stream_managers.pop(0)


def test_validate_payload_and_json_text() -> None:
    parsed = validate_payload(_SimpleModel, {"value": 3})
    assert parsed.value == 3
    assert validate_json_text(_SimpleModel, '{"value":4}').value == 4

    with pytest.raises(ValidationError):
        validate_payload(_SimpleModel, {"value": "bad"})

    with pytest.raises(ValidationError, match="Structured JSON validation failed"):
        validate_json_text(_SimpleModel, "not-json")


def test_mock_client_generates_structured_defaults() -> None:
    client = MockLLMClient()
    response = client.generate_structured(
        StructuredGenerationRequest(
            system_prompt="system",
            user_prompt="user",
            response_model=ChatFinalResponse,
            model_name="mock-chat",
        )
    )

    assert isinstance(response.content, ChatFinalResponse)
    assert response.content.answer.startswith("TBD:")


def test_mock_client_supports_chat_turns_and_streams() -> None:
    tool_call = ChatToolCall(call_id="1", tool_name="list_directory", arguments={})
    client = MockLLMClient(
        chat_canned_turns=[
            ChatTurnResponse(
                assistant_message=ChatMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[tool_call],
                ),
                tool_calls=[tool_call],
                finish_reason="tool_calls",
            )
        ]
    )

    turn = client.generate_chat_turn(
        ChatTurnRequest(
            messages=[ChatMessage(role="user", content="hi")],
            response_model=ChatFinalResponse,
            model_name="mock-chat",
        )
    )
    assert turn.finish_reason == "tool_calls"

    stream_client = MockLLMClient(
        chat_canned_streams=[
            [
                ChatAssistantDeltaEvent(
                    delta_text="Hello",
                    accumulated_text="Hello",
                ),
                ChatFinalResponseEvent(
                    assistant_message=ChatMessage(role="assistant", content="Hello"),
                    final_response=ChatFinalResponse(answer="Hello"),
                ),
            ]
        ]
    )
    stream = stream_client.stream_chat_turn(
        ChatTurnRequest(
            messages=[ChatMessage(role="user", content="hi")],
            response_model=ChatFinalResponse,
            model_name="mock-chat",
        )
    )
    events = list(stream)
    assert events[0].event_type == "assistant_delta"
    assert events[-1].event_type == "final_response"


def test_mock_stream_cancel_yields_interrupted_event() -> None:
    client = MockLLMClient(
        chat_canned_streams=[
            [
                ChatAssistantDeltaEvent(
                    delta_text="Hello",
                    accumulated_text="Hello",
                ),
            ]
        ]
    )
    stream = client.stream_chat_turn(
        ChatTurnRequest(
            messages=[ChatMessage(role="user", content="hi")],
            response_model=ChatFinalResponse,
            model_name="mock-chat",
        )
    )
    stream.cancel()
    events = list(stream)
    assert events[-1].event_type == "interrupted"


def test_create_chat_llm_client_supports_mock_and_ollama() -> None:
    _FakeOpenAIClient.reset()
    config = ChatLLMConfig(provider="mock", model_name="mock-chat")
    assert isinstance(create_chat_llm_client(config), MockLLMClient)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            "engllm_chat.llm.openai_compatible.OpenAI",
            _FakeOpenAIClient,
        )
        ollama_client = create_chat_llm_client(
            ChatLLMConfig(provider="ollama", model_name="qwen"),
            provider="ollama",
            model_name="qwen3",
            api_base_url="http://localhost:11434",
            timeout_seconds=12.0,
        )
        assert isinstance(ollama_client, OpenAICompatibleChatLLMClient)
        assert _FakeOpenAIClient.last_init_kwargs == {
            "api_key": "ollama",
            "base_url": "http://localhost:11434/v1",
            "timeout": 12.0,
        }

    with pytest.raises(LLMError, match="Unsupported chat LLM provider"):
        create_chat_llm_client(config, provider="bad-provider")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("provider", "env_var", "base_url"),
    [
        ("openai", "OPENAI_API_KEY", "https://api.openai.com/v1"),
        ("xai", "XAI_API_KEY", "https://api.x.ai/v1"),
        ("anthropic", "ANTHROPIC_API_KEY", "https://api.anthropic.com/v1/"),
        (
            "gemini",
            "GEMINI_API_KEY",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        ),
    ],
)
def test_create_chat_llm_client_supports_hosted_openai_compatible_providers(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    env_var: str,
    base_url: str,
) -> None:
    monkeypatch.setattr("engllm_chat.llm.openai_compatible.OpenAI", _FakeOpenAIClient)
    monkeypatch.setenv(env_var, "test-token")
    _FakeOpenAIClient.reset()

    client = create_chat_llm_client(
        ChatLLMConfig(provider=provider, model_name="hosted-model"),  # type: ignore[arg-type]
        provider=provider,  # type: ignore[arg-type]
        model_name="override-model",
        timeout_seconds=12.0,
    )

    assert isinstance(client, OpenAICompatibleChatLLMClient)
    assert _FakeOpenAIClient.last_init_kwargs == {
        "api_key": "test-token",
        "base_url": base_url,
        "timeout": 12.0,
    }


def test_create_chat_llm_client_supports_hosted_base_url_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("engllm_chat.llm.openai_compatible.OpenAI", _FakeOpenAIClient)
    monkeypatch.setenv("OPENAI_API_KEY", "test-token")
    _FakeOpenAIClient.reset()

    client = create_chat_llm_client(
        ChatLLMConfig(provider="openai", model_name="gpt-test"),
        api_base_url="https://proxy.example/v1",
    )

    assert isinstance(client, OpenAICompatibleChatLLMClient)
    assert _FakeOpenAIClient.last_init_kwargs is not None
    assert _FakeOpenAIClient.last_init_kwargs["base_url"] == "https://proxy.example/v1"


def test_openai_compatible_client_requires_configured_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("engllm_chat.llm.openai_compatible.OpenAI", _FakeOpenAIClient)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(LLMError, match="OPENAI_API_KEY is not configured"):
        OpenAICompatibleChatLLMClient(
            model_name="gpt-test",
            provider_name="openai",
            api_key_env_var="OPENAI_API_KEY",
            base_url="https://api.openai.com/v1",
        )


def test_openai_compatible_generate_chat_turn_parses_final_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("engllm_chat.llm.openai_compatible.OpenAI", _FakeOpenAIClient)
    _FakeOpenAIClient.reset()
    _FakeOpenAIClient.queued_parse_responses = [
        _FakeChatCompletionResponse(
            message=_FakeMessage(
                content="Hosted done",
                parsed=ChatFinalResponse(answer="Hosted done"),
            ),
            usage=_FakeUsage(prompt_tokens=3, completion_tokens=5, total_tokens=8),
        )
    ]

    client = OpenAICompatibleChatLLMClient(
        model_name="gpt-test",
        provider_name="openai",
        api_key_env_var=None,
        api_key="secret",
        base_url="https://api.openai.com/v1",
    )

    response = client.generate_chat_turn(
        ChatTurnRequest(
            messages=[ChatMessage(role="user", content="hello")],
            response_model=ChatFinalResponse,
            model_name="gpt-test",
        )
    )

    assert response.finish_reason == "final_response"
    assert response.final_response == ChatFinalResponse(answer="Hosted done")
    assert response.token_usage is not None
    assert response.token_usage.total_tokens == 8
    assert response.raw_text == "Hosted done"
    assert _FakeOpenAIClient.captured_parse_payloads[-1]["model"] == "gpt-test"
    assert (
        _FakeOpenAIClient.captured_parse_payloads[-1]["response_format"]
        is ChatFinalResponse
    )


def test_openai_compatible_generate_chat_turn_parses_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("engllm_chat.llm.openai_compatible.OpenAI", _FakeOpenAIClient)
    tool_builder_calls: list[tuple[type[BaseModel], str, str]] = []

    def _fake_pydantic_function_tool(
        model: type[BaseModel],
        *,
        name: str,
        description: str,
    ) -> dict[str, object]:
        tool_builder_calls.append((model, name, description))
        return {
            "type": "function",
            "function": {"name": name, "description": description, "strict": True},
        }

    monkeypatch.setattr(
        "engllm_chat.llm.openai_compatible.pydantic_function_tool",
        _fake_pydantic_function_tool,
    )
    _FakeOpenAIClient.reset()
    _FakeOpenAIClient.queued_parse_responses = [
        _FakeChatCompletionResponse(
            message=_FakeMessage(
                content="",
                tool_calls=[
                    _FakeToolCall(
                        "call-1",
                        "read_file",
                        '{"path":"README.md"}',
                        parsed_arguments=_ReadFileArgs(path="README.md"),
                    )
                ],
            )
        )
    ]

    client = OpenAICompatibleChatLLMClient(
        model_name="gpt-test",
        provider_name="xai",
        api_key_env_var=None,
        api_key="secret",
        base_url="https://api.x.ai/v1",
    )

    response = client.generate_chat_turn(
        ChatTurnRequest(
            messages=[ChatMessage(role="user", content="inspect repo")],
            response_model=ChatFinalResponse,
            model_name="gpt-test",
            tools=[
                ChatToolDefinition(
                    name="read_file",
                    description="Read one file",
                    input_schema={
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                    argument_model=_ReadFileArgs,
                )
            ],
        )
    )

    assert response.finish_reason == "tool_calls"
    assert response.tool_calls[0].tool_name == "read_file"
    assert response.tool_calls[0].arguments == {"path": "README.md"}
    assert tool_builder_calls == [(_ReadFileArgs, "read_file", "Read one file")]
    assert _FakeOpenAIClient.captured_parse_payloads[-1]["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read one file",
                "strict": True,
            },
        }
    ]


def test_openai_compatible_stream_chat_turn_supports_final_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("engllm_chat.llm.openai_compatible.OpenAI", _FakeOpenAIClient)
    _FakeOpenAIClient.reset()
    final_stream = _FakeStreamingResponse(
        events=[
            _FakeStreamingEvent("content.delta", delta="Hello"),
            _FakeStreamingEvent(
                "content.delta",
                delta=" world",
                chunk=type(
                    "_Chunk",
                    (),
                    {
                        "usage": _FakeUsage(
                            prompt_tokens=2,
                            completion_tokens=4,
                            total_tokens=6,
                        )
                    },
                )(),
            ),
        ],
        final_completion=_FakeChatCompletionResponse(
            message=_FakeMessage(
                content="Hello world",
                parsed=ChatFinalResponse(answer="Hello world"),
            ),
            usage=_FakeUsage(prompt_tokens=2, completion_tokens=4, total_tokens=6),
        ),
    )
    final_manager = _FakeStreamingManager(final_stream)
    _FakeOpenAIClient.queued_stream_managers = [final_manager]

    client = OpenAICompatibleChatLLMClient(
        model_name="gpt-test",
        provider_name="gemini",
        api_key_env_var=None,
        api_key="secret",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    final_events = list(
        client.stream_chat_turn(
            ChatTurnRequest(
                messages=[ChatMessage(role="user", content="hello")],
                response_model=ChatFinalResponse,
                model_name="gpt-test",
            )
        )
    )
    assert isinstance(final_events[0], ChatAssistantDeltaEvent)
    assert isinstance(final_events[-1], ChatFinalResponseEvent)
    assert final_events[-1].final_response == ChatFinalResponse(answer="Hello world")
    assert (
        _FakeOpenAIClient.captured_stream_payloads[0]["response_format"]
        is ChatFinalResponse
    )
    assert _FakeOpenAIClient.captured_stream_payloads[0]["stream_options"] == {
        "include_usage": True
    }
    assert final_manager.entered is True
    assert final_manager.exited is True
    assert final_stream.closed is True


def test_openai_compatible_stream_chat_turn_supports_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("engllm_chat.llm.openai_compatible.OpenAI", _FakeOpenAIClient)
    _FakeOpenAIClient.reset()
    tool_stream = _FakeStreamingResponse(
        events=[_FakeStreamingEvent("content.delta", delta="Searching...")],
        final_completion=_FakeChatCompletionResponse(
            message=_FakeMessage(
                content="Searching...",
                tool_calls=[
                    _FakeToolCall(
                        "call-1",
                        "search_text",
                        '{"query":"TODO"}',
                        parsed_arguments={"query": "TODO"},
                    )
                ],
            )
        ),
    )
    _FakeOpenAIClient.queued_stream_managers = [_FakeStreamingManager(tool_stream)]

    client = OpenAICompatibleChatLLMClient(
        model_name="gpt-test",
        provider_name="gemini",
        api_key_env_var=None,
        api_key="secret",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    tool_events = list(
        client.stream_chat_turn(
            ChatTurnRequest(
                messages=[ChatMessage(role="user", content="find todos")],
                response_model=ChatFinalResponse,
                model_name="gpt-test",
            )
        )
    )
    assert isinstance(tool_events[0], ChatAssistantDeltaEvent)
    assert isinstance(tool_events[-1], ChatToolCallsEvent)
    assert tool_events[-1].tool_calls[0].tool_name == "search_text"
    assert tool_events[-1].tool_calls[0].arguments == {"query": "TODO"}


def test_openai_compatible_stream_chat_turn_supports_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("engllm_chat.llm.openai_compatible.OpenAI", _FakeOpenAIClient)
    _FakeOpenAIClient.reset()
    cancel_stream = _FakeStreamingResponse(
        events=[_FakeStreamingEvent("content.delta", delta="partial")],
        final_completion=_FakeChatCompletionResponse(
            message=_FakeMessage(
                content="unused",
                parsed=ChatFinalResponse(answer="unused"),
            )
        ),
    )
    cancel_manager = _FakeStreamingManager(cancel_stream)
    _FakeOpenAIClient.queued_stream_managers = [cancel_manager]

    client = OpenAICompatibleChatLLMClient(
        model_name="gpt-test",
        provider_name="gemini",
        api_key_env_var=None,
        api_key="secret",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    interrupted = client.stream_chat_turn(
        ChatTurnRequest(
            messages=[ChatMessage(role="user", content="cancel")],
            response_model=ChatFinalResponse,
            model_name="gpt-test",
        )
    )
    interrupted.cancel()
    interrupted_events = list(interrupted)
    assert isinstance(interrupted_events[-1], ChatInterruptedEvent)
    assert cancel_stream.closed is True
    assert cancel_manager.exited is True


def test_ollama_generate_chat_turn_parses_final_response(monkeypatch) -> None:
    monkeypatch.setattr("engllm_chat.llm.openai_compatible.OpenAI", _FakeOpenAIClient)
    _FakeOpenAIClient.reset()
    _FakeOpenAIClient.queued_parse_responses = [
        _FakeChatCompletionResponse(
            message=_FakeMessage(
                content="Done",
                parsed=ChatFinalResponse(answer="Done"),
            ),
            usage=_FakeUsage(prompt_tokens=4, completion_tokens=6, total_tokens=10),
        )
    ]
    client = OpenAICompatibleChatLLMClient(
        model_name="qwen",
        provider_name="ollama",
        api_key_env_var=None,
        api_key="ollama",
        base_url=normalize_ollama_base_url("http://localhost:11434"),
    )
    response = client.generate_chat_turn(
        ChatTurnRequest(
            messages=[ChatMessage(role="user", content="hello")],
            response_model=ChatFinalResponse,
            model_name="qwen",
        )
    )

    assert response.finish_reason == "final_response"
    assert response.final_response == ChatFinalResponse(answer="Done")
    assert response.token_usage is not None
    assert response.token_usage.total_tokens == 10
    assert _FakeOpenAIClient.last_init_kwargs is not None
    assert _FakeOpenAIClient.last_init_kwargs["base_url"] == "http://localhost:11434/v1"
    assert _FakeOpenAIClient.last_init_kwargs["api_key"] == "ollama"
    assert (
        _FakeOpenAIClient.captured_parse_payloads[-1]["response_format"]
        is ChatFinalResponse
    )


def test_ollama_generate_chat_turn_parses_tool_calls(monkeypatch) -> None:
    monkeypatch.setattr("engllm_chat.llm.openai_compatible.OpenAI", _FakeOpenAIClient)
    _FakeOpenAIClient.reset()
    _FakeOpenAIClient.queued_parse_responses = [
        _FakeChatCompletionResponse(
            message=_FakeMessage(
                content="",
                tool_calls=[
                    _FakeToolCall(
                        "call-1",
                        "list_directory",
                        '{"path":"."}',
                        parsed_arguments={"path": "."},
                    )
                ],
            )
        )
    ]
    client = OpenAICompatibleChatLLMClient(
        model_name="qwen",
        provider_name="ollama",
        api_key_env_var=None,
        api_key="ollama",
        base_url=normalize_ollama_base_url(None),
    )
    response = client.generate_chat_turn(
        ChatTurnRequest(
            messages=[ChatMessage(role="user", content="hello")],
            response_model=ChatFinalResponse,
            model_name="qwen",
        )
    )

    assert response.finish_reason == "tool_calls"
    assert response.tool_calls[0].tool_name == "list_directory"


def test_ollama_stream_chat_turn_supports_final_and_cancel(monkeypatch) -> None:
    monkeypatch.setattr("engllm_chat.llm.openai_compatible.OpenAI", _FakeOpenAIClient)
    _FakeOpenAIClient.reset()
    final_stream = _FakeStreamingResponse(
        events=[_FakeStreamingEvent("content.delta", delta="Hi")],
        final_completion=_FakeChatCompletionResponse(
            message=_FakeMessage(
                content="Hi",
                parsed=ChatFinalResponse(answer="Hi"),
            ),
            usage=_FakeUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
        ),
    )
    cancel_stream = _FakeStreamingResponse(
        events=[_FakeStreamingEvent("content.delta", delta="partial")],
        final_completion=_FakeChatCompletionResponse(
            message=_FakeMessage(
                content="unused",
                parsed=ChatFinalResponse(answer="unused"),
            )
        ),
    )
    _FakeOpenAIClient.queued_stream_managers = [
        _FakeStreamingManager(final_stream),
        _FakeStreamingManager(cancel_stream),
    ]
    client = OpenAICompatibleChatLLMClient(
        model_name="qwen",
        provider_name="ollama",
        api_key_env_var=None,
        api_key="ollama",
        base_url=normalize_ollama_base_url(None),
    )
    stream = client.stream_chat_turn(
        ChatTurnRequest(
            messages=[ChatMessage(role="user", content="hello")],
            response_model=ChatFinalResponse,
            model_name="qwen",
        )
    )
    events = list(stream)
    assert events[0].event_type == "assistant_delta"
    assert events[-1].event_type == "final_response"

    cancel_stream = client.stream_chat_turn(
        ChatTurnRequest(
            messages=[ChatMessage(role="user", content="hello")],
            response_model=ChatFinalResponse,
            model_name="qwen",
        )
    )
    cancel_stream.cancel()
    cancel_events = list(cancel_stream)
    assert cancel_events[-1].event_type == "interrupted"
    assert _FakeOpenAIClient.last_init_kwargs is not None


def test_probe_main_requires_api_key(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        probe_main(["--base-url", "http://localhost:11434/v1"])

    captured = capsys.readouterr()
    assert "API key is required" in captured.err
