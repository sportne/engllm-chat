"""Focused tests for standalone chat LLM components."""

from __future__ import annotations

import json
from typing import Any

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
from engllm_chat.llm.ollama import OllamaLLMClient
from engllm_chat.llm.openai_compatible import OpenAICompatibleChatLLMClient
from engllm_chat.probe_openai_api import main as probe_main


class _SimpleModel(BaseModel):
    value: int


class _FakeResponse:
    def __init__(
        self,
        *,
        status: int = 200,
        body: str = "",
        lines: list[str] | None = None,
    ) -> None:
        self.status = status
        self._body = body
        self._lines = [line.encode("utf-8") for line in (lines or [])]
        self.closed = False

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def read(self) -> bytes:
        return self._body.encode("utf-8")

    def readline(self) -> bytes:
        if not self._lines:
            return b""
        return self._lines.pop(0)

    def close(self) -> None:
        self.closed = True


class _FakeFunction:
    def __init__(self, name: str, arguments: str | dict[str, object]) -> None:
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(
        self, call_id: str, name: str, arguments: str | dict[str, object]
    ) -> None:
        self.id = call_id
        self.function = _FakeFunction(name, arguments)


class _FakeMessage:
    def __init__(
        self,
        *,
        content: str | None = None,
        tool_calls: list[_FakeToolCall] | None = None,
    ) -> None:
        self.content = content
        self.tool_calls = tool_calls or []


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


class _FakeDeltaFunction:
    def __init__(self, name: str | None = None, arguments: str | None = None) -> None:
        self.name = name
        self.arguments = arguments


class _FakeDeltaToolCall:
    def __init__(
        self,
        *,
        index: int,
        call_id: str | None = None,
        name: str | None = None,
        arguments: str | None = None,
    ) -> None:
        self.index = index
        self.id = call_id
        self.function = _FakeDeltaFunction(name=name, arguments=arguments)


class _FakeDelta:
    def __init__(
        self,
        *,
        content: str | None = None,
        tool_calls: list[_FakeDeltaToolCall] | None = None,
    ) -> None:
        self.content = content
        self.tool_calls = tool_calls or []


class _FakeStreamingChoice:
    def __init__(self, delta: _FakeDelta, finish_reason: str | None = None) -> None:
        self.delta = delta
        self.finish_reason = finish_reason


class _FakeStreamingChunk:
    def __init__(
        self,
        *,
        delta: _FakeDelta | None = None,
        finish_reason: str | None = None,
        usage: object | None = None,
        include_choices: bool = True,
    ) -> None:
        self.usage = usage
        self.choices = (
            [_FakeStreamingChoice(delta or _FakeDelta(), finish_reason=finish_reason)]
            if include_choices
            else []
        )


class _FakeStreamingResponse:
    def __init__(self, chunks: list[_FakeStreamingChunk]) -> None:
        self._chunks = chunks
        self.closed = False

    def __iter__(self):
        yield from self._chunks

    def close(self) -> None:
        self.closed = True


class _FakeOpenAIClient:
    last_init_kwargs: dict[str, object] | None = None
    queued_responses: list[object] = []
    captured_payloads: list[dict[str, object]] = []

    def __init__(self, **kwargs: object) -> None:
        type(self).last_init_kwargs = dict(kwargs)
        self.chat = type(
            "_ChatNamespace",
            (),
            {
                "completions": type(
                    "_CompletionsNamespace",
                    (),
                    {"create": self._create},
                )()
            },
        )()

    @classmethod
    def reset(cls) -> None:
        cls.last_init_kwargs = None
        cls.queued_responses = []
        cls.captured_payloads = []

    def _create(self, **payload: object) -> object:
        type(self).captured_payloads.append(dict(payload))
        if not type(self).queued_responses:
            raise AssertionError("No queued fake OpenAI response")
        return type(self).queued_responses.pop(0)


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
    config = ChatLLMConfig(provider="mock", model_name="mock-chat")
    assert isinstance(create_chat_llm_client(config), MockLLMClient)

    ollama_client = create_chat_llm_client(
        ChatLLMConfig(provider="ollama", model_name="qwen"),
        provider="ollama",
        model_name="qwen3",
        ollama_base_url="http://localhost:11434",
        timeout_seconds=12.0,
    )
    assert isinstance(ollama_client, OllamaLLMClient)

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
    _FakeOpenAIClient.queued_responses = [
        _FakeChatCompletionResponse(
            message=_FakeMessage(content='{"answer":"Hosted done"}'),
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
    assert _FakeOpenAIClient.captured_payloads[-1]["model"] == "gpt-test"


def test_openai_compatible_generate_chat_turn_parses_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("engllm_chat.llm.openai_compatible.OpenAI", _FakeOpenAIClient)
    _FakeOpenAIClient.reset()
    _FakeOpenAIClient.queued_responses = [
        _FakeChatCompletionResponse(
            message=_FakeMessage(
                content="",
                tool_calls=[
                    _FakeToolCall("call-1", "read_file", '{"path":"README.md"}')
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
                )
            ],
        )
    )

    assert response.finish_reason == "tool_calls"
    assert response.tool_calls[0].tool_name == "read_file"
    assert response.tool_calls[0].arguments == {"path": "README.md"}


def test_openai_compatible_stream_chat_turn_supports_final_tool_calls_and_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("engllm_chat.llm.openai_compatible.OpenAI", _FakeOpenAIClient)
    _FakeOpenAIClient.reset()
    final_stream = _FakeStreamingResponse(
        [
            _FakeStreamingChunk(delta=_FakeDelta(content='{"answer":"Hello')),
            _FakeStreamingChunk(
                delta=_FakeDelta(content=' world"}'),
                finish_reason="stop",
                usage=_FakeUsage(prompt_tokens=2, completion_tokens=4, total_tokens=6),
            ),
        ]
    )
    tool_stream = _FakeStreamingResponse(
        [
            _FakeStreamingChunk(
                delta=_FakeDelta(
                    tool_calls=[
                        _FakeDeltaToolCall(
                            index=0,
                            call_id="call-1",
                            name="search_text",
                            arguments='{"query":"',
                        )
                    ]
                )
            ),
            _FakeStreamingChunk(
                delta=_FakeDelta(
                    tool_calls=[_FakeDeltaToolCall(index=0, arguments='TODO"}')]
                ),
                finish_reason="tool_calls",
            ),
        ]
    )
    cancel_stream = _FakeStreamingResponse(
        [
            _FakeStreamingChunk(delta=_FakeDelta(content="partial")),
        ]
    )
    _FakeOpenAIClient.queued_responses = [final_stream, tool_stream, cancel_stream]

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

    tool_events = list(
        client.stream_chat_turn(
            ChatTurnRequest(
                messages=[ChatMessage(role="user", content="find todos")],
                response_model=ChatFinalResponse,
                model_name="gpt-test",
            )
        )
    )
    assert isinstance(tool_events[-1], ChatToolCallsEvent)
    assert tool_events[-1].tool_calls[0].tool_name == "search_text"
    assert tool_events[-1].tool_calls[0].arguments == {"query": "TODO"}

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


def test_ollama_generate_chat_turn_parses_final_response(monkeypatch) -> None:
    payload = {
        "message": {"content": '{"answer":"Done","citations":[]}'},
        "prompt_eval_count": 4,
        "eval_count": 6,
    }

    def _fake_urlopen(request_obj: Any, timeout: float) -> _FakeResponse:
        assert timeout == 60.0
        assert request_obj.full_url.endswith("/api/chat")
        return _FakeResponse(body=json.dumps(payload))

    monkeypatch.setattr("engllm_chat.llm.ollama.request_lib.urlopen", _fake_urlopen)
    client = OllamaLLMClient(model_name="qwen", base_url="http://localhost:11434")
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


def test_ollama_generate_chat_turn_parses_tool_calls(monkeypatch) -> None:
    payload = {
        "message": {
            "content": "",
            "tool_calls": [
                {"function": {"name": "list_directory", "arguments": {"path": "."}}}
            ],
        }
    }

    monkeypatch.setattr(
        "engllm_chat.llm.ollama.request_lib.urlopen",
        lambda *_args, **_kwargs: _FakeResponse(body=json.dumps(payload)),
    )

    client = OllamaLLMClient(model_name="qwen")
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
    lines = [
        json.dumps({"message": {"content": '{"answer":"Hi"'}, "done": False}),
        json.dumps(
            {
                "message": {"content": "}", "tool_calls": []},
                "done": True,
                "prompt_eval_count": 1,
                "eval_count": 2,
            }
        ),
    ]
    monkeypatch.setattr(
        "engllm_chat.llm.ollama.request_lib.urlopen",
        lambda *_args, **_kwargs: _FakeResponse(lines=lines),
    )

    client = OllamaLLMClient(model_name="qwen")
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

    response = _FakeResponse(
        lines=[json.dumps({"message": {"content": "partial"}, "done": False})]
    )
    monkeypatch.setattr(
        "engllm_chat.llm.ollama.request_lib.urlopen",
        lambda *_args, **_kwargs: response,
    )
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
    assert response.closed is True


def test_probe_main_requires_api_key(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        probe_main(["--base-url", "http://localhost:11434/v1"])

    captured = capsys.readouterr()
    assert "API key is required" in captured.err
