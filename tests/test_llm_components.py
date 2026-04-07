"""Focused tests for standalone chat LLM components."""

from __future__ import annotations

import logging

import pytest
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from engllm_chat.domain.errors import LLMError, ValidationError
from engllm_chat.domain.models import (
    ChatFinalResponse,
    ChatLLMConfig,
    ChatMessage,
    ChatToolCall,
)
from engllm_chat.llm.base import (
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


class _FakeMessage:
    def __init__(
        self,
        *,
        content: str | None = None,
        parsed: object | None = None,
    ) -> None:
        self.content = content
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


class _FakeOpenAIClient:
    last_init_kwargs: dict[str, object] | None = None
    queued_parse_responses: list[object] = []
    captured_parse_payloads: list[dict[str, object]] = []

    def __init__(self, **kwargs: object) -> None:
        type(self).last_init_kwargs = dict(kwargs)
        completions = type(
            "_BetaCompletionsNamespace",
            (),
            {
                "parse": self._parse,
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
        cls.captured_parse_payloads = []

    def _unexpected_create(self, **payload: object) -> object:
        del payload
        raise AssertionError("Raw chat.completions.create should not be used")

    def _parse(self, **payload: object) -> object:
        type(self).captured_parse_payloads.append(dict(payload))
        if not type(self).queued_parse_responses:
            raise AssertionError("No queued fake parsed response")
        return type(self).queued_parse_responses.pop(0)


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


def test_mock_client_supports_chat_turns() -> None:
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


def test_openai_compatible_client_requires_sdk_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("engllm_chat.llm.openai_compatible.OpenAI", None)

    with pytest.raises(LLMError, match="SDK dependencies are unavailable"):
        OpenAICompatibleChatLLMClient(
            model_name="gpt-test",
            provider_name="openai",
            api_key_env_var=None,
            api_key="secret",
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
                parsed={
                    "action": {
                        "kind": "final_response",
                        "response": {"answer": "Hosted done"},
                    }
                },
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
    response_format = _FakeOpenAIClient.captured_parse_payloads[-1]["response_format"]
    assert response_format is not ChatFinalResponse
    assert "action" in response_format.model_fields
    assert "tools" not in _FakeOpenAIClient.captured_parse_payloads[-1]


def test_openai_compatible_generate_chat_turn_retries_sdk_schema_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("engllm_chat.llm.openai_compatible.OpenAI", _FakeOpenAIClient)
    _FakeOpenAIClient.reset()

    class _SchemaErrorThenSuccessClient(_FakeOpenAIClient):
        attempts = 0

        @classmethod
        def reset(cls) -> None:
            super().reset()
            cls.attempts = 0

        def _parse(self, **payload: object) -> object:
            type(self).captured_parse_payloads.append(dict(payload))
            type(self).attempts += 1
            if type(self).attempts == 1:
                try:
                    ChatFinalResponse.model_validate({"answer": ""})
                except PydanticValidationError as exc:
                    raise exc
            return _FakeChatCompletionResponse(
                message=_FakeMessage(
                    content="Hosted done after retry",
                    parsed={
                        "action": {
                            "kind": "final_response",
                            "response": {"answer": "Hosted done after retry"},
                        }
                    },
                ),
                usage=_FakeUsage(prompt_tokens=2, completion_tokens=4, total_tokens=6),
            )

    monkeypatch.setattr(
        "engllm_chat.llm.openai_compatible.OpenAI",
        _SchemaErrorThenSuccessClient,
    )
    _SchemaErrorThenSuccessClient.reset()

    client = OpenAICompatibleChatLLMClient(
        model_name="gpt-test",
        provider_name="gemini",
        api_key_env_var=None,
        api_key="secret",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    response = client.generate_chat_turn(
        ChatTurnRequest(
            messages=[ChatMessage(role="user", content="hello")],
            response_model=ChatFinalResponse,
            model_name="gemini-test",
        )
    )

    assert response.finish_reason == "final_response"
    assert response.final_response == ChatFinalResponse(
        answer="Hosted done after retry"
    )
    assert response.token_usage is not None
    assert response.token_usage.total_tokens == 6
    assert len(_SchemaErrorThenSuccessClient.captured_parse_payloads) == 2
    retry_messages = _SchemaErrorThenSuccessClient.captured_parse_payloads[1][
        "messages"
    ]
    assert isinstance(retry_messages, list)
    assert "did not satisfy the required structured schema" in str(
        retry_messages[-1]["content"]
    )


def test_openai_compatible_verbose_logging_emits_request_and_response_messages(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr("engllm_chat.llm.openai_compatible.OpenAI", _FakeOpenAIClient)
    _FakeOpenAIClient.reset()
    _FakeOpenAIClient.queued_parse_responses = [
        _FakeChatCompletionResponse(
            message=_FakeMessage(
                content="Hosted done",
                parsed={
                    "action": {
                        "kind": "final_response",
                        "response": {"answer": "Hosted done"},
                    }
                },
            )
        )
    ]

    client = OpenAICompatibleChatLLMClient(
        model_name="gpt-test",
        provider_name="openai",
        api_key_env_var=None,
        api_key="secret",
        base_url="https://api.openai.com/v1",
        verbose_logging=True,
    )

    with caplog.at_level(logging.INFO, logger="engllm_chat.llm.openai_compatible"):
        client.generate_chat_turn(
            ChatTurnRequest(
                messages=[ChatMessage(role="user", content="hello from user")],
                response_model=ChatFinalResponse,
                model_name="gpt-test",
            )
        )

    log_text = caplog.text
    assert "LLM request -> provider=openai model=gpt-test attempt=1" in log_text
    assert '"content": "hello from user"' in log_text
    assert "LLM response <- provider=openai model=gpt-test" in log_text
    assert '"answer": "Hosted done"' in log_text


def test_openai_compatible_generate_chat_turn_parses_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("engllm_chat.llm.openai_compatible.OpenAI", _FakeOpenAIClient)
    _FakeOpenAIClient.reset()
    _FakeOpenAIClient.queued_parse_responses = [
        _FakeChatCompletionResponse(
            message=_FakeMessage(
                content="",
                parsed={
                    "action": {
                        "kind": "tool_request",
                        "tool_name": "read_file",
                        "arguments": {"path": "README.md"},
                    }
                },
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
    assert "tools" not in _FakeOpenAIClient.captured_parse_payloads[-1]
    assert "tool_request" in response.raw_text


def test_ollama_generate_chat_turn_parses_final_response(monkeypatch) -> None:
    monkeypatch.setattr("engllm_chat.llm.openai_compatible.OpenAI", _FakeOpenAIClient)
    _FakeOpenAIClient.reset()
    _FakeOpenAIClient.queued_parse_responses = [
        _FakeChatCompletionResponse(
            message=_FakeMessage(
                content="Done",
                parsed={
                    "action": {
                        "kind": "final_response",
                        "response": {"answer": "Done"},
                    }
                },
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
            tools=[
                ChatToolDefinition(
                    name="list_directory",
                    description="List one directory",
                    input_schema={
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                    },
                )
            ],
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
        "action"
        in _FakeOpenAIClient.captured_parse_payloads[-1]["response_format"].model_fields
    )


def test_ollama_generate_chat_turn_parses_tool_calls(monkeypatch) -> None:
    monkeypatch.setattr("engllm_chat.llm.openai_compatible.OpenAI", _FakeOpenAIClient)
    _FakeOpenAIClient.reset()
    _FakeOpenAIClient.queued_parse_responses = [
        _FakeChatCompletionResponse(
            message=_FakeMessage(
                content="",
                parsed={
                    "action": {
                        "kind": "tool_request",
                        "tool_name": "list_directory",
                        "arguments": {"path": "."},
                    }
                },
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
            tools=[
                ChatToolDefinition(
                    name="list_directory",
                    description="List one directory",
                    input_schema={
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                    },
                )
            ],
        )
    )

    assert response.finish_reason == "tool_calls"
    assert response.tool_calls[0].tool_name == "list_directory"


def test_openai_compatible_generate_chat_turn_retries_after_schema_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("engllm_chat.llm.openai_compatible.OpenAI", _FakeOpenAIClient)
    _FakeOpenAIClient.reset()
    _FakeOpenAIClient.queued_parse_responses = [
        _FakeChatCompletionResponse(
            message=_FakeMessage(content="not-json", parsed=None),
        ),
        _FakeChatCompletionResponse(
            message=_FakeMessage(
                content="Recovered",
                parsed={
                    "action": {
                        "kind": "final_response",
                        "response": {"answer": "Recovered"},
                    }
                },
            ),
            usage=_FakeUsage(prompt_tokens=6, completion_tokens=4, total_tokens=10),
        ),
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
    assert response.final_response == ChatFinalResponse(answer="Recovered")
    assert len(_FakeOpenAIClient.captured_parse_payloads) == 2
    second_messages = _FakeOpenAIClient.captured_parse_payloads[1]["messages"]
    assert isinstance(second_messages, list)
    assert "did not satisfy the required structured schema" in str(
        second_messages[-1]["content"]
    )
    assert "malformed action JSON" in str(second_messages[-1]["content"])


def test_openai_compatible_generate_chat_turn_parses_json_fenced_raw_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("engllm_chat.llm.openai_compatible.OpenAI", _FakeOpenAIClient)
    _FakeOpenAIClient.reset()
    _FakeOpenAIClient.queued_parse_responses = [
        _FakeChatCompletionResponse(
            message=_FakeMessage(
                content=(
                    "```json\n"
                    '{"action":{"kind":"final_response","response":{"answer":"Recovered from fenced JSON"}}}\n'
                    "```"
                ),
                parsed=None,
            ),
            usage=_FakeUsage(prompt_tokens=6, completion_tokens=4, total_tokens=10),
        ),
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
    assert response.final_response == ChatFinalResponse(
        answer="Recovered from fenced JSON"
    )
    assert len(_FakeOpenAIClient.captured_parse_payloads) == 1


def test_openai_compatible_generate_chat_turn_raises_after_three_schema_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("engllm_chat.llm.openai_compatible.OpenAI", _FakeOpenAIClient)
    _FakeOpenAIClient.reset()
    _FakeOpenAIClient.queued_parse_responses = [
        _FakeChatCompletionResponse(message=_FakeMessage(content="bad-1", parsed=None)),
        _FakeChatCompletionResponse(message=_FakeMessage(content="bad-2", parsed=None)),
        _FakeChatCompletionResponse(message=_FakeMessage(content="bad-3", parsed=None)),
    ]

    client = OpenAICompatibleChatLLMClient(
        model_name="gpt-test",
        provider_name="openai",
        api_key_env_var=None,
        api_key="secret",
        base_url="https://api.openai.com/v1",
    )

    with pytest.raises(
        LLMError,
        match="failed to return a valid structured response after 3 attempts",
    ):
        client.generate_chat_turn(
            ChatTurnRequest(
                messages=[ChatMessage(role="user", content="hello")],
                response_model=ChatFinalResponse,
                model_name="gpt-test",
            )
        )

    assert len(_FakeOpenAIClient.captured_parse_payloads) == 3


def test_openai_compatible_generate_chat_turn_surfaces_request_and_protocol_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("engllm_chat.llm.openai_compatible.OpenAI", _FakeOpenAIClient)

    class _RaisingClient(_FakeOpenAIClient):
        def _parse(self, **payload: object) -> object:
            del payload
            raise RuntimeError("boom")

    monkeypatch.setattr("engllm_chat.llm.openai_compatible.OpenAI", _RaisingClient)
    client = OpenAICompatibleChatLLMClient(
        model_name="gpt-test",
        provider_name="openai",
        api_key_env_var=None,
        api_key="secret",
        base_url="https://api.openai.com/v1",
    )
    with pytest.raises(LLMError, match="request failed: boom"):
        client.generate_chat_turn(
            ChatTurnRequest(
                messages=[ChatMessage(role="user", content="hello")],
                response_model=ChatFinalResponse,
                model_name="gpt-test",
            )
        )

    class _NoChoicesClient(_FakeOpenAIClient):
        def _parse(self, **payload: object) -> object:
            del payload
            return type(
                "_NoChoiceResponse",
                (),
                {"choices": [], "usage": None},
            )()

    monkeypatch.setattr("engllm_chat.llm.openai_compatible.OpenAI", _NoChoicesClient)
    client = OpenAICompatibleChatLLMClient(
        model_name="gpt-test",
        provider_name="openai",
        api_key_env_var=None,
        api_key="secret",
        base_url="https://api.openai.com/v1",
    )
    with pytest.raises(LLMError, match="returned no choices"):
        client.generate_chat_turn(
            ChatTurnRequest(
                messages=[ChatMessage(role="user", content="hello")],
                response_model=ChatFinalResponse,
                model_name="gpt-test",
            )
        )

    class _MissingMessageClient(_FakeOpenAIClient):
        def _parse(self, **payload: object) -> object:
            del payload
            return type(
                "_MissingMessageResponse",
                (),
                {"choices": [type("_Choice", (), {"message": None})()], "usage": None},
            )()

    monkeypatch.setattr(
        "engllm_chat.llm.openai_compatible.OpenAI", _MissingMessageClient
    )
    client = OpenAICompatibleChatLLMClient(
        model_name="gpt-test",
        provider_name="openai",
        api_key_env_var=None,
        api_key="secret",
        base_url="https://api.openai.com/v1",
    )
    with pytest.raises(LLMError, match="response missing message"):
        client.generate_chat_turn(
            ChatTurnRequest(
                messages=[ChatMessage(role="user", content="hello")],
                response_model=ChatFinalResponse,
                model_name="gpt-test",
            )
        )


def test_probe_main_requires_api_key(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        probe_main(["--base-url", "http://localhost:11434/v1"])

    captured = capsys.readouterr()
    assert "API key is required" in captured.err
