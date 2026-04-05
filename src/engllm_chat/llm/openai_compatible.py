"""OpenAI-compatible chat provider implementation."""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator
from typing import Any, Protocol

from engllm_chat.domain.errors import LLMError
from engllm_chat.domain.models import ChatMessage, ChatTokenUsage, ChatToolCall
from engllm_chat.llm.base import (
    ChatAssistantDeltaEvent,
    ChatFinalResponseEvent,
    ChatInterruptedEvent,
    ChatToolCallsEvent,
    ChatToolDefinition,
    ChatTurnRequest,
    ChatTurnResponse,
    ChatTurnStream,
    validate_json_text,
)

_openai_sdk: Any = None
try:
    from openai import OpenAI as _OpenAIClient

    _openai_sdk = _OpenAIClient
except Exception:  # pragma: no cover - optional dependency
    pass

OpenAI: Any = _openai_sdk


def _serialize_chat_message(message: ChatMessage) -> dict[str, object]:
    if message.role == "tool":
        if message.tool_result is None:
            raise LLMError("Tool chat message missing tool_result")
        return {
            "role": "tool",
            "tool_call_id": message.tool_result.call_id,
            "content": json.dumps(message.tool_result.model_dump(mode="json")),
        }

    payload: dict[str, object] = {"role": message.role}
    if message.content is not None:
        payload["content"] = message.content
    elif message.role in {"system", "user"}:
        payload["content"] = ""

    if message.role == "assistant" and message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": tool_call.call_id,
                "type": "function",
                "function": {
                    "name": tool_call.tool_name,
                    "arguments": json.dumps(tool_call.arguments, sort_keys=True),
                },
            }
            for tool_call in message.tool_calls
        ]
    return payload


def _serialize_tool_definition(tool: ChatToolDefinition) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        },
    }


def _extract_message_text(message: Any) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if not isinstance(content, list):
        return ""

    text_parts: list[str] = []
    for item in content:
        if isinstance(item, dict):
            text_value = item.get("text")
            if isinstance(text_value, str):
                text_parts.append(text_value)
            continue

        text_value = getattr(item, "text", None)
        if isinstance(text_value, str):
            text_parts.append(text_value)
    return "".join(text_parts)


def _extract_token_usage(response: Any) -> ChatTokenUsage | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None

    input_tokens = getattr(usage, "prompt_tokens", 0)
    output_tokens = getattr(usage, "completion_tokens", 0)
    total_tokens = getattr(usage, "total_tokens", None)
    if not isinstance(input_tokens, int):
        input_tokens = 0
    if not isinstance(output_tokens, int):
        output_tokens = 0
    if not isinstance(total_tokens, int | type(None)):
        total_tokens = None
    return ChatTokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def _parse_tool_arguments(raw_arguments: Any) -> dict[str, object]:
    if isinstance(raw_arguments, dict):
        return raw_arguments
    if isinstance(raw_arguments, str):
        try:
            parsed = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            raise LLMError(
                "OpenAI-compatible provider returned malformed tool-call arguments"
            ) from exc
        if not isinstance(parsed, dict):
            raise LLMError(
                "OpenAI-compatible provider returned non-object tool-call arguments"
            )
        return parsed
    raise LLMError("OpenAI-compatible provider returned malformed tool-call arguments")


def _extract_tool_calls(message: Any) -> list[ChatToolCall]:
    raw_tool_calls = getattr(message, "tool_calls", None)
    if not isinstance(raw_tool_calls, list):
        return []

    tool_calls: list[ChatToolCall] = []
    for index, entry in enumerate(raw_tool_calls):
        function = getattr(entry, "function", None)
        tool_name = getattr(function, "name", None)
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise LLMError("OpenAI-compatible tool call missing function name")
        raw_arguments = getattr(function, "arguments", None)
        tool_calls.append(
            ChatToolCall(
                call_id=getattr(entry, "id", None) or f"tool-call-{index}",
                tool_name=tool_name,
                arguments=_parse_tool_arguments(raw_arguments),
            )
        )
    return tool_calls


class _StreamingChatResponse(Protocol):
    def __iter__(self) -> Iterator[Any]:
        """Yield chat completion chunks."""

    def close(self) -> None:
        """Close the underlying stream."""


class _OpenAICompatibleChatTurnStream:
    """Streaming OpenAI-compatible chat handle with cooperative cancellation."""

    def __init__(
        self,
        *,
        provider_name: str,
        stream: _StreamingChatResponse,
        response_model: type,
    ) -> None:
        self._provider_name = provider_name
        self._stream = stream
        self._response_model = response_model
        self._cancel_requested = threading.Event()
        self._accumulated_text = ""
        self._closed = False
        self._tool_call_buffers: dict[int, dict[str, str | None]] = {}
        self._last_usage: ChatTokenUsage | None = None

    def cancel(self) -> None:
        self._cancel_requested.set()
        self._close_stream()

    def __iter__(
        self,
    ) -> Iterator[
        ChatAssistantDeltaEvent
        | ChatToolCallsEvent
        | ChatFinalResponseEvent
        | ChatInterruptedEvent
    ]:
        try:
            for chunk in self._stream:
                if self._cancel_requested.is_set():
                    yield self._build_interrupted_event()
                    return

                usage = _extract_token_usage(chunk)
                if usage is not None:
                    self._last_usage = usage

                choices = getattr(chunk, "choices", None)
                if not isinstance(choices, list) or not choices:
                    continue

                choice = choices[0]
                delta = getattr(choice, "delta", None)
                if delta is not None:
                    delta_text = _extract_message_text(delta)
                    if delta_text:
                        self._accumulated_text += delta_text
                        yield ChatAssistantDeltaEvent(
                            delta_text=delta_text,
                            accumulated_text=self._accumulated_text,
                        )
                    self._accumulate_stream_tool_calls(delta)

                finish_reason = getattr(choice, "finish_reason", None)
                if finish_reason == "tool_calls":
                    tool_calls = self._build_stream_tool_calls()
                    yield ChatToolCallsEvent(
                        assistant_message=ChatMessage(
                            role="assistant",
                            content=self._accumulated_text or None,
                            tool_calls=tool_calls,
                        ),
                        tool_calls=tool_calls,
                        token_usage=self._last_usage,
                        raw_text=self._accumulated_text,
                    )
                    return

                if finish_reason in {"stop", "length"}:
                    if not self._accumulated_text:
                        raise LLMError(
                            f"{self._provider_name} response missing assistant content"
                        )
                    parsed = validate_json_text(
                        self._response_model, self._accumulated_text
                    )
                    yield ChatFinalResponseEvent(
                        assistant_message=ChatMessage(
                            role="assistant",
                            content=self._accumulated_text,
                        ),
                        final_response=parsed,
                        token_usage=self._last_usage,
                        raw_text=self._accumulated_text,
                    )
                    return
        except Exception as exc:  # pragma: no cover - provider/network dependent
            if self._cancel_requested.is_set():
                yield self._build_interrupted_event()
                return
            raise LLMError(
                f"{self._provider_name} streaming request failed: {exc}"
            ) from exc
        finally:
            self._close_stream()

        if self._cancel_requested.is_set():
            yield self._build_interrupted_event()
            return

        if self._tool_call_buffers:
            tool_calls = self._build_stream_tool_calls()
            yield ChatToolCallsEvent(
                assistant_message=ChatMessage(
                    role="assistant",
                    content=self._accumulated_text or None,
                    tool_calls=tool_calls,
                ),
                tool_calls=tool_calls,
                token_usage=self._last_usage,
                raw_text=self._accumulated_text,
            )
            return

        if self._accumulated_text:
            parsed = validate_json_text(self._response_model, self._accumulated_text)
            yield ChatFinalResponseEvent(
                assistant_message=ChatMessage(
                    role="assistant",
                    content=self._accumulated_text,
                ),
                final_response=parsed,
                token_usage=self._last_usage,
                raw_text=self._accumulated_text,
            )
            return

        raise LLMError(
            f"{self._provider_name} stream ended without a final response or tool calls."
        )

    def _accumulate_stream_tool_calls(self, delta: Any) -> None:
        raw_tool_calls = getattr(delta, "tool_calls", None)
        if not isinstance(raw_tool_calls, list):
            return

        for entry in raw_tool_calls:
            index = getattr(entry, "index", None)
            if not isinstance(index, int):
                index = len(self._tool_call_buffers)
            buffer = self._tool_call_buffers.setdefault(
                index,
                {"id": None, "name": None, "arguments": ""},
            )
            entry_id = getattr(entry, "id", None)
            if isinstance(entry_id, str) and entry_id.strip():
                buffer["id"] = entry_id
            function = getattr(entry, "function", None)
            if function is None:
                continue
            name = getattr(function, "name", None)
            if isinstance(name, str) and name.strip():
                buffer["name"] = name
            arguments = getattr(function, "arguments", None)
            if isinstance(arguments, str):
                buffer["arguments"] = (buffer["arguments"] or "") + arguments

    def _build_stream_tool_calls(self) -> list[ChatToolCall]:
        tool_calls: list[ChatToolCall] = []
        for index in sorted(self._tool_call_buffers):
            buffer = self._tool_call_buffers[index]
            name = buffer["name"]
            if not isinstance(name, str) or not name.strip():
                raise LLMError(
                    "OpenAI-compatible streaming tool call missing function name"
                )
            tool_calls.append(
                ChatToolCall(
                    call_id=buffer["id"] or f"tool-call-{index}",
                    tool_name=name,
                    arguments=_parse_tool_arguments(buffer["arguments"]),
                )
            )
        return tool_calls

    def _build_interrupted_event(self) -> ChatInterruptedEvent:
        return ChatInterruptedEvent(
            assistant_message=ChatMessage(
                role="assistant",
                content=self._accumulated_text or None,
                completion_state="interrupted",
            ),
            token_usage=self._last_usage,
            raw_text=self._accumulated_text,
            reason="Interrupted by stream cancellation.",
        )

    def _close_stream(self) -> None:
        if self._closed:
            return
        try:
            self._stream.close()
        except Exception:
            pass
        self._closed = True


class OpenAICompatibleChatLLMClient:
    """Chat adapter for providers exposing an OpenAI-compatible API surface."""

    def __init__(
        self,
        model_name: str,
        provider_name: str,
        api_key_env_var: str | None,
        base_url: str | None,
        timeout_seconds: float = 60.0,
        api_key: str | None = None,
    ) -> None:
        if OpenAI is None:
            raise LLMError(
                "OpenAI SDK dependencies are unavailable. "
                "Install project dependencies to use hosted providers."
            )

        api_token = api_key or (os.getenv(api_key_env_var) if api_key_env_var else None)
        if not api_token:
            env_name = api_key_env_var or "API key"
            raise LLMError(f"{env_name} is not configured")

        self._model_name = model_name
        self._provider_name = provider_name
        self._client = OpenAI(
            api_key=api_token,
            base_url=base_url,
            timeout=timeout_seconds,
        )

    def generate_chat_turn(self, request: ChatTurnRequest) -> ChatTurnResponse:
        """Send one non-streaming chat turn to an OpenAI-compatible provider."""

        payload: dict[str, object] = {
            "model": request.model_name or self._model_name,
            "messages": [
                _serialize_chat_message(message) for message in request.messages
            ],
            "temperature": request.temperature,
        }
        if request.tools:
            payload["tools"] = [
                _serialize_tool_definition(tool) for tool in request.tools
            ]
        try:
            response = self._client.chat.completions.create(**payload)
        except Exception as exc:  # pragma: no cover - provider/network dependent
            raise LLMError(f"{self._provider_name} request failed: {exc}") from exc

        choices = getattr(response, "choices", None)
        if not isinstance(choices, list) or not choices:
            raise LLMError(f"{self._provider_name} returned no choices")

        choice = choices[0]
        message = getattr(choice, "message", None)
        if message is None:
            raise LLMError(f"{self._provider_name} response missing message")

        tool_calls = _extract_tool_calls(message)
        content_text = _extract_message_text(message).strip()
        token_usage = _extract_token_usage(response)

        if tool_calls:
            return ChatTurnResponse(
                assistant_message=ChatMessage(
                    role="assistant",
                    content=content_text or None,
                    tool_calls=tool_calls,
                ),
                tool_calls=tool_calls,
                token_usage=token_usage,
                raw_text=content_text,
                finish_reason="tool_calls",
            )

        if not content_text:
            raise LLMError(f"{self._provider_name} response missing assistant content")

        parsed = validate_json_text(request.response_model, content_text)
        return ChatTurnResponse(
            assistant_message=ChatMessage(role="assistant", content=content_text),
            final_response=parsed,
            token_usage=token_usage,
            raw_text=content_text,
            finish_reason="final_response",
        )

    def stream_chat_turn(self, request: ChatTurnRequest) -> ChatTurnStream:
        """Send one streaming chat turn to an OpenAI-compatible provider."""

        payload: dict[str, object] = {
            "model": request.model_name or self._model_name,
            "messages": [
                _serialize_chat_message(message) for message in request.messages
            ],
            "temperature": request.temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if request.tools:
            payload["tools"] = [
                _serialize_tool_definition(tool) for tool in request.tools
            ]
        try:
            stream = self._client.chat.completions.create(**payload)
        except Exception as exc:  # pragma: no cover - provider/network dependent
            raise LLMError(
                f"{self._provider_name} streaming request failed: {exc}"
            ) from exc
        return _OpenAICompatibleChatTurnStream(
            provider_name=self._provider_name,
            stream=stream,
            response_model=request.response_model,
        )
