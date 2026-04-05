"""OpenAI-compatible chat provider implementation."""

from __future__ import annotations

import json
import os
import urllib.parse
from typing import Any, cast

from pydantic import BaseModel

from engllm_chat.domain.errors import LLMError
from engllm_chat.domain.models import ChatMessage, ChatTokenUsage, ChatToolCall
from engllm_chat.llm.base import (
    ChatToolDefinition,
    ChatTurnRequest,
    ChatTurnResponse,
    validate_json_text,
    validate_payload,
)

_openai_sdk: Any = None
_pydantic_function_tool_sdk: Any = None
try:
    from openai import OpenAI as _OpenAIClient
    from openai import pydantic_function_tool as _pydantic_function_tool

    _openai_sdk = _OpenAIClient
    _pydantic_function_tool_sdk = _pydantic_function_tool
except Exception:  # pragma: no cover - optional dependency
    pass

OpenAI: Any = _openai_sdk
pydantic_function_tool: Any = _pydantic_function_tool_sdk


def normalize_ollama_base_url(base_url: str | None) -> str:
    """Normalize an Ollama host/base URL to its OpenAI-compatible `/v1` form."""

    default_base_url = "http://127.0.0.1:11434"
    raw = (base_url or default_base_url).strip() or default_base_url
    if "://" not in raw:
        raw = f"http://{raw}"

    parsed = urllib.parse.urlparse(raw)
    path = parsed.path.rstrip("/")

    if not path:
        normalized_path = "/v1"
    elif path == "/v1" or path.endswith("/v1"):
        normalized_path = path
    else:
        normalized_path = f"{path}/v1"

    normalized = parsed._replace(path=normalized_path, params="", query="", fragment="")
    return urllib.parse.urlunparse(normalized)


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
    if tool.argument_model is not None:
        if pydantic_function_tool is None:
            raise LLMError(
                "OpenAI SDK dependencies are unavailable. "
                "Install project dependencies to use hosted providers."
            )
        return cast(
            dict[str, object],
            pydantic_function_tool(
                tool.argument_model,
                name=tool.name,
                description=tool.description,
            ),
        )
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


def _normalize_parsed_tool_arguments(parsed_arguments: Any) -> dict[str, object] | None:
    if parsed_arguments is None:
        return None
    if isinstance(parsed_arguments, BaseModel):
        return parsed_arguments.model_dump(mode="json")
    if isinstance(parsed_arguments, dict):
        return parsed_arguments
    return None


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
        parsed_arguments = _normalize_parsed_tool_arguments(
            getattr(function, "parsed_arguments", None)
        )
        raw_arguments = getattr(function, "arguments", None)
        tool_calls.append(
            ChatToolCall(
                call_id=getattr(entry, "id", None) or f"tool-call-{index}",
                tool_name=tool_name,
                arguments=parsed_arguments or _parse_tool_arguments(raw_arguments),
            )
        )
    return tool_calls


def _extract_final_response(
    response_model: type[BaseModel],
    message: Any,
) -> tuple[BaseModel, str]:
    content_text = _extract_message_text(message).strip()
    parsed = getattr(message, "parsed", None)

    if isinstance(parsed, BaseModel):
        return parsed, content_text or parsed.model_dump_json()
    if isinstance(parsed, dict):
        parsed_model = validate_payload(response_model, parsed)
        return parsed_model, content_text or parsed_model.model_dump_json()
    if parsed is not None:
        parsed_model = response_model.model_validate(parsed)
        return parsed_model, content_text or parsed_model.model_dump_json()
    if content_text:
        return validate_json_text(response_model, content_text), content_text
    raise LLMError("OpenAI-compatible response missing assistant content")


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
        if provider_name == "ollama" and not api_token:
            api_token = "ollama"
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
            "response_format": request.response_model,
        }
        if request.tools:
            payload["tools"] = [
                _serialize_tool_definition(tool) for tool in request.tools
            ]
        try:
            response = self._client.beta.chat.completions.parse(**payload)
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
        token_usage = _extract_token_usage(response)

        if tool_calls:
            content_text = _extract_message_text(message).strip()
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

        parsed, content_text = _extract_final_response(request.response_model, message)
        return ChatTurnResponse(
            assistant_message=ChatMessage(role="assistant", content=content_text),
            final_response=parsed,
            token_usage=token_usage,
            raw_text=content_text,
            finish_reason="final_response",
        )
