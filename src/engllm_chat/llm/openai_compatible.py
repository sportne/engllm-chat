"""OpenAI-compatible chat provider implementation."""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.parse
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, create_model

from engllm_chat.domain.errors import LLMError, ValidationError
from engllm_chat.domain.models import ChatMessage, ChatTokenUsage, ChatToolCall
from engllm_chat.llm.base import (
    ChatToolDefinition,
    ChatTurnRequest,
    ChatTurnResponse,
    validate_payload,
)

_openai_sdk: Any = None
try:
    from openai import OpenAI as _OpenAIClient

    _openai_sdk = _OpenAIClient
except Exception:  # pragma: no cover - optional dependency
    pass

OpenAI: Any = _openai_sdk
_MAX_SCHEMA_ATTEMPTS = 3
_LOGGER = logging.getLogger("engllm_chat.llm.openai_compatible")


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
            "role": "user",
            "content": (
                "Tool result:\n"
                f"{json.dumps(message.tool_result.model_dump(mode='json'), sort_keys=True)}"
            ),
        }

    payload: dict[str, object] = {"role": message.role}
    if message.role == "assistant" and message.tool_calls:
        request_text = json.dumps(
            [tool_call.model_dump(mode="json") for tool_call in message.tool_calls],
            sort_keys=True,
        )
        content_parts = []
        if message.content is not None:
            content_parts.append(message.content)
        content_parts.append(f"Tool request:\n{request_text}")
        payload["content"] = "\n\n".join(part for part in content_parts if part.strip())
        return payload

    if message.content is not None:
        payload["content"] = message.content
    elif message.role in {"system", "user", "assistant"}:
        payload["content"] = ""
    return payload


def _build_chat_turn_action_model(
    response_model: type[BaseModel],
    tools: list[ChatToolDefinition],
) -> type[BaseModel]:
    final_action_model = create_model(
        f"{response_model.__name__}FinalAction",
        kind=(Literal["final_response"], "final_response"),
        response=(response_model, ...),
    )

    action_field_type: Any = final_action_model
    if tools:
        tool_name_pattern = (
            "^(" + "|".join(re.escape(tool.name) for tool in tools) + ")$"
        )

        tool_action_model = create_model(
            f"{response_model.__name__}ToolAction",
            kind=(Literal["tool_request"], "tool_request"),
            tool_name=(str, Field(pattern=tool_name_pattern)),
            arguments=(dict[str, object], Field(default_factory=dict)),
        )
        action_field_type = Annotated[
            tool_action_model | final_action_model,
            Field(discriminator="kind"),
        ]

    return create_model(
        f"{response_model.__name__}ChatTurnActionEnvelope",
        action=(action_field_type, ...),
    )


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


def _extract_action(
    action_response_model: type[BaseModel],
    message: Any,
) -> BaseModel:
    parsed = getattr(message, "parsed", None)
    if isinstance(parsed, BaseModel):
        return parsed
    if isinstance(parsed, dict):
        return validate_payload(action_response_model, parsed)

    content_text = _extract_message_text(message).strip()
    if not content_text:
        raise LLMError("OpenAI-compatible response missing assistant content")
    try:
        return action_response_model.model_validate_json(content_text)
    except Exception as exc:
        raise LLMError(
            "OpenAI-compatible provider returned malformed action JSON"
        ) from exc


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


def _extract_chat_turn_result(
    *,
    response_model: type[BaseModel],
    action_response_model: type[BaseModel],
    message: Any,
) -> tuple[str, ChatToolCall | None, BaseModel | None, str]:
    action_envelope = _extract_action(action_response_model, message)
    action = getattr(action_envelope, "action", None)
    if action is None:
        if isinstance(action_envelope, response_model):
            raw_text = (
                _extract_message_text(message).strip()
                or action_envelope.model_dump_json()
            )
            return "final_response", None, action_envelope, raw_text
        raise LLMError("OpenAI-compatible response missing action payload")

    content_text = _extract_message_text(message).strip()
    action_kind = getattr(action, "kind", None)
    if action_kind == "tool_request":
        tool_name = getattr(action, "tool_name", None)
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise LLMError("OpenAI-compatible tool action missing tool_name")
        arguments = getattr(action, "arguments", None)
        if not isinstance(arguments, dict):
            raise LLMError("OpenAI-compatible tool action missing arguments object")
        tool_call = ChatToolCall(
            call_id="tool-call-0",
            tool_name=tool_name,
            arguments=arguments,
        )
        raw_text = content_text or json.dumps(
            {
                "action": {
                    "kind": "tool_request",
                    "tool_name": tool_name,
                    "arguments": arguments,
                }
            },
            sort_keys=True,
        )
        return "tool_calls", tool_call, None, raw_text

    if action_kind != "final_response":
        raise LLMError("OpenAI-compatible response returned unknown action kind")

    final_response = getattr(action, "response", None)
    if isinstance(final_response, BaseModel):
        parsed_model = final_response
    elif isinstance(final_response, dict):
        parsed_model = validate_payload(response_model, final_response)
    else:
        parsed_model = response_model.model_validate(final_response)
    raw_text = content_text or parsed_model.model_dump_json()
    return "final_response", None, parsed_model, raw_text


def _build_schema_retry_feedback(error_message: str) -> ChatMessage:
    """Return one corrective user message after a schema validation failure."""

    return ChatMessage(
        role="user",
        content=(
            "The previous response did not satisfy the required structured schema.\n"
            f"Validation error: {error_message}\n"
            "Return exactly one valid action object that matches the requested schema."
        ),
    )


def _to_loggable_payload(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _to_loggable_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_loggable_payload(item) for item in value]
    if isinstance(value, tuple):
        return [_to_loggable_payload(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


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
        verbose_logging: bool = False,
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
        self._verbose_logging = verbose_logging
        self._client = OpenAI(
            api_key=api_token,
            base_url=base_url,
            timeout=timeout_seconds,
        )

    def _log_request_messages(
        self,
        *,
        model_name: str,
        attempt_index: int,
        messages: list[dict[str, object]],
    ) -> None:
        if not self._verbose_logging:
            return
        _LOGGER.info(
            "LLM request -> provider=%s model=%s attempt=%s\n%s",
            self._provider_name,
            model_name,
            attempt_index + 1,
            json.dumps(messages, indent=2, sort_keys=True),
        )

    def _log_response_message(
        self,
        *,
        model_name: str,
        message: Any,
    ) -> None:
        if not self._verbose_logging:
            return

        parsed = getattr(message, "parsed", None)
        payload = {
            "content": _extract_message_text(message),
            "parsed": _to_loggable_payload(parsed),
        }
        _LOGGER.info(
            "LLM response <- provider=%s model=%s\n%s",
            self._provider_name,
            model_name,
            json.dumps(payload, indent=2, sort_keys=True),
        )

    def generate_chat_turn(self, request: ChatTurnRequest) -> ChatTurnResponse:
        """Send one non-streaming chat turn to an OpenAI-compatible provider."""

        action_response_model = _build_chat_turn_action_model(
            request.response_model,
            request.tools,
        )
        request_messages = list(request.messages)
        last_schema_error: Exception | None = None

        for attempt_index in range(_MAX_SCHEMA_ATTEMPTS):
            serialized_messages = [
                _serialize_chat_message(message) for message in request_messages
            ]
            payload: dict[str, object] = {
                "model": request.model_name or self._model_name,
                "messages": serialized_messages,
                "temperature": request.temperature,
                "response_format": action_response_model,
            }
            self._log_request_messages(
                model_name=str(payload["model"]),
                attempt_index=attempt_index,
                messages=serialized_messages,
            )
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
            self._log_response_message(
                model_name=str(payload["model"]),
                message=message,
            )

            token_usage = _extract_token_usage(response)
            try:
                finish_reason, tool_call, final_response, raw_text = (
                    _extract_chat_turn_result(
                        response_model=request.response_model,
                        action_response_model=action_response_model,
                        message=message,
                    )
                )
            except (ValidationError, ValueError, TypeError, LLMError) as exc:
                last_schema_error = exc
                if attempt_index + 1 >= _MAX_SCHEMA_ATTEMPTS:
                    raise LLMError(
                        "OpenAI-compatible provider failed to return a valid "
                        f"structured response after {_MAX_SCHEMA_ATTEMPTS} attempts: {exc}"
                    ) from exc
                request_messages.append(_build_schema_retry_feedback(str(exc)))
                continue

            if finish_reason == "tool_calls":
                if tool_call is None:
                    raise LLMError(
                        "OpenAI-compatible tool action completed without tool call"
                    )
                return ChatTurnResponse(
                    assistant_message=ChatMessage(
                        role="assistant",
                        content=raw_text or None,
                        tool_calls=[tool_call],
                    ),
                    tool_calls=[tool_call],
                    token_usage=token_usage,
                    raw_text=raw_text,
                    finish_reason="tool_calls",
                )

            if final_response is None:
                raise LLMError(
                    "OpenAI-compatible final response completed without response payload"
                )
            return ChatTurnResponse(
                assistant_message=ChatMessage(role="assistant", content=raw_text),
                final_response=final_response,
                token_usage=token_usage,
                raw_text=raw_text,
                finish_reason="final_response",
            )

        raise LLMError(
            "OpenAI-compatible provider failed to return a valid structured response"
        ) from last_schema_error
