"""Response parsing helpers for OpenAI-compatible chat adapters."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from engllm_chat.domain.errors import LLMError
from engllm_chat.domain.models import ChatTokenUsage, ChatToolCall
from engllm_chat.llm.base import validate_payload


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
    # Prefer the SDK-native parsed payload when it is available, but keep a raw
    # text fallback so the adapter still works with providers that only return
    # the JSON envelope as assistant content.
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
        # Some providers may hand back a parsed final response directly. We keep
        # accepting that path so the workflow contract stays stable even when
        # provider parsing behavior varies.
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
