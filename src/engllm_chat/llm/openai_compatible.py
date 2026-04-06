"""OpenAI-compatible chat provider implementation.

This file stays as the public façade even though most of the implementation now
lives in ``llm._openai_compatible``. That keeps imports stable for the rest of
the project and makes this module the best place to start when learning how the
provider adapter fits together.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from engllm_chat.domain.errors import LLMError, ValidationError
from engllm_chat.domain.models import ChatMessage
from engllm_chat.llm._openai_compatible.parsing import (
    _extract_action,
    _extract_chat_turn_result,
    _extract_message_text,
    _extract_token_usage,
    _to_loggable_payload,
)
from engllm_chat.llm._openai_compatible.retries import (
    _MAX_SCHEMA_ATTEMPTS,
    _build_schema_retry_feedback,
)
from engllm_chat.llm._openai_compatible.serialization import (
    _build_chat_turn_action_model,
    _serialize_chat_message,
)
from engllm_chat.llm._openai_compatible.transport import (
    DEFAULT_OPENAI,
    build_openai_client,
    normalize_ollama_base_url,
)
from engllm_chat.llm.base import ChatToolDefinition, ChatTurnRequest, ChatTurnResponse

OpenAI: Any = DEFAULT_OPENAI
_LOGGER = logging.getLogger("engllm_chat.llm.openai_compatible")

__all__ = [
    "OpenAI",
    "OpenAICompatibleChatLLMClient",
    "ChatToolDefinition",
    "normalize_ollama_base_url",
    "_serialize_chat_message",
    "_build_chat_turn_action_model",
    "_extract_message_text",
    "_extract_action",
    "_extract_token_usage",
    "_extract_chat_turn_result",
    "_build_schema_retry_feedback",
    "_to_loggable_payload",
]


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
        self._model_name = model_name
        self._provider_name = provider_name
        self._verbose_logging = verbose_logging
        self._client = build_openai_client(
            openai_client_class=OpenAI,
            provider_name=provider_name,
            api_key_env_var=api_key_env_var,
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
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

        # We ask the provider for one schema-validated action at a time rather
        # than using provider-native tool calling. The action envelope is the
        # stable contract that the workflow understands across providers.
        action_response_model = _build_chat_turn_action_model(
            request.response_model,
            request.tools,
        )
        request_messages = list(request.messages)
        last_schema_error: Exception | None = None

        for attempt_index in range(_MAX_SCHEMA_ATTEMPTS):
            # Tool results stay in the normal chat transcript as plain messages,
            # so every provider sees the same conversation shape.
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
            except PydanticValidationError as exc:
                last_schema_error = exc
                if attempt_index + 1 >= _MAX_SCHEMA_ATTEMPTS:
                    raise LLMError(
                        "OpenAI-compatible provider failed to return a valid "
                        f"structured response after {_MAX_SCHEMA_ATTEMPTS} attempts: {exc}"
                    ) from exc
                request_messages.append(_build_schema_retry_feedback(str(exc)))
                continue
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
                # Corrective feedback is appended as another user message so the
                # retry loop stays provider-neutral and easy to replay in tests.
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
