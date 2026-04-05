"""Local Ollama provider implementation (isolated in llm module)."""

from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.parse
import urllib.request as request_lib
from collections.abc import Iterator
from typing import Protocol

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
    StructuredGenerationRequest,
    StructuredGenerationResponse,
    validate_json_text,
)

DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"


def _normalize_chat_url(base_url: str) -> str:
    raw = base_url.strip() or DEFAULT_OLLAMA_BASE_URL
    if "://" not in raw:
        raw = f"http://{raw}"

    parsed = urllib.parse.urlparse(raw)
    path = parsed.path.rstrip("/")

    if not path:
        normalized_path = "/api/chat"
    elif path == "/api":
        normalized_path = "/api/chat"
    elif path == "/api/chat":
        normalized_path = "/api/chat"
    else:
        normalized_path = f"{path}/api/chat"

    normalized = parsed._replace(path=normalized_path, params="", query="", fragment="")
    return urllib.parse.urlunparse(normalized)


def _serialize_chat_message(message: ChatMessage) -> dict[str, object]:
    if message.role == "tool":
        if message.tool_result is None:
            raise LLMError("Tool chat message missing tool_result")
        content = json.dumps(
            message.tool_result.model_dump(mode="json"), sort_keys=True
        )
        return {"role": "tool", "content": content}

    content = (message.content or "").strip()
    return {"role": message.role, "content": content}


def _serialize_tool_definition(tool: ChatToolDefinition) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        },
    }


def _extract_token_usage(response_payload: dict[str, object]) -> ChatTokenUsage:
    input_tokens = response_payload.get("prompt_eval_count", 0)
    output_tokens = response_payload.get("eval_count", 0)

    if not isinstance(input_tokens, int):
        input_tokens = 0
    if not isinstance(output_tokens, int):
        output_tokens = 0

    return ChatTokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )


def _extract_tool_calls(message: dict[str, object]) -> list[ChatToolCall]:
    raw_tool_calls = message.get("tool_calls")
    if not isinstance(raw_tool_calls, list):
        return []

    tool_calls: list[ChatToolCall] = []
    for index, entry in enumerate(raw_tool_calls):
        if not isinstance(entry, dict):
            raise LLMError("Ollama tool-call entry is malformed")

        function = entry.get("function")
        if not isinstance(function, dict):
            raise LLMError("Ollama tool-call entry missing function payload")

        tool_name = function.get("name")
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise LLMError("Ollama tool-call entry missing function name")

        arguments = function.get("arguments")
        if not isinstance(arguments, dict):
            raise LLMError("Ollama tool-call entry missing structured arguments")

        tool_calls.append(
            ChatToolCall(
                call_id=f"ollama-tool-call-{index}",
                tool_name=tool_name,
                arguments=arguments,
            )
        )

    return tool_calls


class _StreamingHTTPResponse(Protocol):
    """Minimal interface shared by urllib responses and streaming test doubles."""

    def readline(self) -> bytes: ...

    def close(self) -> None: ...


class _OllamaChatTurnStream:
    """Streaming Ollama chat handle with cooperative cancellation."""

    def __init__(
        self,
        *,
        request: request_lib.Request,
        timeout_seconds: float,
        response_model: type,
    ) -> None:
        self._request = request
        self._timeout_seconds = timeout_seconds
        self._response_model = response_model
        self._cancel_requested = threading.Event()
        self._response: _StreamingHTTPResponse | None = None
        self._accumulated_text = ""
        self._raw_chunks: list[str] = []
        self._closed = False

    def cancel(self) -> None:
        self._cancel_requested.set()
        self._close_response()

    def __iter__(
        self,
    ) -> Iterator[
        ChatAssistantDeltaEvent
        | ChatToolCallsEvent
        | ChatFinalResponseEvent
        | ChatInterruptedEvent
    ]:
        try:
            with request_lib.urlopen(
                self._request, timeout=self._timeout_seconds
            ) as resp:
                self._response = resp
                status = int(getattr(resp, "status", 200))
                if status < 200 or status >= 300:
                    raw_response = resp.read().decode("utf-8", errors="replace")
                    raise LLMError(
                        f"Ollama request failed with status {status}: "
                        f"{raw_response or 'Empty response body'}"
                    )

                while True:
                    if self._cancel_requested.is_set():
                        yield self._build_interrupted_event(
                            "Interrupted by stream cancellation."
                        )
                        return

                    try:
                        line = resp.readline()
                    except (OSError, ValueError) as exc:
                        if self._cancel_requested.is_set():
                            yield self._build_interrupted_event(
                                "Interrupted by stream cancellation."
                            )
                            return
                        raise LLMError(
                            "Ollama streaming response aborted unexpectedly."
                        ) from exc

                    if not line:
                        break

                    decoded_line = line.decode("utf-8", errors="replace").strip()
                    if not decoded_line:
                        continue
                    self._raw_chunks.append(decoded_line)

                    try:
                        payload = json.loads(decoded_line)
                    except json.JSONDecodeError as exc:
                        raise LLMError(
                            "Ollama returned malformed JSON stream chunk."
                        ) from exc

                    if not isinstance(payload, dict):
                        raise LLMError("Ollama returned malformed JSON stream chunk.")

                    message = payload.get("message", {})
                    if message is None:
                        message = {}
                    if not isinstance(message, dict):
                        raise LLMError("Ollama stream chunk missing message payload.")

                    content = message.get("content")
                    if content is not None and not isinstance(content, str):
                        raise LLMError("Ollama stream chunk missing assistant content.")
                    if isinstance(content, str) and content:
                        self._accumulated_text += content
                        yield ChatAssistantDeltaEvent(
                            delta_text=content,
                            accumulated_text=self._accumulated_text,
                        )

                    tool_calls = _extract_tool_calls(message)
                    if tool_calls:
                        assistant_message = ChatMessage(
                            role="assistant",
                            content=self._accumulated_text or None,
                            tool_calls=tool_calls,
                        )
                        yield ChatToolCallsEvent(
                            assistant_message=assistant_message,
                            tool_calls=tool_calls,
                            token_usage=_extract_token_usage(payload),
                            raw_text="\n".join(self._raw_chunks),
                        )
                        return

                    if bool(payload.get("done")):
                        if not self._accumulated_text:
                            raise LLMError(
                                "Ollama response missing assistant content "
                                "and tool calls"
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
                            token_usage=_extract_token_usage(payload),
                            raw_text=self._accumulated_text,
                        )
                        return
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            detail = error_body or exc.reason or "Unknown error"
            raise LLMError(
                f"Ollama request failed with status {exc.code}: {detail}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if self._cancel_requested.is_set():
                yield self._build_interrupted_event(
                    "Interrupted by stream cancellation."
                )
                return
            raise LLMError(
                f"Cannot connect to Ollama at {self._request.full_url}. "
                "Ensure Ollama is running and OLLAMA_BASE_URL is correct."
            ) from exc
        finally:
            self._close_response()

        if self._cancel_requested.is_set():
            yield self._build_interrupted_event("Interrupted by stream cancellation.")
            return

        raise LLMError("Ollama stream ended without a final response or tool calls.")

    def _build_interrupted_event(self, reason: str) -> ChatInterruptedEvent:
        return ChatInterruptedEvent(
            assistant_message=ChatMessage(
                role="assistant",
                content=self._accumulated_text or None,
                completion_state="interrupted",
            ),
            raw_text=self._accumulated_text,
            reason=reason,
        )

    def _close_response(self) -> None:
        if self._closed:
            return
        if self._response is not None:
            try:
                self._response.close()
            except OSError:
                pass
        self._closed = True


class OllamaLLMClient:
    """Structured Ollama adapter behind the provider-neutral interface."""

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        env_base_url = os.getenv("OLLAMA_BASE_URL")
        resolved_base_url = base_url or env_base_url or DEFAULT_OLLAMA_BASE_URL
        self._model_name = model_name
        self._chat_url = _normalize_chat_url(resolved_base_url)
        self._timeout_seconds = timeout_seconds

    def generate_structured(
        self,
        request: StructuredGenerationRequest,
    ) -> StructuredGenerationResponse:
        """Send one schema-constrained request to Ollama and validate JSON output."""

        request_payload = {
            "model": request.model_name or self._model_name,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
            "stream": False,
            "format": request.response_model.model_json_schema(),
            "options": {"temperature": request.temperature},
        }

        body = json.dumps(request_payload).encode("utf-8")
        http_request = request_lib.Request(
            self._chat_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request_lib.urlopen(
                http_request, timeout=self._timeout_seconds
            ) as resp:
                status = int(getattr(resp, "status", 200))
                raw_response = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            detail = error_body or exc.reason or "Unknown error"
            raise LLMError(
                f"Ollama request failed with status {exc.code}: {detail}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise LLMError(
                f"Cannot connect to Ollama at {self._chat_url}. "
                "Ensure Ollama is running and OLLAMA_BASE_URL is correct."
            ) from exc

        if status < 200 or status >= 300:
            raise LLMError(
                f"Ollama request failed with status {status}: "
                f"{raw_response or 'Empty response body'}"
            )

        try:
            response_payload = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise LLMError("Ollama returned malformed JSON response.") from exc

        if not isinstance(response_payload, dict):
            raise LLMError("Ollama returned malformed JSON response.")

        message = response_payload.get("message")
        if not isinstance(message, dict):
            raise LLMError("Ollama response missing 'message.content'")

        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise LLMError("Ollama response missing 'message.content'")

        parsed = validate_json_text(request.response_model, content)
        return StructuredGenerationResponse(
            content=parsed,
            raw_text=content,
            model_name=request.model_name or self._model_name,
        )

    def generate_chat_turn(
        self,
        request: ChatTurnRequest,
    ) -> ChatTurnResponse:
        """Send one non-streaming chat turn to Ollama and parse tools/final answer."""

        request_payload: dict[str, object] = {
            "model": request.model_name or self._model_name,
            "messages": [
                _serialize_chat_message(message) for message in request.messages
            ],
            "stream": False,
            "options": {"temperature": request.temperature},
        }

        if request.tools:
            request_payload["tools"] = [
                _serialize_tool_definition(tool) for tool in request.tools
            ]
        else:
            request_payload["format"] = request.response_model.model_json_schema()

        body = json.dumps(request_payload).encode("utf-8")
        http_request = request_lib.Request(
            self._chat_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request_lib.urlopen(
                http_request, timeout=self._timeout_seconds
            ) as resp:
                status = int(getattr(resp, "status", 200))
                raw_response = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            detail = error_body or exc.reason or "Unknown error"
            raise LLMError(
                f"Ollama request failed with status {exc.code}: {detail}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise LLMError(
                f"Cannot connect to Ollama at {self._chat_url}. "
                "Ensure Ollama is running and OLLAMA_BASE_URL is correct."
            ) from exc

        if status < 200 or status >= 300:
            raise LLMError(
                f"Ollama request failed with status {status}: "
                f"{raw_response or 'Empty response body'}"
            )

        try:
            response_payload = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise LLMError("Ollama returned malformed JSON response.") from exc

        if not isinstance(response_payload, dict):
            raise LLMError("Ollama returned malformed JSON response.")

        message = response_payload.get("message")
        if not isinstance(message, dict):
            raise LLMError("Ollama response missing 'message'")

        tool_calls = _extract_tool_calls(message)
        content = message.get("content")
        if content is not None and not isinstance(content, str):
            raise LLMError("Ollama response missing assistant content")
        content_text = content.strip() if isinstance(content, str) else ""
        token_usage = _extract_token_usage(response_payload)

        if tool_calls:
            return ChatTurnResponse(
                assistant_message=ChatMessage(
                    role="assistant",
                    content=content_text or None,
                    tool_calls=tool_calls,
                ),
                tool_calls=tool_calls,
                token_usage=token_usage,
                raw_text=raw_response,
                finish_reason="tool_calls",
            )

        if not content_text:
            raise LLMError("Ollama response missing assistant content and tool calls")

        parsed = validate_json_text(request.response_model, content_text)
        return ChatTurnResponse(
            assistant_message=ChatMessage(role="assistant", content=content_text),
            final_response=parsed,
            token_usage=token_usage,
            raw_text=content_text,
            finish_reason="final_response",
        )

    def stream_chat_turn(
        self,
        request: ChatTurnRequest,
    ) -> ChatTurnStream:
        """Send one streaming chat turn to Ollama with cancellation support."""

        request_payload: dict[str, object] = {
            "model": request.model_name or self._model_name,
            "messages": [
                _serialize_chat_message(message) for message in request.messages
            ],
            "stream": True,
            "options": {"temperature": request.temperature},
        }

        if request.tools:
            request_payload["tools"] = [
                _serialize_tool_definition(tool) for tool in request.tools
            ]
        else:
            request_payload["format"] = request.response_model.model_json_schema()

        body = json.dumps(request_payload).encode("utf-8")
        http_request = request_lib.Request(
            self._chat_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return _OllamaChatTurnStream(
            request=http_request,
            timeout_seconds=self._timeout_seconds,
            response_model=request.response_model,
        )
