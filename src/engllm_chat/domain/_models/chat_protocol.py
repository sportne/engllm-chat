"""Provider-neutral chat protocol models."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from .common import (
    ChatAssistantCompletionState,
    ChatMessageRole,
    DomainModel,
)


class ChatToolCall(DomainModel):
    """Structured tool-call request emitted by a chat provider."""

    call_id: str
    tool_name: str
    arguments: dict[str, object] = Field(default_factory=dict)

    @field_validator("call_id", "tool_name")
    @classmethod
    def validate_required_strings(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("tool call identifiers must not be empty")
        return cleaned


class ChatToolResult(DomainModel):
    """Structured tool execution result returned to the model."""

    call_id: str
    tool_name: str
    status: Literal["ok", "error"] = "ok"
    payload: dict[str, object] = Field(default_factory=dict)
    error_message: str | None = None

    @field_validator("call_id", "tool_name")
    @classmethod
    def validate_required_strings(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("tool result identifiers must not be empty")
        return cleaned

    @model_validator(mode="after")
    def validate_error_fields(self) -> ChatToolResult:
        # Downstream code branches on status alone, so we keep the error shape
        # strict here instead of forcing every consumer to re-check it.
        if self.status == "error" and not (self.error_message or "").strip():
            raise ValueError("error_message is required when tool result status=error")
        if self.status == "ok" and self.error_message is not None:
            raise ValueError(
                "error_message is only allowed when tool result status=error"
            )
        return self


class ChatMessage(DomainModel):
    """Provider-neutral chat message envelope."""

    role: ChatMessageRole
    content: str | None = None
    tool_calls: list[ChatToolCall] = Field(default_factory=list)
    tool_result: ChatToolResult | None = None
    completion_state: ChatAssistantCompletionState = "complete"

    @model_validator(mode="after")
    def validate_message_shape(self) -> ChatMessage:
        has_content = bool((self.content or "").strip())

        if self.role in {"system", "user"}:
            if not has_content:
                raise ValueError(f"{self.role} messages must include content")
            if self.tool_calls:
                raise ValueError(f"{self.role} messages cannot include tool calls")
            if self.tool_result is not None:
                raise ValueError(f"{self.role} messages cannot include tool results")
            if self.completion_state != "complete":
                raise ValueError(
                    f"{self.role} messages cannot use non-default completion_state"
                )
            return self

        if self.role == "assistant":
            # Assistant messages are allowed to be "tool request only" messages,
            # which is how the provider hands the workflow its next action.
            if (
                self.completion_state == "complete"
                and not has_content
                and not self.tool_calls
            ):
                raise ValueError(
                    "assistant messages must include content or at least one tool call"
                )
            if self.tool_result is not None:
                raise ValueError("assistant messages cannot include tool results")
            return self

        if self.tool_calls:
            raise ValueError("tool messages cannot include tool calls")
        # Tool results travel back through the transcript as a dedicated message
        # role rather than through provider-native tool result payloads.
        if self.tool_result is None:
            raise ValueError("tool messages must include a tool result")
        if self.completion_state != "complete":
            raise ValueError("tool messages cannot use non-default completion_state")
        return self
