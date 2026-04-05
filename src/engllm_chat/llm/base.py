"""Provider-neutral structured generation interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, TypeVar

from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from engllm_chat.domain.errors import LLMError, ValidationError
from engllm_chat.domain.models import ChatMessage, ChatTokenUsage, ChatToolCall

ModelT = TypeVar("ModelT", bound=BaseModel)


@dataclass(frozen=True)
class StructuredGenerationRequest:
    """Structured generation request."""

    system_prompt: str
    user_prompt: str
    response_model: type[BaseModel]
    model_name: str
    temperature: float = 0.2


@dataclass(frozen=True)
class StructuredGenerationResponse:
    """Structured generation response."""

    content: BaseModel
    raw_text: str
    model_name: str


@dataclass(frozen=True)
class ChatToolDefinition:
    """Provider-neutral tool metadata for interactive chat turns."""

    name: str
    description: str
    input_schema: dict[str, object]
    argument_model: type[BaseModel] | None = None


@dataclass(frozen=True)
class ChatTurnRequest:
    """Non-streaming provider-neutral chat turn request."""

    messages: list[ChatMessage]
    response_model: type[BaseModel]
    model_name: str
    tools: list[ChatToolDefinition] = field(default_factory=list)
    temperature: float = 0.2


@dataclass(frozen=True)
class ChatTurnResponse:
    """Non-streaming provider-neutral chat turn response."""

    assistant_message: ChatMessage
    tool_calls: list[ChatToolCall] = field(default_factory=list)
    final_response: BaseModel | None = None
    token_usage: ChatTokenUsage | None = None
    raw_text: str = ""
    finish_reason: Literal["tool_calls", "final_response"] = "final_response"

    def __post_init__(self) -> None:
        """Validate finish-reason and payload invariants."""

        if self.assistant_message.role != "assistant":
            raise ValueError("assistant_message must have role='assistant'")

        if self.finish_reason == "tool_calls":
            if not self.tool_calls:
                raise ValueError(
                    "finish_reason='tool_calls' requires at least one tool call"
                )
            if self.final_response is not None:
                raise ValueError(
                    "finish_reason='tool_calls' does not allow final_response"
                )
            return

        if self.tool_calls:
            raise ValueError("finish_reason='final_response' does not allow tool_calls")
        if self.final_response is None:
            raise ValueError("finish_reason='final_response' requires final_response")


class LLMClient(Protocol):
    """Provider-neutral client contract."""

    def generate_structured(
        self,
        request: StructuredGenerationRequest,
    ) -> StructuredGenerationResponse:
        """Generate and validate a structured response."""


class ChatLLMClient(Protocol):
    """Provider-neutral interactive chat client contract."""

    def generate_chat_turn(
        self,
        request: ChatTurnRequest,
    ) -> ChatTurnResponse:
        """Generate one complete non-streaming chat turn result."""


def validate_payload(
    response_model: type[BaseModel],
    payload: dict[str, object],
) -> BaseModel:
    """Validate a provider payload against the requested schema."""

    try:
        return response_model.model_validate(payload)
    except PydanticValidationError as exc:
        raise ValidationError(f"Structured response validation failed: {exc}") from exc


def validate_json_text(response_model: type[BaseModel], json_text: str) -> BaseModel:
    """Validate JSON text against response schema."""

    try:
        return response_model.model_validate_json(json_text)
    except PydanticValidationError as exc:
        raise ValidationError(f"Structured JSON validation failed: {exc}") from exc
    except ValueError as exc:
        raise LLMError(
            f"Provider returned non-JSON structured response: {exc}"
        ) from exc
