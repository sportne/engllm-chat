"""Typed models for the interactive chat workflow."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from engllm_chat.domain.models import (
    ChatFinalResponse,
    ChatMessage,
    ChatTokenUsage,
    ChatToolResult,
    DomainModel,
)

ChatWorkflowTurnStatus = Literal["completed", "needs_continuation", "interrupted"]


class _PathArgument(DomainModel):
    """Common validation for tool arguments that include a relative path."""

    path: str = "."

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("path must not be empty")
        return cleaned


class ListDirectoryArgs(_PathArgument):
    """Arguments for `list_directory`."""


class ListDirectoryRecursiveArgs(_PathArgument):
    """Arguments for `list_directory_recursive`."""

    max_depth: int | None = Field(default=None, gt=0)


class FindFilesArgs(_PathArgument):
    """Arguments for `find_files`."""

    pattern: str

    @field_validator("pattern")
    @classmethod
    def validate_pattern(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("pattern must not be empty")
        return cleaned


class SearchTextArgs(_PathArgument):
    """Arguments for `search_text`."""

    query: str

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("query must not be empty")
        return cleaned


class GetFileInfoArgs(DomainModel):
    """Arguments for `get_file_info`."""

    path: str

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("path must not be empty")
        return cleaned


class ReadFileArgs(GetFileInfoArgs):
    """Arguments for `read_file`."""

    start_char: int | None = Field(default=None, ge=0)
    end_char: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_range(self) -> ReadFileArgs:
        if (
            self.start_char is not None
            and self.end_char is not None
            and self.end_char <= self.start_char
        ):
            raise ValueError("end_char must be greater than start_char")
        return self


class ChatSessionTurnRecord(DomainModel):
    """One stored visible chat turn in the in-memory session."""

    status: ChatWorkflowTurnStatus
    new_messages: list[ChatMessage] = Field(default_factory=list)
    final_response: ChatFinalResponse | None = None
    token_usage: ChatTokenUsage | None = None
    tool_results: list[ChatToolResult] = Field(default_factory=list)
    continuation_reason: str | None = None
    interruption_reason: str | None = None

    @model_validator(mode="after")
    def validate_status_fields(self) -> ChatSessionTurnRecord:
        if self.status == "completed":
            if self.final_response is None:
                raise ValueError("completed chat turns require final_response")
            if (
                self.continuation_reason is not None
                or self.interruption_reason is not None
            ):
                raise ValueError(
                    "completed chat turns do not allow "
                    "continuation/interruption reasons"
                )
            return self

        if self.status == "needs_continuation":
            if self.final_response is not None:
                raise ValueError(
                    "needs_continuation chat turns do not allow final_response"
                )
            if not (self.continuation_reason or "").strip():
                raise ValueError(
                    "needs_continuation chat turns require continuation_reason"
                )
            if self.interruption_reason is not None:
                raise ValueError(
                    "needs_continuation chat turns do not allow interruption_reason"
                )
            return self

        if self.final_response is not None:
            raise ValueError("interrupted chat turns do not allow final_response")
        if self.continuation_reason is not None:
            raise ValueError("interrupted chat turns do not allow continuation_reason")
        if not (self.interruption_reason or "").strip():
            raise ValueError("interrupted chat turns require interruption_reason")
        return self


class ChatSessionState(DomainModel):
    """In-memory visible chat history plus active-context window metadata."""

    turns: list[ChatSessionTurnRecord] = Field(default_factory=list)
    active_context_start_turn: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_active_context_start_turn(self) -> ChatSessionState:
        if self.active_context_start_turn > len(self.turns):
            raise ValueError(
                "active_context_start_turn cannot be greater than the number of turns"
            )
        return self


class ChatWorkflowTurnResult(DomainModel):
    """Provider-neutral result for one orchestrated chat turn attempt."""

    status: ChatWorkflowTurnStatus
    new_messages: list[ChatMessage] = Field(default_factory=list)
    final_response: ChatFinalResponse | None = None
    token_usage: ChatTokenUsage | None = None
    tool_results: list[ChatToolResult] = Field(default_factory=list)
    continuation_reason: str | None = None
    interruption_reason: str | None = None
    session_state: ChatSessionState | None = None
    context_warning: str | None = None

    @model_validator(mode="after")
    def validate_status_fields(self) -> ChatWorkflowTurnResult:
        if self.status == "completed":
            if self.final_response is None:
                raise ValueError("completed chat turns require final_response")
            if (
                self.continuation_reason is not None
                or self.interruption_reason is not None
            ):
                raise ValueError(
                    "completed chat turns do not allow "
                    "continuation/interruption reasons"
                )
            return self

        if self.status == "needs_continuation":
            if self.final_response is not None:
                raise ValueError(
                    "needs_continuation chat turns do not allow final_response"
                )
            if not (self.continuation_reason or "").strip():
                raise ValueError(
                    "needs_continuation chat turns require continuation_reason"
                )
            if self.interruption_reason is not None:
                raise ValueError(
                    "needs_continuation chat turns do not allow interruption_reason"
                )
            return self

        if self.final_response is not None:
            raise ValueError("interrupted chat turns do not allow final_response")
        if self.continuation_reason is not None:
            raise ValueError("interrupted chat turns do not allow continuation_reason")
        if not (self.interruption_reason or "").strip():
            raise ValueError("interrupted chat turns require interruption_reason")
        return self


class ChatWorkflowStatusEvent(DomainModel):
    """One transient workflow status update for the UI."""

    event_type: Literal["status"] = "status"
    status: str


class ChatWorkflowAssistantDeltaEvent(DomainModel):
    """One incremental assistant text update for the active transcript row."""

    event_type: Literal["assistant_delta"] = "assistant_delta"
    delta_text: str
    accumulated_text: str


class ChatWorkflowResultEvent(DomainModel):
    """Terminal workflow event carrying the completed turn result."""

    event_type: Literal["result"] = "result"
    result: ChatWorkflowTurnResult
