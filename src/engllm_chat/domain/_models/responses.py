"""Response, citation, and token-usage models for the chat project."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator, model_validator

from .common import DomainModel


class ChatCitation(DomainModel):
    """Grounding citation displayed with a final chat answer."""

    source_path: Path
    line_start: int | None = Field(default=None, ge=1)
    line_end: int | None = Field(default=None, ge=1)
    excerpt: str | None = None

    @model_validator(mode="after")
    def validate_line_range(self) -> ChatCitation:
        if (
            self.line_start is not None
            and self.line_end is not None
            and self.line_end < self.line_start
        ):
            raise ValueError("line_end must be greater than or equal to line_start")
        return self


class ChatFinalResponse(DomainModel):
    """Validated final answer emitted by the interactive chat loop."""

    answer: str
    citations: list[ChatCitation] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    uncertainty: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    follow_up_suggestions: list[str] = Field(default_factory=list)

    @field_validator("answer")
    @classmethod
    def validate_answer(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("answer must not be empty")
        return value

    @field_validator("uncertainty", "missing_information", "follow_up_suggestions")
    @classmethod
    def validate_string_lists(cls, value: list[str]) -> list[str]:
        cleaned = [entry.strip() for entry in value]
        if any(not entry for entry in cleaned):
            raise ValueError("final response list entries must not be empty")
        return cleaned


class ChatTokenUsage(DomainModel):
    """Token-usage accounting for one turn or session summary."""

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    session_tokens: int | None = Field(default=None, ge=0)
    active_context_tokens: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def populate_total_tokens(self) -> ChatTokenUsage:
        derived_total = self.input_tokens + self.output_tokens
        if self.total_tokens is None:
            self.total_tokens = derived_total
        elif self.total_tokens < derived_total:
            raise ValueError(
                "total_tokens cannot be less than input_tokens + output_tokens"
            )
        return self
