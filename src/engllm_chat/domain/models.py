"""Typed domain models for the standalone EngLLM chat project."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DomainModel(BaseModel):
    """Base model with strict shape checking."""

    model_config = ConfigDict(extra="forbid")


ChatProvider = Literal[
    "ollama",
    "mock",
    "openai",
    "xai",
    "anthropic",
    "gemini",
]
ChatMessageRole = Literal["system", "user", "assistant", "tool"]
ChatAssistantCompletionState = Literal["complete", "interrupted"]

_PROVIDER_DEFAULT_API_BASE_URLS: dict[ChatProvider, str | None] = {
    "mock": None,
    "ollama": None,
    "openai": "https://api.openai.com/v1",
    "xai": "https://api.x.ai/v1",
    "anthropic": "https://api.anthropic.com/v1/",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
}

_PROVIDER_DEFAULT_API_KEY_ENV_VARS: dict[ChatProvider, str | None] = {
    "mock": None,
    "ollama": None,
    "openai": "OPENAI_API_KEY",
    "xai": "XAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


class ChatCredentialPromptMetadata(DomainModel):
    """UI-safe credential prompt policy without any persisted secret value."""

    provider: str
    api_key_env_var: str | None = None
    prompt_for_api_key_if_missing: bool = True
    expects_api_key: bool = False
    secret_kind: Literal["api_key"] = "api_key"
    mask_input: bool = True
    allow_empty_secret: bool = True
    persist_secret: bool = False


class ChatLLMConfig(DomainModel):
    """Standalone chat-provider configuration."""

    provider: ChatProvider = "ollama"
    model_name: str = "qwen2.5:7b-instruct"
    temperature: float = 0.1
    ollama_base_url: str = "http://127.0.0.1:11434"
    api_base_url: str | None = None
    timeout_seconds: float = 60.0
    api_key_env_var: str | None = None
    prompt_for_api_key_if_missing: bool = True

    @field_validator("model_name", "ollama_base_url")
    @classmethod
    def validate_non_empty_strings(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("chat LLM string settings must not be empty")
        return cleaned

    @field_validator("api_base_url")
    @classmethod
    def validate_optional_api_base_url(cls, value: str | None) -> str | None:
        if value is None:
            return value
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("api_base_url must not be empty when provided")
        return cleaned

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("temperature must be between 0.0 and 1.0")
        return value

    @field_validator("timeout_seconds")
    @classmethod
    def validate_timeout_seconds(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("timeout_seconds must be greater than 0")
        return value

    @field_validator("api_key_env_var")
    @classmethod
    def validate_api_key_env_var(cls, value: str | None) -> str | None:
        if value is None:
            return value
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("api_key_env_var must not be empty when provided")
        return cleaned

    def credential_prompt_metadata(self) -> ChatCredentialPromptMetadata:
        """Return UI-safe credential-prompt metadata derived from config."""

        return ChatCredentialPromptMetadata(
            provider=self.provider,
            api_key_env_var=self.resolved_api_key_env_var(),
            prompt_for_api_key_if_missing=self.prompt_for_api_key_if_missing,
            expects_api_key=self.provider not in {"ollama", "mock"},
        )

    def resolved_api_key_env_var(self) -> str | None:
        """Return the explicit or provider-default API-key env var."""

        return self.api_key_env_var or _PROVIDER_DEFAULT_API_KEY_ENV_VARS[self.provider]

    def resolved_api_base_url(self) -> str | None:
        """Return the explicit or provider-default hosted API base URL."""

        return self.api_base_url or _PROVIDER_DEFAULT_API_BASE_URLS[self.provider]


class ChatSourceFilters(DomainModel):
    """Directory discovery filters applied inside the selected root."""

    include: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)
    include_hidden: bool = False

    @field_validator("include", "exclude")
    @classmethod
    def validate_glob_patterns(cls, value: list[str]) -> list[str]:
        cleaned = [entry.strip() for entry in value]
        if any(not entry for entry in cleaned):
            raise ValueError("source filter patterns must not be empty")
        return cleaned


class ChatSessionConfig(DomainModel):
    """Per-session context and tool-call safety limits."""

    max_context_tokens: int = 24000
    max_tool_round_trips: int = 8
    max_tool_calls_per_round: int = 4
    max_total_tool_calls_per_turn: int = 12

    @field_validator(
        "max_context_tokens",
        "max_tool_round_trips",
        "max_tool_calls_per_round",
        "max_total_tool_calls_per_turn",
    )
    @classmethod
    def validate_positive_ints(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("chat session limits must be positive integers")
        return value


class ChatToolLimits(DomainModel):
    """Deterministic upper bounds for read-only tool execution."""

    max_entries_per_call: int = 200
    max_recursive_depth: int = 12
    max_search_matches: int = 50
    max_read_lines: int = 200
    max_file_size_characters: int = 262144
    max_read_file_chars: int | None = None
    max_tool_result_chars: int = 24000

    @field_validator(
        "max_entries_per_call",
        "max_recursive_depth",
        "max_search_matches",
        "max_read_lines",
        "max_file_size_characters",
        "max_tool_result_chars",
    )
    @classmethod
    def validate_positive_ints(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("chat tool limits must be positive integers")
        return value

    @field_validator("max_read_file_chars")
    @classmethod
    def validate_optional_positive_int(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("chat tool limits must be positive integers")
        return value


class ChatUIConfig(DomainModel):
    """Minimal Textual UI toggles for the interactive chat app."""

    show_token_usage: bool = True
    show_footer_help: bool = True


class ChatConfig(DomainModel):
    """Standalone chat configuration."""

    llm: ChatLLMConfig = Field(default_factory=ChatLLMConfig)
    source_filters: ChatSourceFilters = Field(default_factory=ChatSourceFilters)
    session: ChatSessionConfig = Field(default_factory=ChatSessionConfig)
    tool_limits: ChatToolLimits = Field(default_factory=ChatToolLimits)
    ui: ChatUIConfig = Field(default_factory=ChatUIConfig)


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
        if self.tool_result is None:
            raise ValueError("tool messages must include a tool result")
        if self.completion_state != "complete":
            raise ValueError("tool messages cannot use non-default completion_state")
        return self


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


class SectionDraft(DomainModel):
    """Simple structured section draft retained for mock-provider tests."""

    section_id: str
    title: str
    content: str
    evidence_refs: list[dict[str, object]] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class SectionUpdateProposal(DomainModel):
    """Simple structured update proposal retained for mock-provider tests."""

    section_id: str
    title: str
    existing_text: str
    proposed_text: str
    rationale: str
    uncertainty_list: list[str] = Field(default_factory=list)
    review_priority: Literal["low", "medium", "high"] = "medium"
    evidence_refs: list[dict[str, object]] = Field(default_factory=list)


class QueryAnswer(DomainModel):
    """Simple structured answer retained for mock-provider tests."""

    answer: str
    citations: list[ChatCitation] = Field(default_factory=list)
    uncertainty: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
