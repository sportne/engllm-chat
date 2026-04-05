"""Provider and runtime configuration models for the chat project."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from .common import (
    _PROVIDER_DEFAULT_API_BASE_URLS,
    _PROVIDER_DEFAULT_API_KEY_ENV_VARS,
    ChatProvider,
    DomainModel,
)


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
    api_base_url: str | None = None
    timeout_seconds: float = 60.0
    api_key_env_var: str | None = None
    prompt_for_api_key_if_missing: bool = True
    verbose_llm_logging: bool = False

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_ollama_base_url(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        if "api_base_url" not in value and "ollama_base_url" in value:
            migrated = dict(value)
            migrated["api_base_url"] = migrated.pop("ollama_base_url")
            return migrated
        return value

    @field_validator("model_name")
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
