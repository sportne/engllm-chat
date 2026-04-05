"""Shared base types and provider defaults for domain models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


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
    "ollama": "http://127.0.0.1:11434",
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
