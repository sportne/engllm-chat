"""Shared base types for domain models."""

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

DEFAULT_CHAT_API_KEY_ENV_VAR = "ENGLLM_CHAT_API_KEY"
