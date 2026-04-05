"""Configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError as PydanticValidationError

from engllm_chat.domain.errors import ConfigError, ValidationError
from engllm_chat.domain.models import ChatConfig


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Configuration file not found: {path}")
    if not path.is_file():
        raise ConfigError(f"Configuration path is not a file: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML at {path}: {exc}") from exc

    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"Expected mapping at root of YAML file: {path}")
    return raw


def load_chat_config(path: Path) -> ChatConfig:
    """Load and validate the standalone chat configuration file."""

    raw = _load_yaml(path)

    for section_name in ("llm", "source_filters", "session", "tool_limits", "ui"):
        section_value = raw.get(section_name)
        if section_value is not None and not isinstance(section_value, dict):
            raise ConfigError(f"chat config '{section_name}' must be a mapping")

    try:
        return ChatConfig.model_validate(raw)
    except PydanticValidationError as exc:
        raise ValidationError(f"Invalid chat config at {path}: {exc}") from exc
