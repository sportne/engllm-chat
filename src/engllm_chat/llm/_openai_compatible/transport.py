"""Transport and credential helpers for OpenAI-compatible chat adapters."""

from __future__ import annotations

import os
from typing import Any

from engllm_chat.domain.errors import LLMError

_openai_sdk: Any = None
try:
    from openai import OpenAI as _OpenAIClient

    _openai_sdk = _OpenAIClient
except Exception:  # pragma: no cover - optional dependency
    pass

DEFAULT_OPENAI: Any = _openai_sdk


def resolve_api_token(
    *,
    api_key_env_var: str,
    api_key: str | None,
) -> str:
    """Resolve the API token for one OpenAI-compatible endpoint."""

    api_token = api_key if api_key is not None else os.getenv(api_key_env_var)
    if api_token is None:
        raise LLMError(f"{api_key_env_var} is not configured")
    return api_token


def build_openai_client(
    *,
    openai_client_class: Any,
    provider_name: str,
    api_key_env_var: str,
    api_key: str | None,
    base_url: str | None,
    timeout_seconds: float,
) -> Any:
    """Create the SDK client for one OpenAI-compatible provider."""

    if openai_client_class is None:
        raise LLMError(
            "OpenAI SDK dependencies are unavailable. "
            "Install project dependencies to use hosted providers."
        )

    api_token = resolve_api_token(
        api_key_env_var=api_key_env_var,
        api_key=api_key,
    )
    return openai_client_class(
        api_key=api_token,
        base_url=base_url,
        timeout=timeout_seconds,
    )
