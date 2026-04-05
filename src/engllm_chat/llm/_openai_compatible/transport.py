"""Transport and credential helpers for OpenAI-compatible chat adapters."""

from __future__ import annotations

import os
import urllib.parse
from typing import Any

from engllm_chat.domain.errors import LLMError

_openai_sdk: Any = None
try:
    from openai import OpenAI as _OpenAIClient

    _openai_sdk = _OpenAIClient
except Exception:  # pragma: no cover - optional dependency
    pass

DEFAULT_OPENAI: Any = _openai_sdk


def normalize_ollama_base_url(base_url: str | None) -> str:
    """Normalize an Ollama host/base URL to its OpenAI-compatible `/v1` form."""

    default_base_url = "http://127.0.0.1:11434"
    raw = (base_url or default_base_url).strip() or default_base_url
    if "://" not in raw:
        raw = f"http://{raw}"

    parsed = urllib.parse.urlparse(raw)
    path = parsed.path.rstrip("/")

    if not path:
        normalized_path = "/v1"
    elif path == "/v1" or path.endswith("/v1"):
        normalized_path = path
    else:
        normalized_path = f"{path}/v1"

    normalized = parsed._replace(path=normalized_path, params="", query="", fragment="")
    return urllib.parse.urlunparse(normalized)


def resolve_api_token(
    *,
    provider_name: str,
    api_key_env_var: str | None,
    api_key: str | None,
) -> str:
    """Resolve the API token for one provider, applying Ollama defaults."""

    api_token = api_key or (os.getenv(api_key_env_var) if api_key_env_var else None)
    if provider_name == "ollama" and not api_token:
        api_token = "ollama"
    if not api_token:
        env_name = api_key_env_var or "API key"
        raise LLMError(f"{env_name} is not configured")
    return api_token


def build_openai_client(
    *,
    openai_client_class: Any,
    provider_name: str,
    api_key_env_var: str | None,
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
        provider_name=provider_name,
        api_key_env_var=api_key_env_var,
        api_key=api_key,
    )
    return openai_client_class(
        api_key=api_token,
        base_url=base_url,
        timeout=timeout_seconds,
    )
