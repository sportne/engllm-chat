"""Focused tests for internal OpenAI-compatible adapter helpers."""

from __future__ import annotations

import pytest

from engllm_chat.domain.errors import LLMError
from engllm_chat.domain.models import ChatMessage
from engllm_chat.llm._openai_compatible.retries import (
    _MAX_SCHEMA_ATTEMPTS,
    _build_schema_retry_feedback,
)
from engllm_chat.llm._openai_compatible.transport import (
    build_openai_client,
    resolve_api_token,
)


class _FakeOpenAIClient:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


def test_transport_helpers_cover_token_resolution_and_client_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TEST_API_KEY", raising=False)

    monkeypatch.setenv("TEST_API_KEY", "from-env")
    assert (
        resolve_api_token(
            api_key_env_var="TEST_API_KEY",
            api_key=None,
        )
        == "from-env"
    )

    client = build_openai_client(
        openai_client_class=_FakeOpenAIClient,
        provider_name="openai",
        api_key_env_var="TEST_API_KEY",
        api_key=None,
        base_url="https://api.openai.com/v1",
        timeout_seconds=12.0,
    )
    assert client.kwargs == {
        "api_key": "from-env",
        "base_url": "https://api.openai.com/v1/",
        "timeout": 12.0,
    }


def test_transport_helpers_raise_for_missing_sdk_or_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MISSING_KEY", raising=False)

    with pytest.raises(LLMError, match="MISSING_KEY is not configured"):
        resolve_api_token(
            api_key_env_var="MISSING_KEY",
            api_key=None,
        )

    with pytest.raises(LLMError, match="OpenAI SDK dependencies are unavailable"):
        build_openai_client(
            openai_client_class=None,
            provider_name="openai",
            api_key_env_var="ENGLLM_CHAT_API_KEY",
            api_key="secret",
            base_url="https://api.openai.com/v1",
            timeout_seconds=60.0,
        )


def test_retry_feedback_helper_and_attempt_limit_constant() -> None:
    feedback = _build_schema_retry_feedback("bad schema")

    assert _MAX_SCHEMA_ATTEMPTS == 3
    assert isinstance(feedback, ChatMessage)
    assert feedback.role == "user"
    assert "did not satisfy the required structured schema" in (feedback.content or "")
    assert "bad schema" in (feedback.content or "")
