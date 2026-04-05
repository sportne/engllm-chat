"""Retry helpers for schema-first OpenAI-compatible chat turns."""

from __future__ import annotations

from engllm_chat.domain.models import ChatMessage

_MAX_SCHEMA_ATTEMPTS = 3


def _build_schema_retry_feedback(error_message: str) -> ChatMessage:
    """Return one corrective user message after a schema validation failure."""

    return ChatMessage(
        role="user",
        content=(
            "The previous response did not satisfy the required structured schema.\n"
            f"Validation error: {error_message}\n"
            "Return exactly one valid action object that matches the requested schema."
        ),
    )
