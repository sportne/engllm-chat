"""Interactive chat workflow tool."""

from __future__ import annotations

from typing import Any

__all__ = ["run_chat_turn", "run_chat_session_turn", "run_streaming_chat_session_turn"]


def __getattr__(name: str) -> Any:
    """Resolve package exports lazily to avoid prompt/workflow import cycles."""

    if name == "run_chat_turn":
        from engllm_chat.tools.chat.workflow import run_chat_turn

        return run_chat_turn
    if name == "run_chat_session_turn":
        from engllm_chat.tools.chat.workflow import run_chat_session_turn

        return run_chat_session_turn
    if name == "run_streaming_chat_session_turn":
        from engllm_chat.tools.chat.workflow import run_streaming_chat_session_turn

        return run_streaming_chat_session_turn
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
