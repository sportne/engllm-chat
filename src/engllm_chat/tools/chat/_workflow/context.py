"""Session-context preparation helpers for chat workflow turns."""

from __future__ import annotations

from engllm_chat.domain.models import ChatConfig, ChatMessage
from engllm_chat.prompts.chat import build_chat_system_prompt
from engllm_chat.tools.chat.models import ChatSessionState
from engllm_chat.tools.chat.registry import build_chat_tool_definitions

from .tokens import _estimate_messages_tokens, _flatten_turn_messages


def _build_system_message(config: ChatConfig) -> ChatMessage:
    return ChatMessage(
        role="system",
        content=build_chat_system_prompt(
            tool_limits=config.tool_limits,
            tools=build_chat_tool_definitions(),
        ),
    )


def _prepare_session_context(
    *,
    user_message: str,
    session_state: ChatSessionState,
    config: ChatConfig,
) -> tuple[ChatMessage, ChatMessage, list[ChatMessage], int, str | None]:
    system_message = _build_system_message(config)
    user_chat_message = ChatMessage(role="user", content=user_message)

    active_context_start_turn = session_state.active_context_start_turn
    context_warning: str | None = None
    while active_context_start_turn < len(session_state.turns):
        prior_messages = _flatten_turn_messages(
            session_state.turns[active_context_start_turn:]
        )
        candidate_context_tokens = _estimate_messages_tokens(
            [system_message, *prior_messages, user_chat_message]
        )
        if candidate_context_tokens <= config.session.max_context_tokens:
            break
        active_context_start_turn += 1
        context_warning = (
            "Older turns were removed from active context to stay within the "
            "configured token limit."
        )

    prior_messages = _flatten_turn_messages(
        session_state.turns[active_context_start_turn:]
    )
    return (
        system_message,
        user_chat_message,
        prior_messages,
        active_context_start_turn,
        context_warning,
    )
