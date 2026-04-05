"""Deterministic token estimation helpers for chat workflow state."""

from __future__ import annotations

import json

from engllm_chat.core.tokenize import tokenize
from engllm_chat.domain.models import ChatMessage, ChatTokenUsage
from engllm_chat.tools.chat.models import ChatSessionState, ChatSessionTurnRecord


def _serialize_for_token_estimation(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _estimate_message_tokens(message: ChatMessage) -> int:
    token_count = 0
    if message.content:
        token_count += len(tokenize(message.content))
    if message.tool_calls:
        token_count += len(
            tokenize(
                _serialize_for_token_estimation(
                    [
                        tool_call.model_dump(mode="json")
                        for tool_call in message.tool_calls
                    ]
                )
            )
        )
    if message.tool_result is not None:
        token_count += len(
            tokenize(
                _serialize_for_token_estimation(
                    message.tool_result.model_dump(mode="json")
                )
            )
        )
    return token_count


def _estimate_messages_tokens(messages: list[ChatMessage]) -> int:
    return sum(_estimate_message_tokens(message) for message in messages)


def _estimate_turn_total_tokens(turn: ChatSessionTurnRecord) -> int:
    if turn.token_usage is not None and turn.token_usage.total_tokens is not None:
        return turn.token_usage.total_tokens
    return _estimate_messages_tokens(turn.new_messages)


def _flatten_turn_messages(turns: list[ChatSessionTurnRecord]) -> list[ChatMessage]:
    messages: list[ChatMessage] = []
    for turn in turns:
        messages.extend(turn.new_messages)
    return messages


def _summarize_session_token_usage(
    *,
    base_usage: ChatTokenUsage | None,
    session_state: ChatSessionState,
    active_context_messages: list[ChatMessage],
) -> ChatTokenUsage:
    session_tokens = sum(
        _estimate_turn_total_tokens(turn) for turn in session_state.turns
    )
    active_context_tokens = _estimate_messages_tokens(active_context_messages)

    if base_usage is None:
        current_turn_total_tokens = (
            _estimate_turn_total_tokens(session_state.turns[-1])
            if session_state.turns
            else 0
        )
        return ChatTokenUsage(
            total_tokens=current_turn_total_tokens,
            session_tokens=session_tokens,
            active_context_tokens=active_context_tokens,
        )

    return base_usage.model_copy(
        update={
            "session_tokens": session_tokens,
            "active_context_tokens": active_context_tokens,
        }
    )
