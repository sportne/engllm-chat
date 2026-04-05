"""Focused tests for internal chat workflow helper modules."""

from __future__ import annotations

import json

from engllm_chat.core.tokenize import tokenize
from engllm_chat.domain.models import (
    ChatConfig,
    ChatFinalResponse,
    ChatMessage,
    ChatTokenUsage,
    ChatToolCall,
    ChatToolResult,
)
from engllm_chat.tools.chat._workflow.context import _prepare_session_context
from engllm_chat.tools.chat._workflow.results import (
    _build_continuation_result,
    _build_interrupted_result,
    _finalize_session_turn_result,
    _tool_status_label,
)
from engllm_chat.tools.chat._workflow.tokens import (
    _estimate_message_tokens,
    _estimate_messages_tokens,
    _summarize_session_token_usage,
)
from engllm_chat.tools.chat.models import (
    ChatSessionState,
    ChatSessionTurnRecord,
    ChatWorkflowTurnResult,
)


def _manual_estimate_message_tokens(message: ChatMessage) -> int:
    token_count = 0
    if message.content:
        token_count += len(tokenize(message.content))
    if message.tool_calls:
        token_count += len(
            tokenize(
                json.dumps(
                    [
                        tool_call.model_dump(mode="json")
                        for tool_call in message.tool_calls
                    ],
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        )
    if message.tool_result is not None:
        token_count += len(
            tokenize(
                json.dumps(
                    message.tool_result.model_dump(mode="json"),
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        )
    return token_count


def test_tokens_estimate_message_content_tool_calls_and_tool_results() -> None:
    tool_call = ChatToolCall(
        call_id="call-1",
        tool_name="find_files",
        arguments={"path": "src", "pattern": "**/*.py"},
    )
    tool_result = ChatToolResult(
        call_id="call-1",
        tool_name="find_files",
        payload={"matches": ["src/app.py"]},
    )
    content_message = ChatMessage(role="assistant", content="hello world")
    tool_call_message = ChatMessage(role="assistant", tool_calls=[tool_call])
    tool_result_message = ChatMessage(role="tool", tool_result=tool_result)

    assert _estimate_message_tokens(content_message) == _manual_estimate_message_tokens(
        content_message
    )
    assert _estimate_message_tokens(
        tool_call_message
    ) == _manual_estimate_message_tokens(tool_call_message)
    assert _estimate_message_tokens(
        tool_result_message
    ) == _manual_estimate_message_tokens(tool_result_message)


def test_summarize_session_token_usage_respects_base_usage_and_estimates() -> None:
    turn_record = ChatSessionTurnRecord(
        status="completed",
        new_messages=[ChatMessage(role="user", content="question")],
        final_response=ChatFinalResponse(answer="answer"),
    )
    session_state = ChatSessionState(turns=[turn_record])
    active_context_messages = [
        ChatMessage(role="system", content="sys"),
        ChatMessage(role="user", content="question"),
    ]

    estimated = _summarize_session_token_usage(
        base_usage=None,
        session_state=session_state,
        active_context_messages=active_context_messages,
    )
    base_usage = ChatTokenUsage(total_tokens=99, input_tokens=60, output_tokens=39)
    summarized = _summarize_session_token_usage(
        base_usage=base_usage,
        session_state=session_state,
        active_context_messages=active_context_messages,
    )

    assert estimated.total_tokens == _estimate_messages_tokens(turn_record.new_messages)
    assert estimated.session_tokens == _estimate_messages_tokens(
        turn_record.new_messages
    )
    assert summarized.total_tokens == 99
    assert summarized.input_tokens == 60
    assert summarized.output_tokens == 39
    assert summarized.active_context_tokens == _estimate_messages_tokens(
        active_context_messages
    )


def test_prepare_session_context_keeps_exact_fit_and_truncates_whole_turns() -> None:
    first_turn = ChatSessionTurnRecord(
        status="completed",
        new_messages=[ChatMessage(role="user", content="alpha beta gamma")],
        final_response=ChatFinalResponse(answer="done"),
    )
    session_state = ChatSessionState(turns=[first_turn])

    system_message, user_message, prior_messages, active_start, warning = (
        _prepare_session_context(
            user_message="delta epsilon",
            session_state=session_state,
            config=ChatConfig(),
        )
    )
    candidate_messages = [system_message, *prior_messages, user_message]
    exact_fit_config = ChatConfig.model_validate(
        {
            "session": {
                "max_context_tokens": _estimate_messages_tokens(candidate_messages)
            }
        }
    )
    exact_fit = _prepare_session_context(
        user_message="delta epsilon",
        session_state=session_state,
        config=exact_fit_config,
    )
    truncated = _prepare_session_context(
        user_message="delta epsilon",
        session_state=session_state,
        config=ChatConfig.model_validate({"session": {"max_context_tokens": 1}}),
    )

    assert active_start == 0
    assert warning is None
    assert exact_fit[2] == prior_messages
    assert exact_fit[4] is None
    assert truncated[2] == []
    assert truncated[3] == 1
    assert truncated[4] is not None
    assert "removed from active context" in (truncated[4] or "")


def test_results_helpers_build_statuses_and_finalize_session_state() -> None:
    new_messages = [ChatMessage(role="user", content="Question")]
    tool_result = ChatToolResult(
        call_id="call-1",
        tool_name="find_files",
        payload={"matches": []},
    )

    continuation = _build_continuation_result(
        new_messages=new_messages,
        tool_results=[tool_result],
        token_usage=ChatTokenUsage(total_tokens=7),
        reason="Need more budget.",
    )
    interrupted = _build_interrupted_result(
        new_messages=new_messages,
        tool_results=[],
        token_usage=None,
        reason="Interrupted by user.",
    )
    completed = ChatWorkflowTurnResult(
        status="completed",
        new_messages=new_messages,
        final_response=ChatFinalResponse(answer="Answer"),
        token_usage=ChatTokenUsage(total_tokens=11),
        tool_results=[tool_result],
    )
    finalized = _finalize_session_turn_result(
        turn_result=completed,
        session_state=ChatSessionState(),
        active_context_start_turn=0,
        context_warning=None,
        system_message=ChatMessage(role="system", content="sys"),
    )

    assert continuation.status == "needs_continuation"
    assert continuation.continuation_reason == "Need more budget."
    assert interrupted.status == "interrupted"
    assert interrupted.interruption_reason == "Interrupted by user."
    assert finalized.session_state is not None
    assert finalized.session_state.turns[0].status == "completed"
    assert finalized.token_usage is not None
    assert finalized.token_usage.session_tokens == 11
    assert _tool_status_label("find_files") == "listing files"
    assert _tool_status_label("search_text") == "searching text"
    assert _tool_status_label("read_file") == "reading file"
    assert _tool_status_label("unknown_tool") == "thinking"
