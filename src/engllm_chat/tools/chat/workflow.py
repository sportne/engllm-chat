"""Per-turn chat orchestration loop."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from threading import Lock

from engllm_chat.core.tokenize import tokenize
from engllm_chat.domain.errors import LLMError
from engllm_chat.domain.models import (
    ChatConfig,
    ChatFinalResponse,
    ChatMessage,
    ChatTokenUsage,
    ChatToolResult,
)
from engllm_chat.llm.base import (
    ChatLLMClient,
    ChatTurnRequest,
)
from engllm_chat.prompts.chat import build_chat_system_prompt
from engllm_chat.tools.chat.models import (
    ChatSessionState,
    ChatSessionTurnRecord,
    ChatWorkflowResultEvent,
    ChatWorkflowStatusEvent,
    ChatWorkflowTurnResult,
)
from engllm_chat.tools.chat.registry import (
    build_chat_tool_definitions,
    execute_chat_tool_call,
)


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


def _build_turn_record(turn_result: ChatWorkflowTurnResult) -> ChatSessionTurnRecord:
    return ChatSessionTurnRecord(
        status=turn_result.status,
        new_messages=turn_result.new_messages,
        final_response=turn_result.final_response,
        token_usage=turn_result.token_usage,
        tool_results=turn_result.tool_results,
        continuation_reason=turn_result.continuation_reason,
        interruption_reason=turn_result.interruption_reason,
    )


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


def _build_continuation_result(
    *,
    new_messages: list[ChatMessage],
    tool_results: list[ChatToolResult],
    token_usage: ChatTokenUsage | None,
    reason: str,
) -> ChatWorkflowTurnResult:
    return ChatWorkflowTurnResult(
        status="needs_continuation",
        new_messages=new_messages,
        final_response=None,
        token_usage=token_usage,
        tool_results=tool_results,
        continuation_reason=reason,
    )


def _build_interrupted_result(
    *,
    new_messages: list[ChatMessage],
    tool_results: list[ChatToolResult],
    token_usage: ChatTokenUsage | None,
    reason: str,
) -> ChatWorkflowTurnResult:
    return ChatWorkflowTurnResult(
        status="interrupted",
        new_messages=new_messages,
        final_response=None,
        token_usage=token_usage,
        tool_results=tool_results,
        interruption_reason=reason,
    )


def _tool_status_label(tool_name: str) -> str:
    if tool_name in {"list_directory", "list_directory_recursive", "find_files"}:
        return "listing files"
    if tool_name == "search_text":
        return "searching text"
    if tool_name in {"get_file_info", "read_file"}:
        return "reading file"
    return "thinking"


def _prepare_session_context(
    *,
    user_message: str,
    session_state: ChatSessionState,
    config: ChatConfig,
) -> tuple[ChatMessage, ChatMessage, list[ChatMessage], int, str | None]:
    system_message = ChatMessage(
        role="system",
        content=build_chat_system_prompt(
            tool_limits=config.tool_limits,
            tools=build_chat_tool_definitions(),
        ),
    )
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


def _finalize_session_turn_result(
    *,
    turn_result: ChatWorkflowTurnResult,
    session_state: ChatSessionState,
    active_context_start_turn: int,
    context_warning: str | None,
    system_message: ChatMessage,
) -> ChatWorkflowTurnResult:
    updated_turns = [*session_state.turns, _build_turn_record(turn_result)]
    updated_session_state = ChatSessionState(
        turns=updated_turns,
        active_context_start_turn=active_context_start_turn,
    )
    active_context_messages = [
        system_message,
        *_flatten_turn_messages(
            updated_session_state.turns[
                updated_session_state.active_context_start_turn :
            ]
        ),
    ]
    token_usage = _summarize_session_token_usage(
        base_usage=turn_result.token_usage,
        session_state=updated_session_state,
        active_context_messages=active_context_messages,
    )

    return turn_result.model_copy(
        update={
            "token_usage": token_usage,
            "session_state": updated_session_state,
            "context_warning": context_warning,
        }
    )


class ChatSessionTurnRunner:
    """Session-aware cancellable chat workflow for the interactive UI."""

    def __init__(
        self,
        *,
        user_message: str,
        session_state: ChatSessionState,
        root_path: Path,
        config: ChatConfig,
        llm_client: ChatLLMClient,
    ) -> None:
        (
            self._system_message,
            self._user_chat_message,
            self._prior_messages,
            self._active_context_start_turn,
            self._context_warning,
        ) = _prepare_session_context(
            user_message=user_message,
            session_state=session_state,
            config=config,
        )
        self._user_message = user_message
        self._session_state = session_state
        self._root_path = root_path
        self._config = config
        self._llm_client = llm_client
        self._lock = Lock()
        self._cancel_requested = False

    def cancel(self) -> None:
        with self._lock:
            self._cancel_requested = True

    def __iter__(
        self,
    ) -> Iterator[ChatWorkflowStatusEvent | ChatWorkflowResultEvent]:
        messages: list[ChatMessage] = [
            self._system_message,
            *self._prior_messages,
            self._user_chat_message,
        ]
        new_messages: list[ChatMessage] = [self._user_chat_message]
        tool_results: list[ChatToolResult] = []
        round_count = 0
        executed_tool_call_count = 0
        tool_definitions = build_chat_tool_definitions()
        last_token_usage: ChatTokenUsage | None = None
        yield ChatWorkflowStatusEvent(status="thinking")

        while True:
            if self._cancel_requested:
                result = _build_interrupted_result(
                    new_messages=new_messages,
                    tool_results=tool_results,
                    token_usage=last_token_usage,
                    reason="Interrupted by user.",
                )
                yield ChatWorkflowResultEvent(
                    result=_finalize_session_turn_result(
                        turn_result=result,
                        session_state=self._session_state,
                        active_context_start_turn=self._active_context_start_turn,
                        context_warning=self._context_warning,
                        system_message=self._system_message,
                    )
                )
                return

            response = self._llm_client.generate_chat_turn(
                ChatTurnRequest(
                    messages=messages,
                    response_model=ChatFinalResponse,
                    model_name=self._config.llm.model_name,
                    tools=tool_definitions,
                    temperature=self._config.llm.temperature,
                )
            )
            last_token_usage = response.token_usage

            if self._cancel_requested:
                result = _build_interrupted_result(
                    new_messages=new_messages,
                    tool_results=tool_results,
                    token_usage=last_token_usage,
                    reason="Interrupted by user.",
                )
                yield ChatWorkflowResultEvent(
                    result=_finalize_session_turn_result(
                        turn_result=result,
                        session_state=self._session_state,
                        active_context_start_turn=self._active_context_start_turn,
                        context_warning=self._context_warning,
                        system_message=self._system_message,
                    )
                )
                return

            messages.append(response.assistant_message)
            new_messages.append(response.assistant_message)

            if response.finish_reason == "final_response":
                yield ChatWorkflowStatusEvent(status="drafting answer")
                final_response = ChatFinalResponse.model_validate(
                    response.final_response
                )
                result = ChatWorkflowTurnResult(
                    status="completed",
                    new_messages=new_messages,
                    final_response=final_response,
                    token_usage=last_token_usage,
                    tool_results=tool_results,
                )
                yield ChatWorkflowResultEvent(
                    result=_finalize_session_turn_result(
                        turn_result=result,
                        session_state=self._session_state,
                        active_context_start_turn=self._active_context_start_turn,
                        context_warning=self._context_warning,
                        system_message=self._system_message,
                    )
                )
                return

            if len(response.tool_calls) > self._config.session.max_tool_calls_per_round:
                result = _build_continuation_result(
                    new_messages=new_messages,
                    tool_results=tool_results,
                    token_usage=last_token_usage,
                    reason=(
                        "The model requested more tool calls in one round "
                        "than allowed. "
                        "User confirmation is required before continuing."
                    ),
                )
                yield ChatWorkflowResultEvent(
                    result=_finalize_session_turn_result(
                        turn_result=result,
                        session_state=self._session_state,
                        active_context_start_turn=self._active_context_start_turn,
                        context_warning=self._context_warning,
                        system_message=self._system_message,
                    )
                )
                return

            if (
                executed_tool_call_count + len(response.tool_calls)
                > self._config.session.max_total_tool_calls_per_turn
            ):
                result = _build_continuation_result(
                    new_messages=new_messages,
                    tool_results=tool_results,
                    token_usage=last_token_usage,
                    reason=(
                        "The model needs more total tool-call budget before it can "
                        "continue this turn."
                    ),
                )
                yield ChatWorkflowResultEvent(
                    result=_finalize_session_turn_result(
                        turn_result=result,
                        session_state=self._session_state,
                        active_context_start_turn=self._active_context_start_turn,
                        context_warning=self._context_warning,
                        system_message=self._system_message,
                    )
                )
                return

            for tool_call in response.tool_calls:
                if self._cancel_requested:
                    result = _build_interrupted_result(
                        new_messages=new_messages,
                        tool_results=tool_results,
                        token_usage=last_token_usage,
                        reason="Interrupted by user.",
                    )
                    yield ChatWorkflowResultEvent(
                        result=_finalize_session_turn_result(
                            turn_result=result,
                            session_state=self._session_state,
                            active_context_start_turn=self._active_context_start_turn,
                            context_warning=self._context_warning,
                            system_message=self._system_message,
                        )
                    )
                    return
                yield ChatWorkflowStatusEvent(
                    status=_tool_status_label(tool_call.tool_name)
                )
                if self._cancel_requested:
                    result = _build_interrupted_result(
                        new_messages=new_messages,
                        tool_results=tool_results,
                        token_usage=last_token_usage,
                        reason="Interrupted by user.",
                    )
                    yield ChatWorkflowResultEvent(
                        result=_finalize_session_turn_result(
                            turn_result=result,
                            session_state=self._session_state,
                            active_context_start_turn=self._active_context_start_turn,
                            context_warning=self._context_warning,
                            system_message=self._system_message,
                        )
                    )
                    return
                tool_result = execute_chat_tool_call(
                    tool_call,
                    root_path=self._root_path,
                    config=self._config,
                )
                tool_results.append(tool_result)
                executed_tool_call_count += 1
                tool_message = ChatMessage(role="tool", tool_result=tool_result)
                messages.append(tool_message)
                new_messages.append(tool_message)

            round_count += 1
            if round_count >= self._config.session.max_tool_round_trips:
                result = _build_continuation_result(
                    new_messages=new_messages,
                    tool_results=tool_results,
                    token_usage=last_token_usage,
                    reason=(
                        "The model needs more tool rounds before it can "
                        "provide a final response."
                    ),
                )
                yield ChatWorkflowResultEvent(
                    result=_finalize_session_turn_result(
                        turn_result=result,
                        session_state=self._session_state,
                        active_context_start_turn=self._active_context_start_turn,
                        context_warning=self._context_warning,
                        system_message=self._system_message,
                    )
                )
                return

            yield ChatWorkflowStatusEvent(status="thinking")


def run_chat_turn(
    *,
    user_message: str,
    prior_messages: list[ChatMessage],
    root_path: Path,
    config: ChatConfig,
    llm_client: ChatLLMClient,
) -> ChatWorkflowTurnResult:
    """Run one orchestrated chat turn until completion or continuation boundary."""

    system_message = ChatMessage(
        role="system",
        content=build_chat_system_prompt(
            tool_limits=config.tool_limits,
            tools=build_chat_tool_definitions(),
        ),
    )
    user_chat_message = ChatMessage(role="user", content=user_message)
    messages: list[ChatMessage] = [system_message, *prior_messages, user_chat_message]
    new_messages: list[ChatMessage] = [user_chat_message]
    tool_results: list[ChatToolResult] = []
    round_count = 0
    executed_tool_call_count = 0
    last_token_usage = None
    tool_definitions = build_chat_tool_definitions()

    while True:
        response = llm_client.generate_chat_turn(
            ChatTurnRequest(
                messages=messages,
                response_model=ChatFinalResponse,
                model_name=config.llm.model_name,
                tools=tool_definitions,
                temperature=config.llm.temperature,
            )
        )
        last_token_usage = response.token_usage
        messages.append(response.assistant_message)
        new_messages.append(response.assistant_message)

        if response.finish_reason == "interrupted":
            raise LLMError(
                "Chat provider returned interrupted turn without final response"
            )

        if response.finish_reason == "final_response":
            final_response = ChatFinalResponse.model_validate(response.final_response)
            return ChatWorkflowTurnResult(
                status="completed",
                new_messages=new_messages,
                final_response=final_response,
                token_usage=last_token_usage,
                tool_results=tool_results,
            )

        if len(response.tool_calls) > config.session.max_tool_calls_per_round:
            return _build_continuation_result(
                new_messages=new_messages,
                tool_results=tool_results,
                token_usage=last_token_usage,
                reason=(
                    "The model requested more tool calls in one round than allowed. "
                    "User confirmation is required before continuing."
                ),
            )

        if (
            executed_tool_call_count + len(response.tool_calls)
            > config.session.max_total_tool_calls_per_turn
        ):
            return _build_continuation_result(
                new_messages=new_messages,
                tool_results=tool_results,
                token_usage=last_token_usage,
                reason=(
                    "The model needs more total tool-call budget before it can "
                    "continue this turn."
                ),
            )

        for tool_call in response.tool_calls:
            tool_result = execute_chat_tool_call(
                tool_call,
                root_path=root_path,
                config=config,
            )
            tool_results.append(tool_result)
            executed_tool_call_count += 1
            tool_message = ChatMessage(role="tool", tool_result=tool_result)
            messages.append(tool_message)
            new_messages.append(tool_message)

        round_count += 1
        if round_count >= config.session.max_tool_round_trips:
            return _build_continuation_result(
                new_messages=new_messages,
                tool_results=tool_results,
                token_usage=last_token_usage,
                reason=(
                    "The model needs more tool rounds before it can provide "
                    "a final response."
                ),
            )


def run_chat_session_turn(
    *,
    user_message: str,
    session_state: ChatSessionState,
    root_path: Path,
    config: ChatConfig,
    llm_client: ChatLLMClient,
) -> ChatWorkflowTurnResult:
    """Run one chat turn while maintaining in-memory visible session state."""

    (
        system_message,
        _user_chat_message,
        prior_messages,
        active_context_start_turn,
        context_warning,
    ) = _prepare_session_context(
        user_message=user_message,
        session_state=session_state,
        config=config,
    )
    turn_result = run_chat_turn(
        user_message=user_message,
        prior_messages=prior_messages,
        root_path=root_path,
        config=config,
        llm_client=llm_client,
    )
    return _finalize_session_turn_result(
        turn_result=turn_result,
        session_state=session_state,
        active_context_start_turn=active_context_start_turn,
        context_warning=context_warning,
        system_message=system_message,
    )


def run_interactive_chat_session_turn(
    *,
    user_message: str,
    session_state: ChatSessionState,
    root_path: Path,
    config: ChatConfig,
    llm_client: ChatLLMClient,
) -> ChatSessionTurnRunner:
    """Run one cancellable interactive chat turn while maintaining session state."""

    return ChatSessionTurnRunner(
        user_message=user_message,
        session_state=session_state,
        root_path=root_path,
        config=config,
        llm_client=llm_client,
    )


def run_streaming_chat_session_turn(
    *,
    user_message: str,
    session_state: ChatSessionState,
    root_path: Path,
    config: ChatConfig,
    llm_client: ChatLLMClient,
) -> ChatSessionTurnRunner:
    """Backward-compatible alias for the interactive turn runner."""

    return run_interactive_chat_session_turn(
        user_message=user_message,
        session_state=session_state,
        root_path=root_path,
        config=config,
        llm_client=llm_client,
    )
