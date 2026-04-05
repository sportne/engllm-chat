"""Per-turn chat orchestration loop."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from pydantic import ValidationError as PydanticValidationError

from engllm_chat.core.chat import (
    find_files,
    get_file_info,
    list_directory,
    list_directory_recursive,
    read_file,
    search_text,
)
from engllm_chat.core.tokenize import tokenize
from engllm_chat.domain.errors import LLMError
from engllm_chat.domain.models import (
    ChatConfig,
    ChatFinalResponse,
    ChatMessage,
    ChatTokenUsage,
    ChatToolCall,
    ChatToolResult,
    DomainModel,
)
from engllm_chat.llm.base import (
    ChatLLMClient,
    ChatToolDefinition,
    ChatTurnRequest,
)
from engllm_chat.prompts.chat import build_chat_system_prompt
from engllm_chat.tools.chat.models import (
    ChatSessionState,
    ChatSessionTurnRecord,
    ChatWorkflowResultEvent,
    ChatWorkflowStatusEvent,
    ChatWorkflowTurnResult,
    FindFilesArgs,
    GetFileInfoArgs,
    ListDirectoryArgs,
    ListDirectoryRecursiveArgs,
    ReadFileArgs,
    SearchTextArgs,
)


@dataclass(frozen=True, slots=True)
class _ChatToolSpec:
    """One orchestrated chat tool definition and executor."""

    name: str
    description: str
    argument_model: type[DomainModel]
    runner: Callable[[DomainModel, Path, ChatConfig], DomainModel]


def _run_list_directory(
    args: DomainModel,
    root_path: Path,
    config: ChatConfig,
) -> DomainModel:
    typed_args = ListDirectoryArgs.model_validate(args.model_dump())
    return list_directory(
        root_path,
        typed_args.path,
        source_filters=config.source_filters,
        tool_limits=config.tool_limits,
    )


def _run_list_directory_recursive(
    args: DomainModel,
    root_path: Path,
    config: ChatConfig,
) -> DomainModel:
    typed_args = ListDirectoryRecursiveArgs.model_validate(args.model_dump())
    return list_directory_recursive(
        root_path,
        typed_args.path,
        source_filters=config.source_filters,
        tool_limits=config.tool_limits,
        max_depth=typed_args.max_depth,
    )


def _run_find_files(
    args: DomainModel, root_path: Path, config: ChatConfig
) -> DomainModel:
    typed_args = FindFilesArgs.model_validate(args.model_dump())
    return find_files(
        root_path,
        typed_args.pattern,
        typed_args.path,
        source_filters=config.source_filters,
        tool_limits=config.tool_limits,
    )


def _run_search_text(
    args: DomainModel,
    root_path: Path,
    config: ChatConfig,
) -> DomainModel:
    typed_args = SearchTextArgs.model_validate(args.model_dump())
    return search_text(
        root_path,
        typed_args.query,
        typed_args.path,
        source_filters=config.source_filters,
        tool_limits=config.tool_limits,
    )


def _run_get_file_info(
    args: DomainModel,
    root_path: Path,
    config: ChatConfig,
) -> DomainModel:
    typed_args = GetFileInfoArgs.model_validate(args.model_dump())
    return get_file_info(
        root_path,
        typed_args.path if typed_args.path is not None else typed_args.paths or [],
        session_config=config.session,
        tool_limits=config.tool_limits,
    )


def _run_read_file(
    args: DomainModel, root_path: Path, config: ChatConfig
) -> DomainModel:
    typed_args = ReadFileArgs.model_validate(args.model_dump())
    return read_file(
        root_path,
        typed_args.path,
        session_config=config.session,
        tool_limits=config.tool_limits,
        start_char=typed_args.start_char,
        end_char=typed_args.end_char,
    )


_TOOL_SPECS: tuple[_ChatToolSpec, ...] = (
    _ChatToolSpec(
        name="list_directory",
        description="List immediate children of one directory.",
        argument_model=ListDirectoryArgs,
        runner=_run_list_directory,
    ),
    _ChatToolSpec(
        name="list_directory_recursive",
        description="List one directory subtree as a flat depth-first result.",
        argument_model=ListDirectoryRecursiveArgs,
        runner=_run_list_directory_recursive,
    ),
    _ChatToolSpec(
        name="find_files",
        description="Find files by root-relative glob pattern.",
        argument_model=FindFilesArgs,
        runner=_run_find_files,
    ),
    _ChatToolSpec(
        name="search_text",
        description="Search readable file content for a literal substring.",
        argument_model=SearchTextArgs,
        runner=_run_search_text,
    ),
    _ChatToolSpec(
        name="get_file_info",
        description=(
            "Inspect one file or a small batch of files before deciding whether "
            "or how to read them."
        ),
        argument_model=GetFileInfoArgs,
        runner=_run_get_file_info,
    ),
    _ChatToolSpec(
        name="read_file",
        description=(
            "Read text or converted markdown content, optionally using "
            "start_char/end_char."
        ),
        argument_model=ReadFileArgs,
        runner=_run_read_file,
    ),
)

_TOOL_SPEC_BY_NAME = {spec.name: spec for spec in _TOOL_SPECS}


def _build_tool_definitions() -> list[ChatToolDefinition]:
    return [
        ChatToolDefinition(
            name=spec.name,
            description=spec.description,
            input_schema=spec.argument_model.model_json_schema(),
            argument_model=spec.argument_model,
        )
        for spec in _TOOL_SPECS
    ]


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
        content=build_chat_system_prompt(tool_limits=config.tool_limits),
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


def _execute_tool_call(
    tool_call: ChatToolCall,
    *,
    root_path: Path,
    config: ChatConfig,
) -> ChatToolResult:
    spec = _TOOL_SPEC_BY_NAME.get(tool_call.tool_name)
    if spec is None:
        return ChatToolResult(
            call_id=tool_call.call_id,
            tool_name=tool_call.tool_name,
            status="error",
            payload={},
            error_message=f"Unknown chat tool '{tool_call.tool_name}'",
        )

    try:
        arguments = spec.argument_model.model_validate(tool_call.arguments)
        payload_model = spec.runner(arguments, root_path, config)
        return ChatToolResult(
            call_id=tool_call.call_id,
            tool_name=tool_call.tool_name,
            status="ok",
            payload=payload_model.model_dump(mode="json"),
        )
    except (PydanticValidationError, ValueError, TypeError, Exception) as exc:
        return ChatToolResult(
            call_id=tool_call.call_id,
            tool_name=tool_call.tool_name,
            status="error",
            payload={},
            error_message=str(exc),
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
        tool_definitions = _build_tool_definitions()
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
                tool_result = _execute_tool_call(
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
        content=build_chat_system_prompt(tool_limits=config.tool_limits),
    )
    user_chat_message = ChatMessage(role="user", content=user_message)
    messages: list[ChatMessage] = [system_message, *prior_messages, user_chat_message]
    new_messages: list[ChatMessage] = [user_chat_message]
    tool_results: list[ChatToolResult] = []
    round_count = 0
    executed_tool_call_count = 0
    last_token_usage = None
    tool_definitions = _build_tool_definitions()

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
            tool_result = _execute_tool_call(
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
