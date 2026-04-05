"""Interactive cancellable runner for chat workflow turns."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from threading import Lock

from engllm_chat.domain.models import (
    ChatConfig,
    ChatFinalResponse,
    ChatMessage,
    ChatTokenUsage,
    ChatToolResult,
)
from engllm_chat.llm.base import ChatLLMClient, ChatTurnRequest
from engllm_chat.tools.chat.models import (
    ChatSessionState,
    ChatWorkflowResultEvent,
    ChatWorkflowStatusEvent,
    ChatWorkflowTurnResult,
)
from engllm_chat.tools.chat.registry import (
    build_chat_tool_definitions,
    execute_chat_tool_call,
)

from .context import _prepare_session_context
from .results import (
    _build_continuation_result,
    _build_interrupted_result,
    _finalize_session_turn_result,
    _tool_status_label,
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
        self._session_state = session_state
        self._root_path = root_path
        self._config = config
        self._llm_client = llm_client
        self._lock = Lock()
        self._cancel_requested = False

    def cancel(self) -> None:
        with self._lock:
            self._cancel_requested = True

    def _finalized_event(
        self,
        *,
        result: ChatWorkflowTurnResult,
    ) -> ChatWorkflowResultEvent:
        return ChatWorkflowResultEvent(
            result=_finalize_session_turn_result(
                turn_result=result,
                session_state=self._session_state,
                active_context_start_turn=self._active_context_start_turn,
                context_warning=self._context_warning,
                system_message=self._system_message,
            )
        )

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
                yield self._finalized_event(result=result)
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
                yield self._finalized_event(result=result)
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
                yield self._finalized_event(result=result)
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
                yield self._finalized_event(result=result)
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
                yield self._finalized_event(result=result)
                return

            for tool_call in response.tool_calls:
                if self._cancel_requested:
                    result = _build_interrupted_result(
                        new_messages=new_messages,
                        tool_results=tool_results,
                        token_usage=last_token_usage,
                        reason="Interrupted by user.",
                    )
                    yield self._finalized_event(result=result)
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
                    yield self._finalized_event(result=result)
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
                yield self._finalized_event(result=result)
                return

            yield ChatWorkflowStatusEvent(status="thinking")
