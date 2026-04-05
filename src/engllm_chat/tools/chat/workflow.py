"""Per-turn chat orchestration loop."""

from __future__ import annotations

from pathlib import Path

from engllm_chat.domain.errors import LLMError
from engllm_chat.domain.models import (
    ChatConfig,
    ChatFinalResponse,
    ChatMessage,
    ChatToolResult,
)
from engllm_chat.llm.base import ChatLLMClient, ChatTurnRequest
from engllm_chat.tools.chat._workflow.context import (
    _build_system_message,
    _prepare_session_context,
)
from engllm_chat.tools.chat._workflow.results import (
    _build_continuation_result,
    _finalize_session_turn_result,
)
from engllm_chat.tools.chat._workflow.runner import ChatSessionTurnRunner
from engllm_chat.tools.chat.models import (
    ChatSessionState,
    ChatWorkflowTurnResult,
)
from engllm_chat.tools.chat.registry import (
    build_chat_tool_definitions,
    execute_chat_tool_call,
)


def run_chat_turn(
    *,
    user_message: str,
    prior_messages: list[ChatMessage],
    root_path: Path,
    config: ChatConfig,
    llm_client: ChatLLMClient,
) -> ChatWorkflowTurnResult:
    """Run one orchestrated chat turn until completion or continuation boundary."""

    system_message = _build_system_message(config)
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
