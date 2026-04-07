"""Opt-in local smoke test for the chat workflow."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from engllm_chat.domain.errors import EngLLMError
from engllm_chat.domain.models import (
    ChatConfig,
    ChatFinalResponse,
    ChatTokenUsage,
    ChatToolResult,
)
from engllm_chat.llm.factory import create_chat_llm_client
from engllm_chat.tools.chat.workflow import run_chat_turn

if TYPE_CHECKING:
    from collections.abc import Sequence

DEFAULT_OLLAMA_MODEL = "qwen2.5:14b-instruct-q4_K_M"
DEFAULT_SMOKE_TEST_QUESTION = (
    "Without relying on prior knowledge, use the repository tools to identify "
    "the file that implements the shared OpenAI-compatible chat provider in "
    "this repo. Cite file evidence from the implementation."
)


class ChatSmokeSummary(BaseModel):
    """Serializable summary emitted by the local smoke test."""

    mode: str
    model: str
    base_url: str | None = None
    directory: Path
    question: str
    status: str
    tool_sequence: list[str] = Field(default_factory=list)
    tool_results: list[ChatToolResult] = Field(default_factory=list)
    final_response: ChatFinalResponse | None = None
    token_usage: ChatTokenUsage | None = None
    continuation_reason: str | None = None
    interruption_reason: str | None = None


class ChatSmokeTestError(EngLLMError):
    """Raised when a smoke-test expectation is not satisfied."""


def _configure_verbose_llm_logging(enabled: bool) -> None:
    if not enabled:
        return

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("engllm_chat.llm.openai_compatible").setLevel(logging.INFO)


def _resolve_model(model_name: str | None, *, use_mock: bool) -> str:
    if model_name is not None:
        cleaned = model_name.strip()
        if cleaned:
            return cleaned
        raise ChatSmokeTestError("--model must not be empty when provided")

    if use_mock:
        return "mock-chat"

    return DEFAULT_OLLAMA_MODEL


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m engllm_chat.smoke_chat",
        description="Run a real one-turn chat workflow smoke test.",
    )
    parser.add_argument("--mock", action="store_true")
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path("."),
        help="Root directory the chat workflow may inspect. Default: current directory.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Provider model name. Defaults to the Ollama smoke-test model for "
            "real OpenAI-compatible runs. Defaults to mock-chat for --mock."
        ),
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help=(
            "Override the provider base URL. Defaults to the provider-specific "
            "configured runtime default."
        ),
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help=(
            "Override the provider API key. If omitted, real runs use "
            "ENGLLM_CHAT_API_KEY."
        ),
    )
    parser.add_argument(
        "--question",
        default=DEFAULT_SMOKE_TEST_QUESTION,
        help="Question to send through the real chat workflow.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature to use for the smoke test. Default: 0.1",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=120.0,
        help="Per-request provider timeout in seconds. Default: 120.0",
    )
    parser.add_argument(
        "--max-tool-round-trips",
        type=int,
        default=8,
        help="Maximum tool round trips allowed during the smoke test. Default: 8",
    )
    parser.add_argument(
        "--max-tool-calls-per-round",
        type=int,
        default=2,
        help="Maximum tool calls allowed in one model round. Default: 2",
    )
    parser.add_argument(
        "--max-total-tool-calls-per-turn",
        type=int,
        default=10,
        help="Maximum total tool calls allowed in the smoke-test turn. Default: 10",
    )
    parser.add_argument(
        "--require-tool-call",
        action="store_true",
        help="Fail if the model answers without making at least one tool call.",
    )
    parser.add_argument(
        "--expect-tool",
        action="append",
        default=[],
        help="Require a specific tool name to appear at least once. Repeatable.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the smoke-test summary as JSON.",
    )
    parser.add_argument(
        "--verbose-llm",
        action="store_true",
        help="Enable verbose request/response logging for the LLM adapter.",
    )
    return parser


def _build_config(args: argparse.Namespace) -> ChatConfig:
    model_name = _resolve_model(args.model, use_mock=args.mock)
    if not args.mock and args.base_url is None:
        raise ChatSmokeTestError("--base-url is required unless --mock is used")

    return ChatConfig.model_validate(
        {
            "llm": {
                "model_name": model_name,
                "temperature": args.temperature,
                "api_base_url": args.base_url,
                "timeout_seconds": args.timeout_seconds,
                "verbose_llm_logging": args.verbose_llm,
            },
            "session": {
                "max_tool_round_trips": args.max_tool_round_trips,
                "max_tool_calls_per_round": args.max_tool_calls_per_round,
                "max_total_tool_calls_per_turn": args.max_total_tool_calls_per_turn,
            },
        }
    )


def _run_smoke_test(args: argparse.Namespace) -> ChatSmokeSummary:
    config = _build_config(args)
    llm_client = create_chat_llm_client(
        config.llm,
        use_mock=args.mock,
        api_key=args.api_key,
    )
    root_path = args.directory.resolve()
    turn_result = run_chat_turn(
        user_message=args.question,
        prior_messages=[],
        root_path=root_path,
        config=config,
        llm_client=llm_client,
    )
    return ChatSmokeSummary(
        mode="mock" if args.mock else "openai-compatible",
        model=config.llm.model_name,
        base_url=config.llm.api_base_url,
        directory=root_path,
        question=args.question,
        status=turn_result.status,
        tool_sequence=[
            tool_result.tool_name for tool_result in turn_result.tool_results
        ],
        tool_results=turn_result.tool_results,
        final_response=turn_result.final_response,
        token_usage=turn_result.token_usage,
        continuation_reason=turn_result.continuation_reason,
        interruption_reason=turn_result.interruption_reason,
    )


def _validate_expectations(
    summary: ChatSmokeSummary,
    *,
    require_tool_call: bool,
    expected_tools: list[str],
) -> None:
    if summary.status != "completed":
        reason = summary.continuation_reason or summary.interruption_reason
        detail = f" ({reason})" if reason else ""
        raise ChatSmokeTestError(
            f"Smoke test did not complete successfully: {summary.status}{detail}"
        )

    if summary.final_response is None:
        raise ChatSmokeTestError(
            "Smoke test completed without a final response payload."
        )

    if require_tool_call and not summary.tool_sequence:
        raise ChatSmokeTestError(
            "Smoke test expected at least one tool call, but the model answered "
            "without using tools."
        )

    missing_tools = [
        tool_name
        for tool_name in expected_tools
        if tool_name not in summary.tool_sequence
    ]
    if missing_tools:
        observed = ", ".join(summary.tool_sequence) or "(none)"
        missing = ", ".join(missing_tools)
        raise ChatSmokeTestError(
            f"Smoke test expected tool(s) {missing}, but observed {observed}."
        )


def _print_human_summary(summary: ChatSmokeSummary) -> None:
    print("Chat smoke test passed.")
    print(f"Mode: {summary.mode}")
    print(f"Directory: {summary.directory}")
    print(f"Model: {summary.model}")
    if summary.base_url is not None:
        print(f"Base URL: {summary.base_url}")
    tool_sequence = ", ".join(summary.tool_sequence) or "(none)"
    print(f"Tool sequence: {tool_sequence}")
    if summary.token_usage is not None:
        print(f"Total tokens: {summary.token_usage.total_tokens}")
    if summary.final_response is not None:
        print(f"Answer: {summary.final_response.answer}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    _configure_verbose_llm_logging(args.verbose_llm)

    try:
        summary = _run_smoke_test(args)
        _validate_expectations(
            summary,
            require_tool_call=args.require_tool_call,
            expected_tools=list(args.expect_tool),
        )
    except EngLLMError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(summary.model_dump(mode="json"), indent=2, sort_keys=True))
    else:
        _print_human_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
