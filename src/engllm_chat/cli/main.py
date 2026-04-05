"""Standalone command-line interface for EngLLM Chat."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from engllm_chat.config import load_chat_config
from engllm_chat.domain.errors import EngLLMError
from engllm_chat.domain.models import ChatConfig
from engllm_chat.probe_openai_api import main as probe_openai_api_main


def _resolve_temperature(raw_value: float | None, default: float) -> float:
    resolved = default if raw_value is None else raw_value
    if resolved < 0.0 or resolved > 1.0:
        raise EngLLMError("--temperature must be between 0.0 and 1.0")
    return resolved


def _resolve_chat_config(args: argparse.Namespace) -> ChatConfig:
    base_config = load_chat_config(args.config)
    raw = base_config.model_dump(mode="python")
    raw.setdefault("llm", {})
    raw.setdefault("session", {})
    raw.setdefault("tool_limits", {})

    if args.provider is not None:
        raw["llm"]["provider"] = args.provider
    if args.model is not None:
        raw["llm"]["model_name"] = args.model
    raw["llm"]["temperature"] = _resolve_temperature(
        args.temperature, base_config.llm.temperature
    )
    if args.ollama_base_url is not None:
        raw["llm"]["ollama_base_url"] = args.ollama_base_url
    if args.api_base_url is not None:
        raw["llm"]["api_base_url"] = args.api_base_url

    session_overrides = {
        "max_context_tokens": args.max_context_tokens,
        "max_tool_round_trips": args.max_tool_round_trips,
        "max_tool_calls_per_round": args.max_tool_calls_per_round,
        "max_total_tool_calls_per_turn": args.max_total_tool_calls_per_turn,
    }
    for field_name, override in session_overrides.items():
        if override is not None:
            raw["session"][field_name] = override

    tool_limit_overrides = {
        "max_entries_per_call": args.max_entries_per_call,
        "max_recursive_depth": args.max_recursive_depth,
        "max_search_matches": args.max_search_matches,
        "max_read_lines": args.max_read_lines,
        "max_file_size_characters": args.max_file_size_characters,
        "max_tool_result_chars": args.max_tool_result_chars,
    }
    for field_name, override in tool_limit_overrides.items():
        if override is not None:
            raw["tool_limits"][field_name] = override

    return ChatConfig.model_validate(raw)


def _launch_chat_app(
    *,
    root_path: Path,
    config: ChatConfig,
    llm_client: Any | None = None,
) -> int:
    from engllm_chat.tools.chat.app import run_chat_app

    return run_chat_app(
        root_path=root_path,
        config=config,
        llm_client=llm_client,
    )


def _run_chat_interactive(args: argparse.Namespace) -> int:
    chat_config = _resolve_chat_config(args)
    return _launch_chat_app(
        root_path=args.directory.resolve(),
        config=chat_config,
    )


def _run_probe_openai_api(args: argparse.Namespace) -> int:
    argv: list[str] = []
    for field_name in (
        "base_url",
        "api_key",
        "text_model",
        "embedding_model",
        "image_model",
        "tts_model",
    ):
        value = getattr(args, field_name)
        if value is not None:
            argv.extend([f"--{field_name.replace('_', '-')}", value])

    if args.timeout_seconds is not None:
        argv.extend(["--timeout-seconds", str(args.timeout_seconds)])
    if args.include_images:
        argv.append("--include-images")
    if args.include_audio:
        argv.append("--include-audio")
    if args.json:
        argv.append("--json")
    if args.no_progress:
        argv.append("--no-progress")
    return probe_openai_api_main(argv)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="engllm-chat",
        description="Interactive directory-scoped chat over repository files.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    interactive = subparsers.add_parser(
        "interactive",
        help="Start an interactive read-only chat session for one directory.",
    )
    interactive.add_argument("directory", type=Path)
    interactive.add_argument("--config", required=True, type=Path)
    interactive.add_argument(
        "--provider",
        choices=["ollama", "mock", "openai", "xai", "anthropic", "gemini"],
    )
    interactive.add_argument("--model", type=str)
    interactive.add_argument("--temperature", type=float)
    interactive.add_argument("--ollama-base-url", type=str)
    interactive.add_argument("--api-base-url", type=str)
    interactive.add_argument("--max-context-tokens", type=int)
    interactive.add_argument("--max-tool-round-trips", type=int)
    interactive.add_argument("--max-tool-calls-per-round", type=int)
    interactive.add_argument("--max-total-tool-calls-per-turn", type=int)
    interactive.add_argument("--max-entries-per-call", type=int)
    interactive.add_argument("--max-recursive-depth", type=int)
    interactive.add_argument("--max-search-matches", type=int)
    interactive.add_argument("--max-read-lines", type=int)
    interactive.add_argument("--max-file-size-characters", type=int)
    interactive.add_argument("--max-tool-result-chars", type=int)
    interactive.set_defaults(run=_run_chat_interactive)

    probe = subparsers.add_parser(
        "probe-openai-api",
        help="Probe an OpenAI-compatible endpoint for supported API surfaces.",
    )
    probe.add_argument("--base-url", required=True)
    probe.add_argument("--api-key")
    probe.add_argument("--text-model")
    probe.add_argument("--embedding-model")
    probe.add_argument("--image-model")
    probe.add_argument("--tts-model")
    probe.add_argument("--timeout-seconds", type=float)
    probe.add_argument("--include-images", action="store_true")
    probe.add_argument("--include-audio", action="store_true")
    probe.add_argument("--json", action="store_true")
    probe.add_argument("--no-progress", action="store_true")
    probe.set_defaults(run=_run_probe_openai_api)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return int(args.run(args))
    except EngLLMError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
