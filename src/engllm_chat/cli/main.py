"""Standalone command-line interface for EngLLM Chat."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from engllm_chat.config import load_chat_config
from engllm_chat.domain.errors import EngLLMError
from engllm_chat.domain.models import ChatConfig
from engllm_chat.probe_openai_api import main as probe_openai_api_main

_CONFIG_EXAMPLES = """# Shared runtime defaults
# - Real provider-backed runs use ENGLLM_CHAT_API_KEY
# - Real provider-backed runs require an explicit api_base_url
# - Use --mock to run the deterministic mock client instead

# Ollama
llm:
  model_name: qwen2.5-coder:7b-instruct-q4_K_M
  api_base_url: http://127.0.0.1:11434/v1

# OpenAI
llm:
  model_name: gpt-5-mini
  api_base_url: https://api.openai.com/v1

# xAI
llm:
  model_name: grok-4-fast-reasoning
  api_base_url: https://api.x.ai/v1

# Anthropic
llm:
  model_name: claude-sonnet-4-5
  api_base_url: https://api.anthropic.com/v1/

# Gemini
llm:
  model_name: gemini-2.5-flash
  api_base_url: https://generativelanguage.googleapis.com/v1beta/openai/

# Mock
llm:
  model_name: mock-chat
"""


def _resolve_temperature(raw_value: float | None, default: float) -> float:
    resolved = default if raw_value is None else raw_value
    if resolved < 0.0 or resolved > 1.0:
        raise EngLLMError("--temperature must be between 0.0 and 1.0")
    return resolved


def _configure_verbose_llm_logging(enabled: bool) -> None:
    if not enabled:
        return

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("engllm_chat.llm.openai_compatible").setLevel(logging.INFO)


def _resolve_chat_config(args: argparse.Namespace) -> ChatConfig:
    base_config = load_chat_config(args.config)
    raw = base_config.model_dump(mode="python")
    raw.setdefault("llm", {})
    raw.setdefault("session", {})
    raw.setdefault("tool_limits", {})

    if args.model is not None:
        raw["llm"]["model_name"] = args.model
    raw["llm"]["temperature"] = _resolve_temperature(
        args.temperature, base_config.llm.temperature
    )
    if args.api_base_url is not None:
        raw["llm"]["api_base_url"] = args.api_base_url
    if args.verbose_llm:
        raw["llm"]["verbose_llm_logging"] = True

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
    mock_mode: bool = False,
    llm_client: Any | None = None,
) -> int:
    from engllm_chat.tools.chat.app import run_chat_app

    return run_chat_app(
        root_path=root_path,
        config=config,
        mock_mode=mock_mode,
        llm_client=llm_client,
    )


def _run_chat_interactive(args: argparse.Namespace) -> int:
    chat_config = _resolve_chat_config(args)
    _configure_verbose_llm_logging(chat_config.llm.verbose_llm_logging)
    return _launch_chat_app(
        root_path=args.directory.resolve(),
        config=chat_config,
        mock_mode=args.mock,
    )


def _run_config_examples(_args: argparse.Namespace) -> int:
    print(_CONFIG_EXAMPLES)
    return 0


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


def _add_chat_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("directory", type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--model", type=str)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--api-base-url", type=str)
    parser.add_argument("--verbose-llm", action="store_true")
    parser.add_argument("--max-context-tokens", type=int)
    parser.add_argument("--max-tool-round-trips", type=int)
    parser.add_argument("--max-tool-calls-per-round", type=int)
    parser.add_argument("--max-total-tool-calls-per-turn", type=int)
    parser.add_argument("--max-entries-per-call", type=int)
    parser.add_argument("--max-recursive-depth", type=int)
    parser.add_argument("--max-search-matches", type=int)
    parser.add_argument("--max-read-lines", type=int)
    parser.add_argument("--max-file-size-characters", type=int)
    parser.add_argument("--max-tool-result-chars", type=int)
    parser.set_defaults(run=_run_chat_interactive)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="engllm-chat",
        description="Interactive directory-scoped chat over repository files.",
        epilog=(
            "Launch chat by default with "
            "`engllm-chat <directory> --config <path>`. "
            "Use `engllm-chat probe-openai-api ...` for endpoint probing."
        ),
    )
    _add_chat_arguments(parser)
    return parser


def _build_probe_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="engllm-chat probe-openai-api",
        description="Probe an OpenAI-compatible endpoint for supported API surfaces.",
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key")
    parser.add_argument("--text-model")
    parser.add_argument("--embedding-model")
    parser.add_argument("--image-model")
    parser.add_argument("--tts-model")
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--include-images", action="store_true")
    parser.add_argument("--include-audio", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.set_defaults(run=_run_probe_openai_api)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_args = list(argv) if argv is not None else sys.argv[1:]
    if raw_args and raw_args[0] == "probe-openai-api":
        parser = _build_probe_parser()
        parse_args = raw_args[1:]
    elif raw_args and raw_args[0] == "config-examples":
        parser = argparse.ArgumentParser(
            prog="engllm-chat config-examples",
            description="Print example chat configuration files and runtime defaults.",
        )
        parser.set_defaults(run=_run_config_examples)
        parse_args = raw_args[1:]
    else:
        parser = build_parser()
        parse_args = (
            raw_args[1:] if raw_args and raw_args[0] == "interactive" else raw_args
        )
    args = parser.parse_args(parse_args)
    try:
        return int(args.run(args))
    except EngLLMError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
