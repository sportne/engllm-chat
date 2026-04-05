"""Offline tests for the opt-in chat smoke runner."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from engllm_chat.domain.models import (
    ChatFinalResponse,
    ChatLLMConfig,
    ChatMessage,
    ChatTokenUsage,
    ChatToolResult,
)
from engllm_chat.smoke_chat import main
from engllm_chat.smoke_ollama_chat import main as ollama_wrapper_main
from engllm_chat.tools.chat.models import ChatWorkflowTurnResult


def test_chat_smoke_reports_json_summary(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    smoke_module = __import__("engllm_chat.smoke_chat", fromlist=["main"])

    monkeypatch.setattr(
        smoke_module,
        "create_chat_llm_client",
        lambda config, api_key=None: object(),
    )
    monkeypatch.setattr(
        smoke_module,
        "run_chat_turn",
        lambda **kwargs: ChatWorkflowTurnResult(
            status="completed",
            new_messages=[
                ChatMessage(role="user", content="question"),
                ChatMessage(role="assistant", content='{"answer":"done"}'),
            ],
            final_response=ChatFinalResponse(answer="done"),
            token_usage=ChatTokenUsage(input_tokens=11, output_tokens=7),
            tool_results=[
                ChatToolResult(
                    call_id="call-1",
                    tool_name="search_text",
                    status="ok",
                    payload={"matches": []},
                )
            ],
        ),
    )

    rc = main(
        [
            "--directory",
            str(tmp_path),
            "--require-tool-call",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    assert rc == 0
    payload = json.loads(captured.out)
    assert payload["status"] == "completed"
    assert payload["tool_sequence"] == ["search_text"]
    assert payload["final_response"]["answer"] == "done"
    assert payload["directory"] == str(tmp_path.resolve())


def test_chat_smoke_fails_when_required_tool_call_is_missing(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    smoke_module = __import__("engllm_chat.smoke_chat", fromlist=["main"])

    monkeypatch.setattr(
        smoke_module,
        "create_chat_llm_client",
        lambda config, api_key=None: object(),
    )
    monkeypatch.setattr(
        smoke_module,
        "run_chat_turn",
        lambda **kwargs: ChatWorkflowTurnResult(
            status="completed",
            new_messages=[
                ChatMessage(role="user", content="question"),
                ChatMessage(role="assistant", content='{"answer":"done"}'),
            ],
            final_response=ChatFinalResponse(answer="done"),
            token_usage=ChatTokenUsage(input_tokens=4, output_tokens=3),
            tool_results=[],
        ),
    )

    rc = main(["--directory", str(tmp_path), "--require-tool-call"])

    captured = capsys.readouterr()
    assert rc == 2
    assert "expected at least one tool call" in captured.err


def test_chat_smoke_enables_verbose_llm_logging(
    monkeypatch,
    tmp_path: Path,
) -> None:
    smoke_module = __import__("engllm_chat.smoke_chat", fromlist=["main"])
    logger = logging.getLogger("engllm_chat.llm.openai_compatible")
    original_level = logger.level

    monkeypatch.setattr(
        smoke_module,
        "create_chat_llm_client",
        lambda config, api_key=None: object(),
    )
    monkeypatch.setattr(
        smoke_module,
        "run_chat_turn",
        lambda **kwargs: ChatWorkflowTurnResult(
            status="completed",
            new_messages=[
                ChatMessage(role="user", content="question"),
                ChatMessage(role="assistant", content='{"answer":"done"}'),
            ],
            final_response=ChatFinalResponse(answer="done"),
            token_usage=ChatTokenUsage(input_tokens=4, output_tokens=3),
            tool_results=[
                ChatToolResult(
                    call_id="call-1",
                    tool_name="search_text",
                    status="ok",
                    payload={"matches": []},
                )
            ],
        ),
    )

    try:
        rc = main(
            [
                "--directory",
                str(tmp_path),
                "--require-tool-call",
                "--verbose-llm",
            ]
        )
        assert rc == 0
        assert logger.level == logging.INFO
    finally:
        logger.setLevel(original_level)


def test_chat_smoke_supports_hosted_provider_arguments(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    smoke_module = __import__("engllm_chat.smoke_chat", fromlist=["main"])
    captured: dict[str, object] = {}

    def _fake_create_chat_llm_client(config, api_key=None):
        captured["config"] = config
        captured["api_key"] = api_key
        return object()

    monkeypatch.setattr(
        smoke_module,
        "create_chat_llm_client",
        _fake_create_chat_llm_client,
    )
    monkeypatch.setattr(
        smoke_module,
        "run_chat_turn",
        lambda **kwargs: ChatWorkflowTurnResult(
            status="completed",
            new_messages=[
                ChatMessage(role="user", content="question"),
                ChatMessage(role="assistant", content='{"answer":"done"}'),
            ],
            final_response=ChatFinalResponse(answer="done"),
            token_usage=ChatTokenUsage(input_tokens=4, output_tokens=3),
            tool_results=[
                ChatToolResult(
                    call_id="call-1",
                    tool_name="search_text",
                    status="ok",
                    payload={"matches": []},
                )
            ],
        ),
    )

    rc = main(
        [
            "--provider",
            "gemini",
            "--model",
            "gemini-test-model",
            "--base-url",
            "https://example.test/v1beta/openai/",
            "--api-key",
            "secret",
            "--directory",
            str(tmp_path),
            "--require-tool-call",
            "--json",
        ]
    )

    captured_output = capsys.readouterr()
    assert rc == 0
    assert captured["api_key"] == "secret"
    config = captured["config"]
    assert isinstance(config, ChatLLMConfig)
    assert config.provider == "gemini"
    assert config.model_name == "gemini-test-model"
    assert config.api_base_url == "https://example.test/v1beta/openai/"
    payload = json.loads(captured_output.out)
    assert payload["provider"] == "gemini"


def test_chat_smoke_requires_explicit_model_for_hosted_provider(
    tmp_path: Path,
    capsys,
) -> None:
    rc = main(
        [
            "--provider",
            "gemini",
            "--directory",
            str(tmp_path),
        ]
    )

    captured = capsys.readouterr()
    assert rc == 2
    assert "--model is required for provider 'gemini'" in captured.err


def test_ollama_smoke_wrapper_delegates_to_generic_module(monkeypatch) -> None:
    smoke_module = __import__("engllm_chat.smoke_chat", fromlist=["main"])
    captured: dict[str, object] = {}

    def _fake_main(argv=None):
        captured["argv"] = argv
        return 7

    monkeypatch.setattr(smoke_module, "main", _fake_main)

    rc = ollama_wrapper_main(["--provider", "ollama"])

    assert rc == 7
    assert captured["argv"] == ["--provider", "ollama"]
