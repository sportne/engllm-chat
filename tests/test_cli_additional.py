"""Standalone CLI tests."""

from __future__ import annotations

from pathlib import Path

from engllm_chat.cli.main import main
from engllm_chat.domain.models import ChatConfig


def _fake_chat_config() -> ChatConfig:
    return ChatConfig.model_validate(
        {
            "llm": {
                "provider": "ollama",
                "model_name": "qwen",
                "temperature": 0.1,
                "api_base_url": "http://127.0.0.1:11434",
            }
        }
    )


def test_interactive_cli_launches_app_with_overrides(
    monkeypatch, tmp_path: Path
) -> None:
    cli_module = __import__("engllm_chat.cli.main", fromlist=["main"])
    captured: dict[str, object] = {}
    logging_flags: list[bool] = []

    monkeypatch.setattr(
        cli_module, "load_chat_config", lambda _path: _fake_chat_config()
    )
    monkeypatch.setattr(
        cli_module,
        "_configure_verbose_llm_logging",
        lambda enabled: logging_flags.append(enabled),
    )

    def _fake_launch_chat_app(*, root_path, config, llm_client=None):
        captured["root_path"] = root_path
        captured["config"] = config
        captured["llm_client"] = llm_client
        return 0

    monkeypatch.setattr(cli_module, "_launch_chat_app", _fake_launch_chat_app)

    rc = main(
        [
            "interactive",
            str(tmp_path),
            "--config",
            str(tmp_path / "chat.yaml"),
            "--provider",
            "xai",
            "--model",
            "grok-4-fast",
            "--temperature",
            "0.3",
            "--api-base-url",
            "https://proxy.example/v1",
            "--verbose-llm",
            "--max-context-tokens",
            "1234",
            "--max-file-size-characters",
            "4096",
        ]
    )

    assert rc == 0
    resolved = captured["config"]
    assert isinstance(resolved, ChatConfig)
    assert resolved.llm.provider == "xai"
    assert resolved.llm.model_name == "grok-4-fast"
    assert resolved.llm.temperature == 0.3
    assert resolved.llm.api_base_url == "https://proxy.example/v1"
    assert resolved.llm.verbose_llm_logging is True
    assert resolved.session.max_context_tokens == 1234
    assert resolved.tool_limits.max_file_size_characters == 4096
    assert captured["root_path"] == tmp_path.resolve()
    assert captured["llm_client"] is None
    assert logging_flags == [True]


def test_interactive_cli_rejects_invalid_temperature(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    cli_module = __import__("engllm_chat.cli.main", fromlist=["main"])
    monkeypatch.setattr(
        cli_module, "load_chat_config", lambda _path: _fake_chat_config()
    )

    rc = main(
        [
            "interactive",
            str(tmp_path),
            "--config",
            str(tmp_path / "chat.yaml"),
            "--temperature",
            "2.0",
        ]
    )

    captured = capsys.readouterr()
    assert rc == 2
    assert "--temperature must be between 0.0 and 1.0" in captured.err


def test_probe_subcommand_delegates_to_probe_module(monkeypatch) -> None:
    cli_module = __import__("engllm_chat.cli.main", fromlist=["main"])
    captured: dict[str, object] = {}

    def _fake_probe(argv):
        captured["argv"] = argv
        return 7

    monkeypatch.setattr(cli_module, "probe_openai_api_main", _fake_probe)

    rc = main(
        [
            "probe-openai-api",
            "--base-url",
            "http://localhost:11434/v1",
            "--api-key",
            "dummy",
            "--include-images",
            "--json",
        ]
    )

    assert rc == 7
    assert captured["argv"] == [
        "--base-url",
        "http://localhost:11434/v1",
        "--api-key",
        "dummy",
        "--include-images",
        "--json",
    ]
