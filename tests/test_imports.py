"""Import smoke tests for the standalone package surface."""

from __future__ import annotations

from importlib import import_module


def test_top_level_modules_import() -> None:
    module_names = [
        "engllm_chat",
        "engllm_chat.cli.main",
        "engllm_chat.config",
        "engllm_chat.core.chat",
        "engllm_chat.domain",
        "engllm_chat.llm",
        "engllm_chat.prompts.chat",
        "engllm_chat.tools.chat",
    ]

    for module_name in module_names:
        assert import_module(module_name)


def test_lazy_chat_exports_resolve() -> None:
    chat_package = import_module("engllm_chat.tools.chat")
    prompts_package = import_module("engllm_chat.prompts.chat")

    assert callable(chat_package.run_chat_turn)
    assert callable(chat_package.run_chat_session_turn)
    assert callable(chat_package.run_streaming_chat_session_turn)
    assert callable(prompts_package.build_chat_system_prompt)
