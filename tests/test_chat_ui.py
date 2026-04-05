"""Textual UI tests for the interactive chat app shell."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

pytest.importorskip("textual")

from textual.containers import VerticalScroll
from textual.widgets import Button, Input, Static, TextArea

from engllm_chat.domain.errors import LLMError
from engllm_chat.domain.models import (
    ChatCitation,
    ChatConfig,
    ChatCredentialPromptMetadata,
    ChatFinalResponse,
    ChatMessage,
    ChatTokenUsage,
)
from engllm_chat.llm.mock import MockLLMClient
from engllm_chat.tools.chat.app import (
    ChatApp,
    ChatScreen,
    ComposerTextArea,
    CredentialModal,
    InterruptConfirmModal,
    TranscriptEntry,
    _format_citation,
    _format_final_response,
    run_chat_app,
)
from engllm_chat.tools.chat.models import (
    ChatSessionState,
    ChatWorkflowAssistantDeltaEvent,
    ChatWorkflowResultEvent,
    ChatWorkflowStatusEvent,
    ChatWorkflowTurnResult,
)


def _transcript_texts(app: ChatApp) -> list[str]:
    transcript = app.screen.query_one("#transcript", VerticalScroll)
    return [
        str(getattr(child, "renderable", child.render()))
        for child in transcript.children
    ]


def _static_text(screen: ChatScreen, selector: str) -> str:
    return str(screen.query_one(selector, Static).renderable)


def _completed_result(
    *,
    user_message: str,
    answer: str = "Done",
    confidence: float | None = 0.5,
    context_warning: str | None = None,
) -> ChatWorkflowTurnResult:
    return ChatWorkflowTurnResult(
        status="completed",
        new_messages=[
            ChatMessage(role="user", content=user_message),
            ChatMessage(role="assistant", content='{"answer":"Done","confidence":0.5}'),
        ],
        final_response=ChatFinalResponse(
            answer=answer,
            confidence=confidence,
        ),
        token_usage=ChatTokenUsage(
            total_tokens=9,
            session_tokens=9,
            active_context_tokens=360,
        ),
        session_state=ChatSessionState(),
        context_warning=context_warning,
    )


def _interrupted_result(
    *,
    user_message: str,
    partial_text: str,
    reason: str = "Interrupted by user.",
) -> ChatWorkflowTurnResult:
    return ChatWorkflowTurnResult(
        status="interrupted",
        new_messages=[
            ChatMessage(role="user", content=user_message),
            ChatMessage(
                role="assistant",
                content=partial_text,
                completion_state="interrupted",
            ),
        ],
        token_usage=ChatTokenUsage(
            total_tokens=5,
            session_tokens=5,
            active_context_tokens=100,
        ),
        session_state=ChatSessionState(),
        interruption_reason=reason,
    )


def _result_event(result: ChatWorkflowTurnResult) -> dict[str, object]:
    return ChatWorkflowResultEvent(result=result).model_dump(mode="json")


def _install_completed_turn_stub(
    screen: ChatScreen,
    *,
    answer: str = "Done",
    confidence: float | None = 0.5,
    context_warning: str | None = None,
) -> None:
    def _fake_run_turn_worker(user_message: str) -> None:
        screen._handle_turn_status(
            ChatWorkflowStatusEvent(status="thinking").model_dump(mode="json")
        )
        screen._handle_turn_delta(
            ChatWorkflowAssistantDeltaEvent(
                delta_text='{"answer":"Done"',
                accumulated_text='{"answer":"Done"',
            ).model_dump(mode="json")
        )
        screen._handle_turn_result(
            _result_event(
                _completed_result(
                    user_message=user_message,
                    answer=answer,
                    confidence=confidence,
                    context_warning=context_warning,
                )
            )
        )

    screen._run_turn_worker = _fake_run_turn_worker  # type: ignore[method-assign]


class _FakeCancelableStream:
    def __init__(self) -> None:
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


def test_chat_app_format_helpers_and_transcript_entry_rendering() -> None:
    assert (
        _format_citation(ChatCitation(source_path=Path("src/app.py"))) == "src/app.py"
    )
    assert (
        _format_citation(
            ChatCitation(source_path=Path("src/app.py"), line_start=4, line_end=6)
        )
        == "src/app.py:4-6"
    )

    formatted = _format_final_response(
        ChatFinalResponse(
            answer="Done",
            citations=[ChatCitation(source_path=Path("src/app.py"), line_start=2)],
            uncertainty=["Unclear"],
            missing_information=["TBD"],
            follow_up_suggestions=["Inspect src"],
        )
    )
    assert "Citations:" in formatted
    assert "Uncertainty:" in formatted
    assert "Missing Information:" in formatted
    assert "Follow-up Suggestions:" in formatted

    entry = TranscriptEntry(role="assistant", text="draft")
    entry.update_text("partial", assistant_completion_state="interrupted")
    assert "Assistant (interrupted):" in str(entry.render())
    assert "partial" in str(entry.render())
    assert "Error: nope" in str(TranscriptEntry(role="error", text="nope").render())


def test_chat_app_launches_with_shell_layout_and_startup_message(
    tmp_path: Path,
) -> None:
    async def _run() -> None:
        app = ChatApp(
            root_path=tmp_path,
            config=ChatConfig(),
            llm_client=MockLLMClient(),
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            screen.query_one("#transcript", VerticalScroll)
            screen.query_one("#composer", TextArea)
            screen.query_one("#send-button", Button)
            screen.query_one("#stop-button", Button)
            screen.query_one("#status-bar", Static)
            footer = _static_text(screen, "#footer-bar")
            assert "Enter send" in footer
            assert "Shift+Enter newline" in footer
            transcript_texts = _transcript_texts(app)
            assert any(str(tmp_path) in text for text in transcript_texts)
            assert any("/help" in text for text in transcript_texts)
            assert any("quit or exit" in text for text in transcript_texts)
            app.exit()
            await pilot.pause()

    asyncio.run(_run())


def test_chat_app_enter_sends_and_updates_transcript_and_footer(tmp_path: Path) -> None:
    async def _run() -> None:
        app = ChatApp(
            root_path=tmp_path,
            config=ChatConfig(),
            llm_client=MockLLMClient(),
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ChatScreen)
            _install_completed_turn_stub(screen)
            composer = screen.query_one("#composer", ComposerTextArea)
            composer.load_text("What is here?")
            composer.focus()
            await pilot.press("enter")
            await pilot.pause()
            transcript_texts = _transcript_texts(app)
            assert any("You:\nWhat is here?" in text for text in transcript_texts)
            assert any("Assistant:\nDone" in text for text in transcript_texts)
            footer = _static_text(screen, "#footer-bar")
            assert "session tokens: 9" in footer
            assert "active context tokens: 360" in footer
            assert "confidence: 0.50" in footer
            assert "thinking" not in "\n".join(transcript_texts)
            app.exit()
            await pilot.pause()

    asyncio.run(_run())


def test_chat_app_shift_enter_inserts_newline_without_sending(tmp_path: Path) -> None:
    async def _run() -> None:
        app = ChatApp(
            root_path=tmp_path,
            config=ChatConfig(),
            llm_client=MockLLMClient(),
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ChatScreen)
            submitted: list[str] = []
            screen._run_turn_worker = submitted.append  # type: ignore[method-assign]
            composer = screen.query_one("#composer", ComposerTextArea)
            composer.load_text("line one")
            composer.move_cursor((0, len("line one")))
            composer.focus()
            await pilot.press("shift+enter")
            await pilot.pause()
            assert composer.text == "line one\n"
            assert submitted == []
            transcript_texts = _transcript_texts(app)
            assert not any("You:\nline one" in text for text in transcript_texts)
            app.exit()
            await pilot.pause()

    asyncio.run(_run())


def test_chat_app_multiline_draft_and_status_updates_stay_out_of_transcript(
    tmp_path: Path,
) -> None:
    async def _run() -> None:
        app = ChatApp(
            root_path=tmp_path,
            config=ChatConfig(),
            llm_client=MockLLMClient(),
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ChatScreen)
            _install_completed_turn_stub(screen, answer="Multi-line answer")
            composer = screen.query_one("#composer", ComposerTextArea)
            composer.load_text("first line\nsecond line")
            await pilot.click("#send-button")
            await pilot.pause()
            transcript_texts = _transcript_texts(app)
            assert any(
                "You:\nfirst line\nsecond line" in text for text in transcript_texts
            )
            assert all("thinking" not in text for text in transcript_texts)
            assert all("drafting answer" not in text for text in transcript_texts)
            status = _static_text(screen, "#status-bar")
            assert status == ""
            app.exit()
            await pilot.pause()

    asyncio.run(_run())


def test_chat_app_shows_credential_modal_when_prompt_metadata_requires_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    created_api_keys: list[str | None] = []

    def _fake_create_chat_llm_client(config, **kwargs):
        del config
        created_api_keys.append(kwargs.get("api_key"))
        if kwargs.get("api_key") != "secret":
            raise LLMError("OPENAI_API_KEY is not configured")
        return MockLLMClient()

    monkeypatch.setattr(
        "engllm_chat.tools.chat.app.create_chat_llm_client",
        _fake_create_chat_llm_client,
    )

    async def _run() -> None:
        app = ChatApp(
            root_path=tmp_path,
            config=ChatConfig(),
            llm_client=None,
            credential_metadata_override=ChatCredentialPromptMetadata(
                provider="openai",
                api_key_env_var="OPENAI_API_KEY",
                prompt_for_api_key_if_missing=True,
                expects_api_key=True,
            ),
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = app.query_one(CredentialModal)
            credential_input = modal.query_one("#credential-input", Input)
            credential_input.value = "secret"
            await pilot.click("#credential-submit")
            await pilot.pause()
            assert isinstance(app.screen, ChatScreen)
            assert app.screen._credential_secret == "secret"
            assert isinstance(app.screen._llm_client, MockLLMClient)
            app.exit()
            await pilot.pause()

    asyncio.run(_run())
    assert created_api_keys == ["secret"]


def test_chat_app_credential_modal_cancel_dismisses_without_secret(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    created_api_keys: list[str | None] = []

    def _fake_create_chat_llm_client(config, **kwargs):
        del config
        created_api_keys.append(kwargs.get("api_key"))
        raise LLMError("OPENAI_API_KEY is not configured")

    monkeypatch.setattr(
        "engllm_chat.tools.chat.app.create_chat_llm_client",
        _fake_create_chat_llm_client,
    )

    async def _run() -> None:
        app = ChatApp(
            root_path=tmp_path,
            config=ChatConfig(),
            llm_client=None,
            credential_metadata_override=ChatCredentialPromptMetadata(
                provider="openai",
                api_key_env_var="OPENAI_API_KEY",
                prompt_for_api_key_if_missing=True,
                expects_api_key=True,
            ),
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one(CredentialModal)
            await pilot.click("#credential-cancel")
            await pilot.pause()
            chat_screen = app.screen
            assert isinstance(chat_screen, ChatScreen)
            assert chat_screen._credential_secret is None
            assert chat_screen._llm_client is None
            transcript_texts = _transcript_texts(app)
            assert any(
                "OPENAI_API_KEY is not configured" in text for text in transcript_texts
            )
            app.exit()
            await pilot.pause()

    asyncio.run(_run())
    assert created_api_keys == [None]


def test_chat_app_skips_credential_modal_when_env_key_is_present(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    created_api_keys: list[str | None] = []

    def _fake_create_chat_llm_client(config, **kwargs):
        del config
        created_api_keys.append(kwargs.get("api_key"))
        return MockLLMClient()

    monkeypatch.setenv("OPENAI_API_KEY", "env-secret")
    monkeypatch.setattr(
        "engllm_chat.tools.chat.app.create_chat_llm_client",
        _fake_create_chat_llm_client,
    )

    async def _run() -> None:
        app = ChatApp(
            root_path=tmp_path,
            config=ChatConfig(),
            llm_client=None,
            credential_metadata_override=ChatCredentialPromptMetadata(
                provider="openai",
                api_key_env_var="OPENAI_API_KEY",
                prompt_for_api_key_if_missing=True,
                expects_api_key=True,
            ),
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            assert isinstance(app.screen, ChatScreen)
            assert app.screen._credential_secret is None
            assert isinstance(app.screen._llm_client, MockLLMClient)
            assert not app.screen.query(CredentialModal)
            app.exit()
            await pilot.pause()

    asyncio.run(_run())
    assert created_api_keys == [None]


def test_chat_app_stop_action_visibility_and_busy_send_interrupt_modal(
    tmp_path: Path,
) -> None:
    async def _run() -> None:
        app = ChatApp(
            root_path=tmp_path,
            config=ChatConfig(),
            llm_client=MockLLMClient(),
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ChatScreen)
            stop_button = screen.query_one("#stop-button", Button)
            assert stop_button.disabled is True
            screen._busy = True
            screen._active_stream = _FakeCancelableStream()  # type: ignore[assignment]
            screen._refresh_footer()
            assert stop_button.disabled is False
            footer = _static_text(screen, "#footer-bar")
            assert "Stop active turn" in footer
            composer = screen.query_one("#composer", ComposerTextArea)
            composer.load_text("replacement draft")
            await pilot.click("#send-button")
            await pilot.pause()
            assert isinstance(app.screen, InterruptConfirmModal)
            await pilot.click("#interrupt-cancel")
            await pilot.pause()
            assert not isinstance(app.screen, InterruptConfirmModal)
            assert composer.text == "replacement draft"
            assert screen._active_stream is not None
            app.exit()
            await pilot.pause()

    asyncio.run(_run())


def test_chat_app_confirming_interrupt_cancels_and_restarts_with_new_draft(
    tmp_path: Path,
) -> None:
    async def _run() -> None:
        app = ChatApp(
            root_path=tmp_path,
            config=ChatConfig(),
            llm_client=MockLLMClient(),
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ChatScreen)
            fake_stream = _FakeCancelableStream()
            screen._busy = True
            screen._active_stream = fake_stream  # type: ignore[assignment]
            screen._refresh_footer()
            restarted: list[str] = []
            screen._run_turn_worker = restarted.append  # type: ignore[method-assign]
            composer = screen.query_one("#composer", ComposerTextArea)
            composer.load_text("new draft")
            await pilot.click("#send-button")
            await pilot.pause()
            assert isinstance(app.screen, InterruptConfirmModal)
            await pilot.click("#interrupt-confirm")
            await pilot.pause()
            assert fake_stream.cancelled is True
            screen._handle_turn_result(
                _result_event(
                    _interrupted_result(
                        user_message="old draft",
                        partial_text='{"answer":"partial"',
                    )
                )
            )
            await pilot.pause()
            assert restarted == ["new draft"]
            app.exit()
            await pilot.pause()

    asyncio.run(_run())


def test_chat_app_interrupted_output_remains_visible_and_marked(
    tmp_path: Path,
) -> None:
    async def _run() -> None:
        app = ChatApp(
            root_path=tmp_path,
            config=ChatConfig(),
            llm_client=MockLLMClient(),
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ChatScreen)
            screen._handle_turn_delta(
                ChatWorkflowAssistantDeltaEvent(
                    delta_text='{"answer":"partial"',
                    accumulated_text='{"answer":"partial"',
                ).model_dump(mode="json")
            )
            screen._handle_turn_result(
                _result_event(
                    _interrupted_result(
                        user_message="question",
                        partial_text='{"answer":"partial"',
                    )
                )
            )
            await pilot.pause()
            transcript_texts = _transcript_texts(app)
            assert any("Assistant (interrupted):" in text for text in transcript_texts)
            assert any('{"answer":"partial"' in text for text in transcript_texts)
            app.exit()
            await pilot.pause()

    asyncio.run(_run())


def test_chat_app_inline_commands_blank_submit_and_no_stream_cancel_are_handled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    async def _run() -> None:
        app = ChatApp(
            root_path=tmp_path,
            config=ChatConfig(),
            llm_client=MockLLMClient(),
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ChatScreen)

            sent: list[str] = []
            screen._run_turn_worker = sent.append  # type: ignore[method-assign]

            screen._submit_draft("   ")
            await pilot.pause()
            assert sent == []

            exit_calls: list[bool] = []
            monkeypatch.setattr(app, "exit", lambda: exit_calls.append(True))
            screen._submit_draft("/help")
            await pilot.pause()
            screen._submit_draft("quit")
            await pilot.pause()

            transcript_texts = _transcript_texts(app)
            assert any("Ask grounded questions" in text for text in transcript_texts)
            assert exit_calls == [True]

            screen._cancel_active_turn(status_text="stopping")
            screen._busy = False
            screen.handle_stop_button()
            app.exit()
            await pilot.pause()

    asyncio.run(_run())


def test_chat_app_handles_continuation_and_interrupted_without_content(
    tmp_path: Path,
) -> None:
    async def _run() -> None:
        app = ChatApp(
            root_path=tmp_path,
            config=ChatConfig(),
            llm_client=MockLLMClient(),
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ChatScreen)

            screen._handle_turn_result(
                _result_event(
                    ChatWorkflowTurnResult(
                        status="needs_continuation",
                        continuation_reason="Need more tool budget.",
                        new_messages=[ChatMessage(role="user", content="question")],
                        session_state=ChatSessionState(),
                    )
                )
            )
            screen._handle_turn_result(
                _result_event(
                    ChatWorkflowTurnResult(
                        status="interrupted",
                        interruption_reason="Stopped.",
                        new_messages=[
                            ChatMessage(role="user", content="question"),
                            ChatMessage(
                                role="assistant",
                                content=None,
                                completion_state="interrupted",
                            ),
                        ],
                        session_state=ChatSessionState(),
                    )
                )
            )
            await pilot.pause()

            transcript_texts = _transcript_texts(app)
            assert any("Need more tool budget." in text for text in transcript_texts)
            assert not any(
                "Assistant (interrupted):" in text for text in transcript_texts
            )
            app.exit()
            await pilot.pause()

    asyncio.run(_run())


def test_chat_app_handle_turn_error_resets_busy_state_and_focus(tmp_path: Path) -> None:
    async def _run() -> None:
        app = ChatApp(
            root_path=tmp_path,
            config=ChatConfig(),
            llm_client=MockLLMClient(),
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ChatScreen)
            screen._busy = True
            screen._active_stream = _FakeCancelableStream()  # type: ignore[assignment]
            screen._active_assistant_entry = screen._append_transcript(
                "assistant",
                "partial",
            )
            screen._handle_turn_error("boom")
            await pilot.pause()
            transcript_texts = _transcript_texts(app)
            assert any("Error: boom" in text for text in transcript_texts)
            assert screen._busy is False
            assert screen._active_stream is None
            assert screen._active_assistant_entry is None
            assert _static_text(screen, "#status-bar") == ""
            assert app.focused is screen.query_one("#composer", ComposerTextArea)
            app.exit()
            await pilot.pause()

    asyncio.run(_run())


def test_chat_app_completed_answer_renders_labeled_sections(tmp_path: Path) -> None:
    async def _run() -> None:
        app = ChatApp(
            root_path=tmp_path,
            config=ChatConfig(),
            llm_client=MockLLMClient(),
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ChatScreen)

            def _fake_run_turn_worker(user_message: str) -> None:
                screen._handle_turn_result(
                    _result_event(
                        ChatWorkflowTurnResult(
                            status="completed",
                            new_messages=[
                                ChatMessage(role="user", content=user_message),
                                ChatMessage(
                                    role="assistant",
                                    content='{"answer":"Done"}',
                                ),
                            ],
                            final_response=ChatFinalResponse(
                                answer="Done",
                                citations=[],
                                uncertainty=["Unsure"],
                                missing_information=["Missing"],
                                follow_up_suggestions=["Next"],
                                confidence=0.8,
                            ),
                            token_usage=ChatTokenUsage(
                                total_tokens=9,
                                session_tokens=9,
                                active_context_tokens=360,
                            ),
                            session_state=ChatSessionState(),
                        )
                    )
                )

            screen._run_turn_worker = _fake_run_turn_worker  # type: ignore[method-assign]
            composer = screen.query_one("#composer", ComposerTextArea)
            composer.load_text("hello")
            await pilot.press("enter")
            await pilot.pause()
            assistant_text = "\n\n".join(_transcript_texts(app))
            assert "Assistant:\nDone" in assistant_text
            assert "Uncertainty:\n- Unsure" in assistant_text
            assert "Missing Information:\n- Missing" in assistant_text
            assert "Follow-up Suggestions:\n- Next" in assistant_text
            app.exit()
            await pilot.pause()

    asyncio.run(_run())


def test_run_chat_app_launches_textual_app(monkeypatch, tmp_path: Path) -> None:
    launched: list[tuple[Path, str]] = []

    def _fake_run(self) -> None:
        launched.append((self._root_path, self._config.llm.model_name))

    monkeypatch.setattr(ChatApp, "run", _fake_run)

    result = run_chat_app(
        root_path=tmp_path,
        config=ChatConfig(),
        llm_client=MockLLMClient(),
    )

    assert result == 0
    assert launched == [(tmp_path, ChatConfig().llm.model_name)]
