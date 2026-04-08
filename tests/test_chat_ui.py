"""Textual UI tests for the interactive chat app shell."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest

pytest.importorskip("textual")

from textual.containers import VerticalScroll
from textual.document._document import Selection
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
    AssistantMarkdownEntry,
    ChatApp,
    ChatScreen,
    ComposerTextArea,
    CredentialModal,
    InterruptConfirmModal,
    TranscriptCopyModal,
    TranscriptEntry,
    run_chat_app,
)
from engllm_chat.tools.chat.models import (
    ChatSessionState,
    ChatWorkflowResultEvent,
    ChatWorkflowStatusEvent,
    ChatWorkflowTurnResult,
)
from engllm_chat.tools.chat.presentation import format_citation, format_final_response


@pytest.fixture(autouse=True)
def _speed_up_status_timing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "engllm_chat.tools.chat.controller._STATUS_MIN_DISPLAY_SECONDS",
        0.05,
    )
    monkeypatch.setattr(
        "engllm_chat.tools.chat.controller._STATUS_ANIMATION_INTERVAL_SECONDS",
        0.02,
    )


def _transcript_texts(app: ChatApp) -> list[str]:
    transcript = app.screen.query_one("#transcript", VerticalScroll)
    texts: list[str] = []
    for child in transcript.children:
        if isinstance(child, AssistantMarkdownEntry):
            texts.append(child.transcript_text)
            continue
        texts.append(str(getattr(child, "renderable", child.render())))
    return texts


def _assistant_markdown_entries(screen: ChatScreen) -> list[AssistantMarkdownEntry]:
    return list(screen.query(AssistantMarkdownEntry))


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


class _GatedTurnStream:
    def __init__(
        self,
        *,
        result: ChatWorkflowTurnResult,
        gate: threading.Event | None = None,
        statuses: tuple[str, ...] = ("thinking",),
    ) -> None:
        self._result = result
        self._gate = gate
        self._statuses = statuses

    def __iter__(self):
        for status in self._statuses:
            yield ChatWorkflowStatusEvent(status=status)
        if self._gate is not None:
            self._gate.wait(timeout=1.0)
        yield ChatWorkflowResultEvent(result=self._result)


def test_chat_app_format_helpers_and_transcript_entry_rendering() -> None:
    assert format_citation(ChatCitation(source_path=Path("src/app.py"))) == "src/app.py"
    assert (
        format_citation(
            ChatCitation(source_path=Path("src/app.py"), line_start=4, line_end=6)
        )
        == "src/app.py:4-6"
    )

    formatted = format_final_response(
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
    assert entry.has_class("transcript-entry")
    assert entry.has_class("assistant")
    entry.update_text("partial", assistant_completion_state="interrupted")
    assert "Assistant (interrupted):" in str(entry.render())
    assert "partial" in str(entry.render())
    user_entry = TranscriptEntry(role="user", text="question")
    assert user_entry.has_class("transcript-entry")
    assert user_entry.has_class("user")
    assert "Error: nope" in str(TranscriptEntry(role="error", text="nope").render())
    markdown_entry = AssistantMarkdownEntry(markdown_text="1. **Bold** and `code`")
    assert markdown_entry.has_class("transcript-entry")
    assert markdown_entry.has_class("assistant")
    assert markdown_entry.transcript_text.startswith("Assistant:\n1. **Bold**")
    assert ".transcript-entry.user" in ChatScreen.DEFAULT_CSS
    assert ".transcript-entry.assistant" in ChatScreen.DEFAULT_CSS
    assert "transcript-entry-markdown" in markdown_entry.classes


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
            chat_layout = screen.query_one("#chat-layout")
            assert [child.id for child in chat_layout.children] == [
                "transcript",
                "status-bar",
                "composer-row",
                "footer-bar",
            ]
            screen.query_one("#transcript", VerticalScroll)
            status_bar = screen.query_one("#status-bar", Static)
            composer_row = screen.query_one("#composer-row")
            composer_actions = screen.query_one("#composer-actions")
            screen.query_one("#composer", TextArea)
            screen.query_one("#send-button", Button)
            screen.query_one("#stop-button", Button)
            assert status_bar.parent is chat_layout
            assert composer_row.parent is chat_layout
            assert composer_actions.parent is composer_row
            assert [child.id for child in composer_actions.children] == [
                "send-button",
                "stop-button",
            ]
            footer = _static_text(screen, "#footer-bar")
            assert "Enter send" in footer
            assert "Shift+Enter newline" in footer
            assert "F6 copy transcript" in footer
            transcript_texts = _transcript_texts(app)
            assert any(str(tmp_path) in text for text in transcript_texts)
            assert any("Model: mock-engllm" in text for text in transcript_texts)
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
            await pilot.pause(0.06)
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
            await pilot.pause(0.06)
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
            raise LLMError("ENGLLM_CHAT_API_KEY is not configured")
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
                api_key_env_var="ENGLLM_CHAT_API_KEY",
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
        raise LLMError("ENGLLM_CHAT_API_KEY is not configured")

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
                api_key_env_var="ENGLLM_CHAT_API_KEY",
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
                "ENGLLM_CHAT_API_KEY is not configured" in text
                for text in transcript_texts
            )
            app.exit()
            await pilot.pause()

    asyncio.run(_run())
    assert created_api_keys == [None]


def test_chat_app_still_shows_credential_modal_when_env_key_is_present(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    created_api_keys: list[str | None] = []

    def _fake_create_chat_llm_client(config, **kwargs):
        del config
        created_api_keys.append(kwargs.get("api_key"))
        return MockLLMClient()

    monkeypatch.setenv("ENGLLM_CHAT_API_KEY", "env-secret")
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
                api_key_env_var="ENGLLM_CHAT_API_KEY",
                prompt_for_api_key_if_missing=True,
                expects_api_key=True,
            ),
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one(CredentialModal)
            await pilot.click("#credential-submit")
            await pilot.pause()
            assert isinstance(app.screen, ChatScreen)
            assert app.screen._credential_secret is None
            assert isinstance(app.screen._llm_client, MockLLMClient)
            app.exit()
            await pilot.pause()

    asyncio.run(_run())
    assert created_api_keys == [None]


def test_chat_app_shows_credential_modal_for_mock_mode_startup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    created_use_mock: list[bool] = []

    def _fake_create_chat_llm_client(config, **kwargs):
        del config
        created_use_mock.append(bool(kwargs.get("use_mock")))
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
            mock_mode=True,
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one(CredentialModal)
            await pilot.click("#credential-submit")
            await pilot.pause()
            assert isinstance(app.screen._llm_client, MockLLMClient)
            app.exit()
            await pilot.pause()

    asyncio.run(_run())
    assert created_use_mock == [True]


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
            screen._active_runner = _FakeCancelableStream()  # type: ignore[assignment]
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
            assert screen._active_runner is not None
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
            screen._active_runner = fake_stream  # type: ignore[assignment]
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
            screen._active_assistant_entry = screen._append_transcript("assistant", "")
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
            screen._submit_draft("/copy")
            await pilot.pause()
            assert isinstance(app.screen, TranscriptCopyModal)
            await pilot.click("#transcript-copy-close")
            await pilot.pause()
            screen._submit_draft("quit")
            await pilot.pause()

            transcript_texts = _transcript_texts(app)
            assert any("Ask grounded questions" in text for text in transcript_texts)
            assert any("Use /model" in text for text in transcript_texts)
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
            await pilot.pause(0.06)

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
            screen._active_runner = _FakeCancelableStream()  # type: ignore[assignment]
            screen._active_assistant_entry = screen._append_transcript(
                "assistant",
                "partial",
            )
            screen._handle_turn_error("boom")
            await pilot.pause()
            transcript_texts = _transcript_texts(app)
            assert any("Error: boom" in text for text in transcript_texts)
            assert screen._busy is False
            assert screen._active_runner is None
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
                                citations=[
                                    ChatCitation(
                                        source_path=Path("src/app.py"),
                                        line_start=2,
                                    )
                                ],
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
            await pilot.pause(0.06)
            assistant_text = "\n\n".join(_transcript_texts(app))
            assert "Assistant:\nDone" in assistant_text
            assert "Citations:\n- src/app.py:2" in assistant_text
            assert "Uncertainty:\n- Unsure" in assistant_text
            assert "Missing Information:\n- Missing" in assistant_text
            assert "Follow-up Suggestions:\n- Next" in assistant_text
            assistant_entries = _assistant_markdown_entries(screen)
            assert len(assistant_entries) == 1
            assert assistant_entries[0].markdown_text.startswith("Done")
            assert "Citations:\n- src/app.py:2" in assistant_entries[0].metadata_text
            app.exit()
            await pilot.pause()

    asyncio.run(_run())


def test_chat_app_completed_answer_uses_markdown_widget_and_falls_back_cleanly(
    monkeypatch: pytest.MonkeyPatch,
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
                    _completed_result(
                        user_message="question",
                        answer="1. **Bold** answer with `code`",
                    )
                )
            )
            await pilot.pause()

            assistant_entries = _assistant_markdown_entries(screen)
            assert len(assistant_entries) == 1
            assert assistant_entries[0].markdown_text.startswith(
                "1. **Bold** answer with `code`"
            )

            class _BrokenAssistantMarkdownEntry:
                def __init__(self, *, markdown_text: str) -> None:
                    del markdown_text
                    raise RuntimeError("boom")

            monkeypatch.setattr(
                "engllm_chat.tools.chat.controller.AssistantMarkdownEntry",
                _BrokenAssistantMarkdownEntry,
            )

            screen._handle_turn_result(
                _result_event(
                    ChatWorkflowTurnResult(
                        status="completed",
                        new_messages=[
                            ChatMessage(role="user", content="question"),
                            ChatMessage(role="assistant", content='{"answer":"Done"}'),
                        ],
                        final_response=ChatFinalResponse(
                            answer="Fallback answer",
                            uncertainty=["Unsure"],
                            confidence=0.6,
                        ),
                        session_state=ChatSessionState(),
                    )
                )
            )
            await pilot.pause()

            transcript_texts = _transcript_texts(app)
            assert any(
                "Assistant:\nFallback answer\n\nUncertainty:\n- Unsure" in text
                for text in transcript_texts
            )
            assert len(_assistant_markdown_entries(screen)) == 1
            app.exit()
            await pilot.pause()

    asyncio.run(_run())


def test_chat_app_worker_path_updates_busy_state_and_completes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    gate = threading.Event()

    def _fake_run_interactive_chat_session_turn(**kwargs):
        assert kwargs["user_message"] == "What is here?"
        return _GatedTurnStream(
            result=_completed_result(user_message="What is here?"),
            gate=gate,
        )

    monkeypatch.setattr(
        "engllm_chat.tools.chat.app.run_interactive_chat_session_turn",
        _fake_run_interactive_chat_session_turn,
    )

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

            composer = screen.query_one("#composer", ComposerTextArea)
            composer.load_text("What is here?")
            await pilot.click("#send-button")

            transcript_texts: list[str] = []
            for _ in range(20):
                await pilot.pause(0.01)
                transcript_texts = _transcript_texts(app)
                if screen._busy or any(
                    "Assistant:\nDone" in text for text in transcript_texts
                ):
                    break

            stop_button = screen.query_one("#stop-button", Button)
            assert any("You:\nWhat is here?" in text for text in _transcript_texts(app))
            if screen._busy:
                assert stop_button.disabled is False
                assert _static_text(screen, "#status-bar").startswith("thinking")
                assert "Stop active turn" in _static_text(screen, "#footer-bar")

            gate.set()
            for _ in range(50):
                await pilot.pause()
                transcript_texts = _transcript_texts(app)
                if (
                    not screen._busy
                    and _static_text(screen, "#status-bar") == ""
                    and any("Assistant:\nDone" in text for text in transcript_texts)
                ):
                    break

            transcript_texts = _transcript_texts(app)
            assert screen._busy is False
            assert stop_button.disabled is True
            assert _static_text(screen, "#status-bar") == ""
            assert any("Assistant:\nDone" in text for text in transcript_texts)
            app.exit()
            await pilot.pause()

    asyncio.run(_run())


def test_chat_app_worker_path_handles_continuation_and_allows_follow_up_submit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    call_messages: list[str] = []

    def _fake_run_interactive_chat_session_turn(**kwargs):
        user_message = kwargs["user_message"]
        call_messages.append(user_message)
        if len(call_messages) == 1:
            return _GatedTurnStream(
                result=ChatWorkflowTurnResult(
                    status="needs_continuation",
                    continuation_reason="Need more tool budget.",
                    new_messages=[ChatMessage(role="user", content=user_message)],
                    session_state=ChatSessionState(),
                )
            )
        return _GatedTurnStream(
            result=_completed_result(
                user_message=user_message,
                answer="Follow-up complete",
                confidence=0.8,
            )
        )

    monkeypatch.setattr(
        "engllm_chat.tools.chat.app.run_interactive_chat_session_turn",
        _fake_run_interactive_chat_session_turn,
    )

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

            composer = screen.query_one("#composer", ComposerTextArea)
            composer.load_text("first question")
            await pilot.click("#send-button")
            for _ in range(50):
                await pilot.pause()
                if not screen._busy:
                    break
            await pilot.pause(0.06)

            transcript_texts = _transcript_texts(app)
            assert any("Need more tool budget." in text for text in transcript_texts)
            assert screen._busy is False
            assert _static_text(screen, "#status-bar") == ""
            await screen.workers.wait_for_complete()

            restarted: list[str] = []
            screen._run_turn_worker = restarted.append  # type: ignore[method-assign]
            screen._submit_draft("second question")
            await pilot.pause()
            assert restarted == ["second question"]

            screen._handle_turn_result(
                _result_event(
                    _completed_result(
                        user_message="second question",
                        answer="Follow-up complete",
                        confidence=0.8,
                    )
                )
            )
            await pilot.pause()

            transcript_texts = _transcript_texts(app)
            assert call_messages == ["first question"]
            assert any("You:\nfirst question" in text for text in transcript_texts)
            assert any("You:\nsecond question" in text for text in transcript_texts)
            assert any(
                "Assistant:\nFollow-up complete" in text for text in transcript_texts
            )
            app.exit()
            await pilot.pause()

    asyncio.run(_run())


def test_chat_app_worker_path_recovers_from_provider_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    call_messages: list[str] = []

    def _fake_run_interactive_chat_session_turn(**kwargs):
        user_message = kwargs["user_message"]
        call_messages.append(user_message)
        if len(call_messages) == 1:
            raise LLMError("boom")
        return _GatedTurnStream(
            result=_completed_result(
                user_message=user_message,
                answer="Recovered",
                confidence=0.9,
            )
        )

    monkeypatch.setattr(
        "engllm_chat.tools.chat.app.run_interactive_chat_session_turn",
        _fake_run_interactive_chat_session_turn,
    )

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

            composer = screen.query_one("#composer", ComposerTextArea)
            composer.load_text("first try")
            await pilot.click("#send-button")
            for _ in range(50):
                await pilot.pause()
                transcript_texts = _transcript_texts(app)
                if not screen._busy and any(
                    "Error: boom" in text for text in transcript_texts
                ):
                    break

            transcript_texts = _transcript_texts(app)
            assert any("Error: boom" in text for text in transcript_texts)
            assert screen._busy is False
            assert screen._active_runner is None
            assert screen._pending_interrupt_draft is None
            assert screen.query_one("#stop-button", Button).disabled is True
            await screen.workers.wait_for_complete()

            restarted: list[str] = []
            screen._run_turn_worker = restarted.append  # type: ignore[method-assign]
            screen._submit_draft("second try")
            await pilot.pause()
            assert restarted == ["second try"]

            screen._handle_turn_result(
                _result_event(
                    _completed_result(
                        user_message="second try",
                        answer="Recovered",
                        confidence=0.9,
                    )
                )
            )
            await pilot.pause()

            transcript_texts = _transcript_texts(app)
            assert call_messages == ["first try"]
            assert any("Assistant:\nRecovered" in text for text in transcript_texts)
            assert app.focused is screen.query_one("#composer", ComposerTextArea)
            app.exit()
            await pilot.pause()

    asyncio.run(_run())


def test_chat_app_status_bar_holds_brief_updates_before_clearing(
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

            screen._set_status("thinking")
            assert _static_text(screen, "#status-bar") == "thinking"

            screen._set_status("")
            await pilot.pause()
            assert _static_text(screen, "#status-bar").startswith("thinking")

            await pilot.pause(0.06)
            assert _static_text(screen, "#status-bar") == ""
            app.exit()
            await pilot.pause()

    asyncio.run(_run())


def test_chat_app_status_bar_animates_active_status_text(tmp_path: Path) -> None:
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

            screen._set_status("searching text")
            observed = [_static_text(screen, "#status-bar")]
            await pilot.pause(0.03)
            observed.append(_static_text(screen, "#status-bar"))
            await pilot.pause(0.03)
            observed.append(_static_text(screen, "#status-bar"))
            await pilot.pause(0.03)
            observed.append(_static_text(screen, "#status-bar"))

            assert observed[0] == "searching text"
            assert all(text.startswith("searching text") for text in observed)
            assert len(set(observed)) >= 2
            assert any(text.endswith(".") for text in observed[1:])

            screen._set_status("")
            await pilot.pause(0.06)
            assert _static_text(screen, "#status-bar") == ""
            app.exit()
            await pilot.pause()

    asyncio.run(_run())


def test_chat_app_transcript_copy_modal_supports_selection_and_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copied_texts: list[str] = []

    async def _run() -> None:
        app = ChatApp(
            root_path=tmp_path,
            config=ChatConfig(),
            llm_client=MockLLMClient(),
        )
        monkeypatch.setattr(app, "copy_to_clipboard", copied_texts.append)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ChatScreen)

            screen._append_transcript("user", "Where did that come from?")
            screen._handle_turn_result(
                _result_event(
                    ChatWorkflowTurnResult(
                        status="completed",
                        new_messages=[
                            ChatMessage(
                                role="user", content="Where did that come from?"
                            ),
                            ChatMessage(role="assistant", content='{"answer":"Done"}'),
                        ],
                        final_response=ChatFinalResponse(
                            answer="Found it in `README.md`.",
                            citations=[
                                ChatCitation(
                                    source_path=Path("README.md"),
                                    line_start=1,
                                    line_end=4,
                                )
                            ],
                            confidence=0.8,
                        ),
                        session_state=ChatSessionState(),
                    )
                )
            )
            await pilot.pause(0.06)

            await pilot.press("f6")
            await pilot.pause()
            assert isinstance(app.screen, TranscriptCopyModal)

            text_area = app.screen.query_one("#transcript-copy-area", TextArea)
            assert text_area.read_only is True
            assert "You:\nWhere did that come from?" in text_area.text
            assert "Assistant:\nFound it in `README.md`." in text_area.text
            assert "Citations:\n- README.md:1-4" in text_area.text

            text_area.selection = Selection((0, 0), (0, 3))
            await pilot.click("#transcript-copy-selection")
            await pilot.pause()
            assert copied_texts[-1]
            assert (
                _static_text(app.screen, "#transcript-copy-status")
                == "Copied selection."
            )

            await pilot.click("#transcript-copy-all")
            await pilot.pause()
            assert copied_texts[-1] == text_area.text
            assert (
                _static_text(app.screen, "#transcript-copy-status")
                == "Copied full transcript."
            )

            await pilot.click("#transcript-copy-close")
            await pilot.pause()
            assert isinstance(app.screen, ChatScreen)
            app.exit()
            await pilot.pause()

    asyncio.run(_run())


def test_chat_app_model_command_lists_current_and_available_models(
    tmp_path: Path,
) -> None:
    class _ListingMockClient(MockLLMClient):
        def list_available_models(self) -> list[str]:
            return ["qwen", "qwen-coder", "qwen-max"]

    async def _run() -> None:
        app = ChatApp(
            root_path=tmp_path,
            config=ChatConfig(),
            llm_client=_ListingMockClient(model_name="qwen"),
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ChatScreen)

            screen._submit_draft("/model")
            await pilot.pause()

            transcript_texts = _transcript_texts(app)
            assert any("Current model: qwen" in text for text in transcript_texts)
            assert any(
                "Available models:\n- qwen\n- qwen-coder\n- qwen-max" in text
                for text in transcript_texts
            )
            app.exit()
            await pilot.pause()

    asyncio.run(_run())


def test_chat_app_model_command_reports_listing_failures(tmp_path: Path) -> None:
    class _FailingListingClient(MockLLMClient):
        def list_available_models(self) -> list[str]:
            raise LLMError("models.list is unavailable")

    async def _run() -> None:
        app = ChatApp(
            root_path=tmp_path,
            config=ChatConfig(),
            llm_client=_FailingListingClient(model_name="qwen"),
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ChatScreen)

            screen._submit_draft("/model")
            await pilot.pause()

            transcript_texts = _transcript_texts(app)
            assert any("Current model: qwen" in text for text in transcript_texts)
            assert any(
                "Unable to list available models: models.list is unavailable" in text
                for text in transcript_texts
            )
            app.exit()
            await pilot.pause()

    asyncio.run(_run())


def test_chat_app_model_command_switches_session_model_without_resetting_context(
    tmp_path: Path,
) -> None:
    created_models: list[str] = []

    def _fake_create_chat_llm_client(config, **kwargs):
        del config
        model_name = kwargs["model_name"]
        assert isinstance(model_name, str)
        created_models.append(model_name)
        return MockLLMClient(model_name=model_name)

    async def _run() -> None:
        app = ChatApp(
            root_path=tmp_path,
            config=ChatConfig(),
            llm_client=MockLLMClient(model_name="qwen"),
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ChatScreen)
            screen._create_chat_llm_client = _fake_create_chat_llm_client
            session_state = ChatSessionState()
            screen._session_state = session_state

            screen._submit_draft("/model qwen-coder")
            await pilot.pause()

            transcript_texts = _transcript_texts(app)
            assert any(
                "Switched model from qwen to qwen-coder." in text
                for text in transcript_texts
            )
            assert screen._active_model_name == "qwen-coder"
            assert screen._session_state is session_state
            assert isinstance(screen._llm_client, MockLLMClient)
            assert screen._llm_client.model_name == "qwen-coder"
            app.exit()
            await pilot.pause()

    asyncio.run(_run())
    assert created_models == ["qwen-coder"]


def test_chat_app_model_command_failure_keeps_previous_model(tmp_path: Path) -> None:
    def _failing_create_chat_llm_client(config, **kwargs):
        del config, kwargs
        raise LLMError("unknown model")

    async def _run() -> None:
        app = ChatApp(
            root_path=tmp_path,
            config=ChatConfig(),
            llm_client=MockLLMClient(model_name="qwen"),
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ChatScreen)
            original_client = screen._llm_client
            screen._create_chat_llm_client = _failing_create_chat_llm_client

            screen._submit_draft("/model bad-model")
            await pilot.pause()

            transcript_texts = _transcript_texts(app)
            assert any(
                "Error: Unable to switch model to bad-model: unknown model" in text
                for text in transcript_texts
            )
            assert screen._active_model_name == "qwen"
            assert screen._llm_client is original_client
            app.exit()
            await pilot.pause()

    asyncio.run(_run())


def test_chat_app_model_command_uses_mock_listing_in_mock_mode(tmp_path: Path) -> None:
    async def _run() -> None:
        app = ChatApp(
            root_path=tmp_path,
            config=ChatConfig(),
            llm_client=MockLLMClient(model_name="mock-chat"),
            mock_mode=True,
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ChatScreen)

            screen._submit_draft("/model")
            await pilot.pause()

            transcript_texts = _transcript_texts(app)
            assert any("Current model: mock-chat" in text for text in transcript_texts)
            assert any(
                "Available models:\n- mock-chat" in text for text in transcript_texts
            )
            app.exit()
            await pilot.pause()

    asyncio.run(_run())


def test_chat_app_model_command_is_rejected_while_busy(tmp_path: Path) -> None:
    async def _run() -> None:
        app = ChatApp(
            root_path=tmp_path,
            config=ChatConfig(),
            llm_client=MockLLMClient(model_name="qwen"),
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ChatScreen)
            screen._busy = True
            screen._active_runner = _FakeCancelableStream()  # type: ignore[assignment]

            screen._submit_draft("/model qwen-coder")
            await pilot.pause()

            transcript_texts = _transcript_texts(app)
            assert any(
                "Stop the active turn before changing models." in text
                for text in transcript_texts
            )
            assert screen._active_model_name == "qwen"
            assert isinstance(screen._llm_client, MockLLMClient)
            assert screen._llm_client.model_name == "qwen"
            app.exit()
            await pilot.pause()

    asyncio.run(_run())


def test_chat_app_next_turn_uses_switched_model_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def _fake_run_interactive_chat_session_turn(**kwargs):
        captured["model_name"] = kwargs["config"].llm.model_name
        captured["session_state"] = kwargs["session_state"]
        return _GatedTurnStream(
            result=_completed_result(
                user_message=kwargs["user_message"],
                answer="Done",
            )
        )

    monkeypatch.setattr(
        "engllm_chat.tools.chat.app.run_interactive_chat_session_turn",
        _fake_run_interactive_chat_session_turn,
    )

    async def _run() -> None:
        app = ChatApp(
            root_path=tmp_path,
            config=ChatConfig(),
            llm_client=MockLLMClient(model_name="qwen"),
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ChatScreen)

            screen._active_model_name = "qwen-coder"
            session_state = ChatSessionState()
            screen._session_state = session_state
            screen._submit_draft("hello")
            await pilot.pause()
            await pilot.pause(0.06)

            assert captured["model_name"] == "qwen-coder"
            assert captured["session_state"] is session_state
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
