"""Textual chat app shell for interactive directory chat."""

from __future__ import annotations

from pathlib import Path

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Static

from engllm_chat.domain.models import (
    ChatConfig,
    ChatCredentialPromptMetadata,
)
from engllm_chat.llm.base import ChatLLMClient
from engllm_chat.llm.factory import create_chat_llm_client
from engllm_chat.tools.chat.controller import ChatScreenController
from engllm_chat.tools.chat.models import (
    ChatSessionState,
    ChatWorkflowStatusEvent,
)
from engllm_chat.tools.chat.presentation import (
    AssistantMarkdownEntry,
    TranscriptEntry,
)
from engllm_chat.tools.chat.screens import (
    ComposerTextArea,
    CredentialModal,
    InterruptConfirmModal,
    TranscriptCopyModal,
)
from engllm_chat.tools.chat.workflow import (
    ChatSessionTurnRunner,
    run_interactive_chat_session_turn,
)

__all__ = [
    "ChatApp",
    "ChatScreen",
    "AssistantMarkdownEntry",
    "ComposerTextArea",
    "CredentialModal",
    "InterruptConfirmModal",
    "TranscriptCopyModal",
    "TranscriptEntry",
    "run_chat_app",
]


class ChatScreen(Screen[None]):
    """Main chat screen with transcript, composer, status, and footer rows."""

    BINDINGS = [
        Binding(
            "f6",
            "open_transcript_copy",
            "Copy transcript",
            show=False,
            priority=True,
        ),
    ]

    DEFAULT_CSS = """
    ChatScreen {
        background: #111317;
        color: #e6e6e6;
    }

    #chat-layout {
        layout: vertical;
    }

    #transcript {
        height: 1fr;
        padding: 1 1 0 1;
    }

    .transcript-entry {
        width: 100%;
        margin: 0 0 1 0;
        padding: 0 1;
        background: #17191d;
    }

    .transcript-entry.user {
        background: #1a2028;
        border-left: tall #4f6f96;
    }

    .transcript-entry.assistant {
        background: #171b1f;
        border-left: tall #4f8a7d;
    }

    .transcript-entry.system {
        background: #16161a;
        color: #b8bec8;
        border-left: tall #5b5f66;
    }

    .transcript-entry.error {
        background: #261718;
        color: #f3d8d9;
        border-left: tall #b76363;
    }

    .transcript-entry-label {
        width: 100%;
        padding: 0 1;
        color: #f2f5f8;
        text-style: bold;
        background: transparent;
    }

    .assistant-markdown-body {
        width: 100%;
        padding: 0 1 1 1;
        color: #e6e6e6;
        background: transparent;
        overflow-y: hidden;
    }

    #status-bar {
        height: 1;
        min-height: 1;
        margin: 0 1;
        padding: 0 1;
        color: #d4e6ff;
        background: #1a2330;
    }

    #composer-row {
        height: 11;
        margin: 0 1;
        layout: horizontal;
    }

    #composer {
        width: 1fr;
    }

    #composer-actions {
        width: 11;
        height: auto;
        margin-left: 1;
        layout: vertical;
    }

    #composer-actions Button {
        width: 100%;
        height: 3;
        min-height: 3;
        margin-bottom: 1;
    }

    #stop-button {
        margin-bottom: 0;
    }

    #footer-bar {
        height: 1;
        min-height: 1;
        margin: 0 1 1 1;
        padding: 0 1;
        color: #c7c7c7;
    }
    """

    def __init__(
        self,
        *,
        root_path: Path,
        config: ChatConfig,
        llm_client: ChatLLMClient | None,
        mock_mode: bool = False,
        credential_metadata_override: ChatCredentialPromptMetadata | None = None,
    ) -> None:
        super().__init__(id="chat-screen")
        self._root_path = root_path
        self._config = config
        self._llm_client = llm_client
        self._use_mock = mock_mode
        self._credential_metadata_override = credential_metadata_override
        self._create_chat_llm_client = create_chat_llm_client
        self._session_state = ChatSessionState()
        self._credential_secret: str | None = None
        self._busy = False
        self._active_assistant_entry: TranscriptEntry | None = None
        self._active_runner: ChatSessionTurnRunner | None = None
        self._pending_interrupt_draft: str | None = None
        self._footer_session_tokens: int | None = None
        self._footer_active_context_tokens: int | None = None
        self._footer_confidence: float | None = None
        self._credential_prompt_completed = False
        self._controller = ChatScreenController(self)

    def compose(self) -> ComposeResult:
        with Vertical(id="chat-layout"):
            yield VerticalScroll(id="transcript")
            yield Static("", id="status-bar")
            with Horizontal(id="composer-row"):
                yield ComposerTextArea("", id="composer")
                with Vertical(id="composer-actions"):
                    yield Button("Send", id="send-button", variant="primary")
                    yield Button(
                        "Stop",
                        id="stop-button",
                        variant="warning",
                        disabled=True,
                    )
            yield Static("", id="footer-bar")

    def on_mount(self) -> None:
        self._append_transcript(
            "system",
            (
                f"Root: {self._root_path}\n"
                "Use /help for guidance. Type quit or exit to leave."
            ),
        )
        metadata = (
            self._credential_metadata_override
            or self._config.llm.credential_prompt_metadata(mock_mode=self._use_mock)
        )
        if self._llm_client is None:
            self.app.push_screen(
                CredentialModal(metadata),
                callback=self._controller.handle_credential_submit,
            )
        self._controller.refresh_footer()
        self.query_one("#composer", ComposerTextArea).focus()

    def _initialize_llm_client(self) -> bool:
        return self._controller.initialize_llm_client()

    def _ensure_llm_client_ready(self) -> bool:
        return self._controller.ensure_llm_client_ready()

    def _append_transcript(
        self,
        role: str,
        text: str,
        *,
        assistant_completion_state: str = "complete",
    ) -> TranscriptEntry:
        return self._controller.append_transcript(
            role,
            text,
            assistant_completion_state=assistant_completion_state,
        )

    def _set_status(self, text: str) -> None:
        self._controller.set_status(text)

    def _refresh_footer(self) -> None:
        self._controller.refresh_footer()

    def _update_footer_metrics(
        self,
        *,
        session_tokens: int | None,
        active_context_tokens: int | None,
        confidence: float | None,
    ) -> None:
        self._controller.update_footer_metrics(
            session_tokens=session_tokens,
            active_context_tokens=active_context_tokens,
            confidence=confidence,
        )

    def _clear_composer(self) -> None:
        self._controller.clear_composer()

    def _handle_credential_submit(self, secret_value: str | None) -> None:
        self._controller.handle_credential_submit(secret_value)

    def _handle_interrupt_confirmation(self, confirmed: bool | None) -> None:
        self._controller.handle_interrupt_confirmation(confirmed)

    def _handle_turn_error(self, error_message: str) -> None:
        self._controller.handle_turn_error(error_message)

    def _handle_turn_status(self, event: object) -> None:
        self._controller.handle_turn_status(event)

    def _handle_turn_result(self, event: object) -> None:
        self._controller.handle_turn_result(event)

    def _handle_inline_command(self, user_message: str) -> bool:
        return self._controller.handle_inline_command(user_message)

    def action_open_transcript_copy(self) -> None:
        self._controller.open_transcript_copy()

    def _cancel_active_turn(self, *, status_text: str) -> None:
        self._controller.cancel_active_turn(status_text=status_text)

    def _submit_draft(self, raw_draft: str) -> None:
        self._controller.submit_draft(raw_draft)

    @on(ComposerTextArea.SubmitRequested, "#composer")
    def handle_composer_submit(self) -> None:
        composer = self.query_one("#composer", ComposerTextArea)
        self._submit_draft(composer.text)

    @on(Button.Pressed, "#send-button")
    def handle_send_button(self) -> None:
        composer = self.query_one("#composer", ComposerTextArea)
        self._submit_draft(composer.text)

    @on(Button.Pressed, "#stop-button")
    def handle_stop_button(self) -> None:
        if not self._busy:
            return
        self._pending_interrupt_draft = None
        self._cancel_active_turn(status_text="stopping")

    @work(thread=True, exclusive=True)
    def _run_turn_worker(self, user_message: str) -> None:
        try:
            llm_client = self._llm_client
            if llm_client is None:
                raise RuntimeError("Chat client is not configured.")
            turn_stream = run_interactive_chat_session_turn(
                user_message=user_message,
                session_state=self._session_state,
                root_path=self._root_path,
                config=self._config,
                llm_client=llm_client,
            )
            self._active_runner = turn_stream
            for event in turn_stream:
                if isinstance(event, ChatWorkflowStatusEvent):
                    self.app.call_from_thread(
                        self._handle_turn_status,
                        event.model_dump(mode="json"),
                    )
                    continue
                self.app.call_from_thread(
                    self._handle_turn_result,
                    event.model_dump(mode="json"),
                )
        except Exception as exc:
            self.app.call_from_thread(self._handle_turn_error, str(exc))
        finally:
            self._active_runner = None


class ChatApp(App[None]):
    """First Textual shell for interactive directory chat."""

    def __init__(
        self,
        *,
        root_path: Path,
        config: ChatConfig,
        llm_client: ChatLLMClient | None,
        mock_mode: bool = False,
        credential_metadata_override: ChatCredentialPromptMetadata | None = None,
    ) -> None:
        super().__init__()
        self._root_path = root_path
        self._config = config
        self._llm_client = llm_client
        self._mock_mode = mock_mode
        self._credential_metadata_override = credential_metadata_override

    def on_mount(self) -> None:
        self.push_screen(
            ChatScreen(
                root_path=self._root_path,
                config=self._config,
                llm_client=self._llm_client,
                mock_mode=self._mock_mode,
                credential_metadata_override=self._credential_metadata_override,
            )
        )


def run_chat_app(
    *,
    root_path: Path,
    config: ChatConfig,
    mock_mode: bool = False,
    llm_client: ChatLLMClient | None = None,
) -> int:
    """Launch the first Textual chat app shell."""

    ChatApp(
        root_path=root_path,
        config=config,
        llm_client=llm_client,
        mock_mode=mock_mode,
    ).run()
    return 0
