"""Textual chat app shell for interactive directory chat."""

from __future__ import annotations

import os
from pathlib import Path

from textual import events, on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen, Screen
from textual.widgets import Button, Input, Static, TextArea

from engllm_chat.domain.models import (
    ChatCitation,
    ChatConfig,
    ChatCredentialPromptMetadata,
    ChatFinalResponse,
)
from engllm_chat.llm.base import ChatLLMClient
from engllm_chat.llm.factory import create_chat_llm_client
from engllm_chat.tools.chat.models import (
    ChatSessionState,
    ChatWorkflowAssistantDeltaEvent,
    ChatWorkflowResultEvent,
    ChatWorkflowStatusEvent,
    ChatWorkflowTurnResult,
)
from engllm_chat.tools.chat.workflow import (
    ChatSessionTurnStream,
    run_streaming_chat_session_turn,
)


def _format_citation(citation: ChatCitation) -> str:
    source_path = citation.source_path.as_posix()
    if citation.line_start is None:
        return source_path
    if citation.line_end is None or citation.line_end == citation.line_start:
        return f"{source_path}:{citation.line_start}"
    return f"{source_path}:{citation.line_start}-{citation.line_end}"


def _format_final_response(response: ChatFinalResponse) -> str:
    sections = [response.answer]
    if response.citations:
        citations = "\n".join(
            f"- {_format_citation(citation)}" for citation in response.citations
        )
        sections.append(f"Citations:\n{citations}")
    if response.uncertainty:
        uncertainty = "\n".join(f"- {item}" for item in response.uncertainty)
        sections.append(f"Uncertainty:\n{uncertainty}")
    if response.missing_information:
        missing_information = "\n".join(
            f"- {item}" for item in response.missing_information
        )
        sections.append(f"Missing Information:\n{missing_information}")
    if response.follow_up_suggestions:
        follow_ups = "\n".join(f"- {item}" for item in response.follow_up_suggestions)
        sections.append(f"Follow-up Suggestions:\n{follow_ups}")
    return "\n\n".join(sections)


class TranscriptEntry(Static):
    """One durable transcript row."""

    def __init__(
        self,
        *,
        role: str,
        text: str,
        assistant_completion_state: str = "complete",
    ) -> None:
        self.role = role
        self._assistant_completion_state = assistant_completion_state
        super().__init__(
            self._format_text(role, text, assistant_completion_state),
            classes=f"transcript-entry {role}",
        )

    def update_text(
        self, text: str, *, assistant_completion_state: str | None = None
    ) -> None:
        if assistant_completion_state is not None:
            self._assistant_completion_state = assistant_completion_state
        self.update(
            self._format_text(
                self.role,
                text,
                self._assistant_completion_state,
            )
        )

    @staticmethod
    def _format_text(
        role: str,
        text: str,
        assistant_completion_state: str,
    ) -> str:
        if role == "assistant":
            if assistant_completion_state == "interrupted":
                return f"Assistant (interrupted):\n{text}".rstrip()
            return f"Assistant:\n{text}"
        if role == "user":
            return f"You:\n{text}"
        if role == "error":
            return f"Error: {text}"
        return f"System:\n{text}"


class CredentialModal(ModalScreen[str | None]):
    """Startup modal for optional session-only credential entry."""

    def __init__(self, metadata: ChatCredentialPromptMetadata) -> None:
        super().__init__(id="credential-modal")
        self._metadata = metadata

    def compose(self) -> ComposeResult:
        prompt_label = self._metadata.api_key_env_var or "API key"
        with Vertical(id="credential-modal-body"):
            yield Static(
                f"Enter {prompt_label} for this session only, or leave it empty.",
                id="credential-copy",
            )
            yield Input(
                placeholder=prompt_label,
                password=self._metadata.mask_input,
                id="credential-input",
            )
            with Horizontal(id="credential-actions"):
                yield Button("Continue", id="credential-submit", variant="primary")
                yield Button("Cancel", id="credential-cancel")

    @on(Button.Pressed, "#credential-submit")
    def handle_submit(self) -> None:
        value = self.query_one("#credential-input", Input).value
        self.dismiss(value)

    @on(Button.Pressed, "#credential-cancel")
    def handle_cancel(self) -> None:
        self.dismiss(None)


class InterruptConfirmModal(ModalScreen[bool]):
    """Confirmation modal shown when the user sends while busy."""

    def compose(self) -> ComposeResult:
        with Vertical(id="interrupt-modal-body"):
            yield Static(
                "A chat turn is already running. Interrupt it and send "
                "the current draft now?",
                id="interrupt-copy",
            )
            with Horizontal(id="interrupt-actions"):
                yield Button("Interrupt", id="interrupt-confirm", variant="warning")
                yield Button("Cancel", id="interrupt-cancel")

    @on(Button.Pressed, "#interrupt-confirm")
    def handle_confirm(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#interrupt-cancel")
    def handle_cancel(self) -> None:
        self.dismiss(False)


class ComposerTextArea(TextArea):
    """Chat composer with explicit Enter/Shift+Enter behavior."""

    class SubmitRequested(Message):
        """Posted when the composer should submit the current draft."""

        def __init__(self, composer: ComposerTextArea) -> None:
            super().__init__()
            self._composer = composer

        @property
        def control(self) -> ComposerTextArea:
            return self._composer

    def on_key(self, event: events.Key) -> None:
        if event.key == "shift+enter":
            event.stop()
            event.prevent_default()
            self.insert("\n")
            return
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self.post_message(self.SubmitRequested(self))


class ChatScreen(Screen[None]):
    """Main chat screen with transcript, composer, status, and footer rows."""

    def __init__(
        self,
        *,
        root_path: Path,
        config: ChatConfig,
        llm_client: ChatLLMClient | None,
        credential_metadata_override: ChatCredentialPromptMetadata | None = None,
    ) -> None:
        super().__init__(id="chat-screen")
        self._root_path = root_path
        self._config = config
        self._llm_client = llm_client
        self._credential_metadata_override = credential_metadata_override
        self._session_state = ChatSessionState()
        self._credential_secret: str | None = None
        self._busy = False
        self._active_assistant_entry: TranscriptEntry | None = None
        self._active_stream: ChatSessionTurnStream | None = None
        self._pending_interrupt_draft: str | None = None
        self._footer_session_tokens: int | None = None
        self._footer_active_context_tokens: int | None = None
        self._footer_confidence: float | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="chat-layout"):
            yield VerticalScroll(id="transcript")
            with Horizontal(id="composer-row"):
                yield ComposerTextArea("", id="composer")
                yield Button("Send", id="send-button", variant="primary")
                yield Button("Stop", id="stop-button", variant="warning", disabled=True)
            yield Static("", id="status-bar")
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
            or self._config.llm.credential_prompt_metadata()
        )
        env_key_present = bool(
            metadata.api_key_env_var and os.getenv(metadata.api_key_env_var)
        )
        if (
            self._llm_client is None
            and metadata.expects_api_key
            and metadata.prompt_for_api_key_if_missing
            and not env_key_present
        ):
            self.app.push_screen(
                CredentialModal(metadata),
                callback=self._handle_credential_submit,
            )
        elif self._llm_client is None:
            self._initialize_llm_client()
        self._refresh_footer()
        self.query_one("#composer", ComposerTextArea).focus()

    def _initialize_llm_client(self) -> bool:
        if self._llm_client is not None:
            return True
        try:
            self._llm_client = create_chat_llm_client(
                self._config.llm,
                provider=self._config.llm.provider,
                model_name=self._config.llm.model_name,
                ollama_base_url=self._config.llm.ollama_base_url,
                api_base_url=self._config.llm.api_base_url,
                timeout_seconds=self._config.llm.timeout_seconds,
                api_key=self._credential_secret,
            )
        except Exception as exc:
            self._append_transcript("error", str(exc))
            self._set_status("")
            return False
        return True

    def _ensure_llm_client_ready(self) -> bool:
        if self._llm_client is not None:
            return True
        metadata = (
            self._credential_metadata_override
            or self._config.llm.credential_prompt_metadata()
        )
        env_key_present = bool(
            metadata.api_key_env_var and os.getenv(metadata.api_key_env_var)
        )
        if (
            metadata.expects_api_key
            and metadata.prompt_for_api_key_if_missing
            and not env_key_present
            and self._credential_secret is None
        ):
            self.app.push_screen(
                CredentialModal(metadata),
                callback=self._handle_credential_submit,
            )
            return False
        return self._initialize_llm_client()

    def _append_transcript(
        self,
        role: str,
        text: str,
        *,
        assistant_completion_state: str = "complete",
    ) -> TranscriptEntry:
        transcript = self.query_one("#transcript", VerticalScroll)
        entry = TranscriptEntry(
            role=role,
            text=text,
            assistant_completion_state=assistant_completion_state,
        )
        transcript.mount(entry)
        transcript.scroll_end(animate=False)
        return entry

    def _set_status(self, text: str) -> None:
        self.query_one("#status-bar", Static).update(text)

    def _refresh_footer(self) -> None:
        session_tokens = (
            self._footer_session_tokens
            if self._footer_session_tokens is not None
            else "-"
        )
        active_context_tokens = (
            self._footer_active_context_tokens
            if self._footer_active_context_tokens is not None
            else "-"
        )
        footer = (
            "session tokens: "
            f"{session_tokens}"
            " | active context tokens: "
            f"{active_context_tokens}"
        )
        if self._footer_confidence is not None:
            footer += f" | confidence: {self._footer_confidence:.2f}"
        if self._busy:
            footer += " | Enter send | Shift+Enter newline | Stop active turn"
        else:
            footer += " | Enter send | Shift+Enter newline | /help | quit"
        self.query_one("#footer-bar", Static).update(footer)
        self.query_one("#stop-button", Button).disabled = not self._busy

    def _update_footer_metrics(
        self,
        *,
        session_tokens: int | None,
        active_context_tokens: int | None,
        confidence: float | None,
    ) -> None:
        self._footer_session_tokens = session_tokens
        self._footer_active_context_tokens = active_context_tokens
        self._footer_confidence = confidence
        self._refresh_footer()

    def _clear_composer(self) -> None:
        self.query_one("#composer", ComposerTextArea).load_text("")

    def _handle_credential_submit(self, secret_value: str | None) -> None:
        self._credential_secret = secret_value or None
        self._initialize_llm_client()
        self.query_one("#composer", ComposerTextArea).focus()

    def _handle_interrupt_confirmation(self, confirmed: bool | None) -> None:
        if confirmed:
            pending_draft = self.query_one("#composer", ComposerTextArea).text
            self._pending_interrupt_draft = (
                pending_draft if pending_draft.strip() else None
            )
            self._cancel_active_turn(status_text="stopping")
        self.query_one("#composer", ComposerTextArea).focus()

    def _handle_turn_error(self, error_message: str) -> None:
        self._append_transcript("error", error_message)
        self._set_status("")
        self._busy = False
        self._active_stream = None
        self._active_assistant_entry = None
        self._refresh_footer()
        self.query_one("#composer", ComposerTextArea).focus()

    def _handle_turn_status(self, event: object) -> None:
        typed_event = ChatWorkflowStatusEvent.model_validate(event)
        self._set_status(typed_event.status)

    def _handle_turn_delta(self, event: object) -> None:
        typed_event = ChatWorkflowAssistantDeltaEvent.model_validate(event)
        if self._active_assistant_entry is None:
            self._active_assistant_entry = self._append_transcript(
                "assistant",
                typed_event.accumulated_text,
            )
        else:
            self._active_assistant_entry.update_text(typed_event.accumulated_text)

    def _handle_turn_result(self, event: object) -> None:
        typed_event = ChatWorkflowResultEvent.model_validate(event)
        result = typed_event.result
        typed_result = ChatWorkflowTurnResult.model_validate(result)
        self._session_state = typed_result.session_state or self._session_state
        if typed_result.context_warning:
            self._append_transcript("system", typed_result.context_warning)
        if (
            typed_result.status == "needs_continuation"
            and typed_result.continuation_reason
        ):
            self._append_transcript("system", typed_result.continuation_reason)
        if typed_result.final_response is not None:
            if self._active_assistant_entry is None:
                self._active_assistant_entry = self._append_transcript(
                    "assistant",
                    _format_final_response(typed_result.final_response),
                )
            else:
                self._active_assistant_entry.update_text(
                    _format_final_response(typed_result.final_response),
                    assistant_completion_state="complete",
                )
        elif typed_result.status == "interrupted":
            interrupted_message = next(
                (
                    message
                    for message in reversed(typed_result.new_messages)
                    if message.role == "assistant"
                    and message.completion_state == "interrupted"
                ),
                None,
            )
            if interrupted_message is not None and interrupted_message.content:
                if self._active_assistant_entry is None:
                    self._active_assistant_entry = self._append_transcript(
                        "assistant",
                        interrupted_message.content,
                        assistant_completion_state="interrupted",
                    )
                else:
                    self._active_assistant_entry.update_text(
                        interrupted_message.content,
                        assistant_completion_state="interrupted",
                    )
        self._update_footer_metrics(
            session_tokens=(
                typed_result.token_usage.session_tokens
                if typed_result.token_usage
                else None
            ),
            active_context_tokens=(
                typed_result.token_usage.active_context_tokens
                if typed_result.token_usage
                else None
            ),
            confidence=(
                typed_result.final_response.confidence
                if typed_result.final_response
                else None
            ),
        )
        self._set_status("")
        self._busy = False
        self._active_stream = None
        if typed_result.status != "interrupted":
            self._active_assistant_entry = None
        pending_draft = self._pending_interrupt_draft
        self._pending_interrupt_draft = None
        self._refresh_footer()
        self.query_one("#composer", ComposerTextArea).focus()
        if pending_draft is not None:
            self._submit_draft(pending_draft)

    def _handle_inline_command(self, user_message: str) -> bool:
        normalized = user_message.strip().lower()
        if normalized == "/help":
            self._append_transcript(
                "system",
                "Ask grounded questions about the selected root. "
                "Use quit or exit to leave.",
            )
            return True
        if normalized in {"quit", "exit"}:
            self.app.exit()
            return True
        return False

    def _cancel_active_turn(self, *, status_text: str) -> None:
        if self._active_stream is None:
            return
        self._set_status(status_text)
        self._active_stream.cancel()

    def _submit_draft(self, raw_draft: str) -> None:
        if not raw_draft.strip():
            return
        if self._busy:
            self.app.push_screen(
                InterruptConfirmModal(),
                callback=self._handle_interrupt_confirmation,
            )
            return
        if not self._ensure_llm_client_ready():
            return
        self._clear_composer()
        if self._handle_inline_command(raw_draft):
            return
        self._append_transcript("user", raw_draft)
        self._active_assistant_entry = None
        self._busy = True
        self._refresh_footer()
        self._run_turn_worker(raw_draft)

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
            turn_stream = run_streaming_chat_session_turn(
                user_message=user_message,
                session_state=self._session_state,
                root_path=self._root_path,
                config=self._config,
                llm_client=llm_client,
            )
            self._active_stream = turn_stream
            for event in turn_stream:
                if isinstance(event, ChatWorkflowStatusEvent):
                    self.app.call_from_thread(
                        self._handle_turn_status,
                        event.model_dump(mode="json"),
                    )
                    continue
                if isinstance(event, ChatWorkflowAssistantDeltaEvent):
                    self.app.call_from_thread(
                        self._handle_turn_delta,
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
            self._active_stream = None


class ChatApp(App[None]):
    """First Textual shell for interactive directory chat."""

    def __init__(
        self,
        *,
        root_path: Path,
        config: ChatConfig,
        llm_client: ChatLLMClient | None,
        credential_metadata_override: ChatCredentialPromptMetadata | None = None,
    ) -> None:
        super().__init__()
        self._root_path = root_path
        self._config = config
        self._llm_client = llm_client
        self._credential_metadata_override = credential_metadata_override

    def on_mount(self) -> None:
        self.push_screen(
            ChatScreen(
                root_path=self._root_path,
                config=self._config,
                llm_client=self._llm_client,
                credential_metadata_override=self._credential_metadata_override,
            )
        )


def run_chat_app(
    *,
    root_path: Path,
    config: ChatConfig,
    llm_client: ChatLLMClient | None = None,
) -> int:
    """Launch the first Textual chat app shell."""

    ChatApp(root_path=root_path, config=config, llm_client=llm_client).run()
    return 0
