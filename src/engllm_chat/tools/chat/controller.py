"""Controller helpers for the interactive chat screen lifecycle."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from textual.containers import VerticalScroll
from textual.widgets import Button, Static

from engllm_chat.tools.chat.models import (
    ChatWorkflowResultEvent,
    ChatWorkflowStatusEvent,
    ChatWorkflowTurnResult,
)
from engllm_chat.tools.chat.presentation import (
    TranscriptEntry,
    format_final_response,
)
from engllm_chat.tools.chat.screens import (
    ComposerTextArea,
    CredentialModal,
    InterruptConfirmModal,
)

if TYPE_CHECKING:
    from engllm_chat.tools.chat.app import ChatScreen


class ChatScreenController:
    """Encapsulate turn lifecycle and UI state transitions for ChatScreen."""

    def __init__(self, screen: ChatScreen) -> None:
        self._screen = screen

    def initialize_llm_client(self) -> bool:
        if self._screen._llm_client is not None:
            return True
        try:
            self._screen._llm_client = self._screen._create_chat_llm_client(
                self._screen._config.llm,
                provider=self._screen._config.llm.provider,
                model_name=self._screen._config.llm.model_name,
                api_base_url=self._screen._config.llm.api_base_url,
                timeout_seconds=self._screen._config.llm.timeout_seconds,
                api_key=self._screen._credential_secret,
            )
        except Exception as exc:
            self.append_transcript("error", str(exc))
            self.set_status("")
            return False
        return True

    def ensure_llm_client_ready(self) -> bool:
        if self._screen._llm_client is not None:
            return True
        metadata = (
            self._screen._credential_metadata_override
            or self._screen._config.llm.credential_prompt_metadata()
        )
        env_key_present = bool(
            metadata.api_key_env_var and os.getenv(metadata.api_key_env_var)
        )
        if (
            metadata.expects_api_key
            and metadata.prompt_for_api_key_if_missing
            and not env_key_present
            and self._screen._credential_secret is None
        ):
            self._screen.app.push_screen(
                CredentialModal(metadata),
                callback=self.handle_credential_submit,
            )
            return False
        return self.initialize_llm_client()

    def append_transcript(
        self,
        role: str,
        text: str,
        *,
        assistant_completion_state: str = "complete",
    ) -> TranscriptEntry:
        transcript = self._screen.query_one("#transcript", VerticalScroll)
        entry = TranscriptEntry(
            role=role,
            text=text,
            assistant_completion_state=assistant_completion_state,
        )
        transcript.mount(entry)
        transcript.scroll_end(animate=False)
        return entry

    def set_status(self, text: str) -> None:
        self._screen.query_one("#status-bar", Static).update(text)

    def refresh_footer(self) -> None:
        session_tokens = (
            self._screen._footer_session_tokens
            if self._screen._footer_session_tokens is not None
            else "-"
        )
        active_context_tokens = (
            self._screen._footer_active_context_tokens
            if self._screen._footer_active_context_tokens is not None
            else "-"
        )
        footer = (
            "session tokens: "
            f"{session_tokens}"
            " | active context tokens: "
            f"{active_context_tokens}"
        )
        if self._screen._footer_confidence is not None:
            footer += f" | confidence: {self._screen._footer_confidence:.2f}"
        if self._screen._busy:
            footer += " | Enter send | Shift+Enter newline | Stop active turn"
        else:
            footer += " | Enter send | Shift+Enter newline | /help | quit"
        self._screen.query_one("#footer-bar", Static).update(footer)
        self._screen.query_one("#stop-button", Button).disabled = not self._screen._busy

    def update_footer_metrics(
        self,
        *,
        session_tokens: int | None,
        active_context_tokens: int | None,
        confidence: float | None,
    ) -> None:
        self._screen._footer_session_tokens = session_tokens
        self._screen._footer_active_context_tokens = active_context_tokens
        self._screen._footer_confidence = confidence
        self.refresh_footer()

    def clear_composer(self) -> None:
        self._screen.query_one("#composer", ComposerTextArea).load_text("")

    def handle_credential_submit(self, secret_value: str | None) -> None:
        self._screen._credential_secret = secret_value or None
        self.initialize_llm_client()
        self._screen.query_one("#composer", ComposerTextArea).focus()

    def handle_interrupt_confirmation(self, confirmed: bool | None) -> None:
        if confirmed:
            pending_draft = self._screen.query_one("#composer", ComposerTextArea).text
            self._screen._pending_interrupt_draft = (
                pending_draft if pending_draft.strip() else None
            )
            self.cancel_active_turn(status_text="stopping")
        self._screen.query_one("#composer", ComposerTextArea).focus()

    def handle_turn_error(self, error_message: str) -> None:
        self.append_transcript("error", error_message)
        self.set_status("")
        self._screen._busy = False
        self._screen._active_runner = None
        self._screen._active_assistant_entry = None
        self.refresh_footer()
        self._screen.query_one("#composer", ComposerTextArea).focus()

    def handle_turn_status(self, event: object) -> None:
        typed_event = ChatWorkflowStatusEvent.model_validate(event)
        self.set_status(typed_event.status)

    def start_assistant_reveal(self, text: str) -> None:
        self._screen._reveal_generation += 1
        generation = self._screen._reveal_generation
        if self._screen._active_assistant_entry is None:
            self._screen._active_assistant_entry = self.append_transcript(
                "assistant", ""
            )
        else:
            self._screen._active_assistant_entry.update_text(
                "", assistant_completion_state="complete"
            )

        chunk_size = max(1, len(text) // 24) if text else 1

        def _step(index: int) -> None:
            if generation != self._screen._reveal_generation:
                return
            next_index = min(len(text), index + chunk_size)
            if self._screen._active_assistant_entry is not None:
                self._screen._active_assistant_entry.update_text(
                    text[:next_index],
                    assistant_completion_state="complete",
                )
            if next_index < len(text):
                self._screen.set_timer(0.01, lambda: _step(next_index))

        _step(0)

    def handle_turn_result(self, event: object) -> None:
        typed_event = ChatWorkflowResultEvent.model_validate(event)
        result = typed_event.result
        typed_result = ChatWorkflowTurnResult.model_validate(result)
        self._screen._session_state = (
            typed_result.session_state or self._screen._session_state
        )
        if typed_result.context_warning:
            self.append_transcript("system", typed_result.context_warning)
        if (
            typed_result.status == "needs_continuation"
            and typed_result.continuation_reason
        ):
            self.append_transcript("system", typed_result.continuation_reason)
        if typed_result.final_response is not None:
            self.start_assistant_reveal(
                format_final_response(typed_result.final_response)
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
                if self._screen._active_assistant_entry is None:
                    self._screen._active_assistant_entry = self.append_transcript(
                        "assistant",
                        interrupted_message.content,
                        assistant_completion_state="interrupted",
                    )
                else:
                    self._screen._active_assistant_entry.update_text(
                        interrupted_message.content,
                        assistant_completion_state="interrupted",
                    )
        self.update_footer_metrics(
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
        self.set_status("")
        self._screen._busy = False
        self._screen._active_runner = None
        pending_draft = self._screen._pending_interrupt_draft
        self._screen._pending_interrupt_draft = None
        self.refresh_footer()
        self._screen.query_one("#composer", ComposerTextArea).focus()
        if pending_draft is not None:
            self.submit_draft(pending_draft)

    def handle_inline_command(self, user_message: str) -> bool:
        normalized = user_message.strip().lower()
        if normalized == "/help":
            self.append_transcript(
                "system",
                "Ask grounded questions about the selected root. "
                "Use quit or exit to leave.",
            )
            return True
        if normalized in {"quit", "exit"}:
            self._screen.app.exit()
            return True
        return False

    def cancel_active_turn(self, *, status_text: str) -> None:
        if self._screen._active_runner is None:
            return
        self.set_status(status_text)
        self._screen._active_runner.cancel()

    def submit_draft(self, raw_draft: str) -> None:
        if not raw_draft.strip():
            return
        if self._screen._busy:
            self._screen.app.push_screen(
                InterruptConfirmModal(),
                callback=self.handle_interrupt_confirmation,
            )
            return
        if not self.ensure_llm_client_ready():
            return
        self.clear_composer()
        if self.handle_inline_command(raw_draft):
            return
        self._screen._reveal_generation += 1
        self.append_transcript("user", raw_draft)
        self._screen._active_assistant_entry = None
        self._screen._busy = True
        self.refresh_footer()
        self._screen._run_turn_worker(raw_draft)
