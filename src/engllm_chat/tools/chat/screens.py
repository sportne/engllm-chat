"""Textual modal and composer widgets for the interactive chat UI."""

from __future__ import annotations

from textual import events, on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static, TextArea

from engllm_chat.domain.models import ChatCredentialPromptMetadata


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
