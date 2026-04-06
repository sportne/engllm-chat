"""Presentation helpers for the interactive chat UI."""

from __future__ import annotations

from rich.console import ConsoleRenderable, Group
from rich.markdown import Markdown as RichMarkdown
from rich.text import Text
from textual.widgets import Static

from engllm_chat.domain.models import ChatCitation, ChatFinalResponse


def format_citation(citation: ChatCitation) -> str:
    """Return one chat citation in transcript-friendly text form."""

    source_path = citation.source_path.as_posix()
    if citation.line_start is None:
        return source_path
    if citation.line_end is None or citation.line_end == citation.line_start:
        return f"{source_path}:{citation.line_start}"
    return f"{source_path}:{citation.line_start}-{citation.line_end}"


def format_final_response(response: ChatFinalResponse) -> str:
    """Return one final chat response in transcript-friendly text form."""

    sections = [response.answer]
    if response.citations:
        citations = "\n".join(
            f"- {format_citation(citation)}" for citation in response.citations
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


def format_final_response_metadata(response: ChatFinalResponse) -> str:
    """Return supplemental response sections as plain transcript text."""

    sections: list[str] = []
    if response.citations:
        citations = "\n".join(
            f"- {format_citation(citation)}" for citation in response.citations
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


class AssistantMarkdownEntry(Static):
    """One completed assistant response rendered with Textual markdown."""

    def __init__(self, *, markdown_text: str, metadata_text: str = "") -> None:
        self.markdown_text = markdown_text
        self.metadata_text = metadata_text
        self.label_text = "Assistant:"
        super().__init__(
            self._build_renderable(),
            classes="transcript-entry assistant transcript-entry-markdown",
        )

    @property
    def transcript_text(self) -> str:
        """Return a plain-text approximation used by tests and fallbacks."""

        parts = [f"{self.label_text}\n{self.markdown_text}".rstrip()]
        if self.metadata_text:
            parts.append(self.metadata_text)
        return "\n\n".join(parts).rstrip()

    def _build_renderable(self) -> Group:
        renderables: list[ConsoleRenderable] = [Text(self.label_text, style="bold")]
        if self.markdown_text:
            renderables.append(RichMarkdown(self.markdown_text))
        if self.metadata_text:
            renderables.append(Text(self.metadata_text))
        return Group(*renderables)


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

    @property
    def transcript_text(self) -> str:
        """Return this transcript row as plain text for export/copy flows."""

        renderable = getattr(self, "renderable", None)
        if renderable is None:
            return ""
        return str(renderable).rstrip()

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
