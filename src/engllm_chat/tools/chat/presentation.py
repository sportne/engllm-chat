"""Presentation helpers for the interactive chat UI."""

from __future__ import annotations

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
