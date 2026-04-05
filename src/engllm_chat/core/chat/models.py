"""Typed result models for deterministic chat filesystem tools."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from engllm_chat.domain.models import DomainModel

DirectoryEntryType = Literal["file", "directory", "symlink", "other"]
FileReadKind = Literal["text", "markitdown", "unsupported"]
FileInfoStatus = Literal["ok", "unsupported", "error"]
FileReadStatus = Literal["ok", "too_large", "unsupported", "error"]


class DirectoryEntry(DomainModel):
    """One filesystem entry exposed by a chat listing tool."""

    path: str
    name: str
    entry_type: DirectoryEntryType
    depth: int
    is_hidden: bool
    is_symlink: bool


class DirectoryListingResult(DomainModel):
    """Structured result for direct or recursive directory listing."""

    requested_path: str
    resolved_path: str
    recursive: bool
    max_depth_applied: int
    entries: list[DirectoryEntry] = Field(default_factory=list)
    truncated: bool = False


class FileMatch(DomainModel):
    """One matched file returned by the find-files tool."""

    path: str
    name: str
    parent_path: str
    is_hidden: bool


class FileSearchResult(DomainModel):
    """Structured result for deterministic file search."""

    requested_path: str
    resolved_path: str
    pattern: str
    matches: list[FileMatch] = Field(default_factory=list)
    truncated: bool = False


class TextSearchMatch(DomainModel):
    """One matching line returned by the text-search tool."""

    path: str
    line_number: int
    line_text: str
    is_hidden: bool


class TextSearchResult(DomainModel):
    """Structured result for deterministic text search."""

    requested_path: str
    resolved_path: str
    query: str
    matches: list[TextSearchMatch] = Field(default_factory=list)
    truncated: bool = False


class FileInfoResult(DomainModel):
    """Structured metadata for one root-confined file."""

    requested_path: str
    resolved_path: str
    name: str
    size_bytes: int
    is_hidden: bool
    is_symlink: bool
    read_kind: FileReadKind
    status: FileInfoStatus = "ok"
    estimated_token_count: int | None = None
    character_count: int | None = None
    line_count: int | None = None
    max_file_size_characters: int
    within_size_limit: bool
    full_read_char_limit: int
    can_read_full: bool
    error_message: str | None = None


class FileReadResult(DomainModel):
    """Structured content read result for one root-confined file."""

    requested_path: str
    resolved_path: str
    read_kind: FileReadKind
    status: FileReadStatus = "ok"
    content: str | None = None
    truncated: bool = False
    content_char_count: int = 0
    character_count: int | None = None
    start_char: int | None = None
    end_char: int | None = None
    file_size_bytes: int
    max_file_size_characters: int
    full_read_char_limit: int
    estimated_token_count: int | None = None
    error_message: str | None = None
