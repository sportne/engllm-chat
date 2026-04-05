"""Readable-content and file-metadata helpers for chat filesystem tools."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from engllm_chat.core.chat.models import FileInfoResult, FileInfoStatus, FileReadKind
from engllm_chat.core.tokenize import tokenize
from engllm_chat.domain.errors import RepositoryError
from engllm_chat.domain.models import ChatSessionConfig, ChatToolLimits

from .paths import _is_hidden

_MARKITDOWN_EXTENSIONS = {
    ".doc",
    ".docx",
    ".epub",
    ".html",
    ".pdf",
    ".ppt",
    ".pptx",
    ".rtf",
    ".xls",
    ".xlsx",
}


@dataclass(frozen=True, slots=True)
class _LoadedReadableContent:
    """Deterministic text representation for one readable file."""

    read_kind: FileReadKind
    status: FileInfoStatus
    content: str | None
    error_message: str | None = None


def _read_searchable_text(path: Path) -> str | None:
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None

    if "\x00" in content:
        return None
    return content


def load_readable_content(
    path: Path,
    *,
    markitdown_cache_path: Callable[[Path], Path],
    convert_with_markitdown: Callable[[Path], str],
) -> _LoadedReadableContent:
    text_content = _read_searchable_text(path)
    if text_content is not None:
        return _LoadedReadableContent(
            read_kind="text",
            status="ok",
            content=text_content,
        )

    if path.suffix.lower() not in _MARKITDOWN_EXTENSIONS:
        return _LoadedReadableContent(
            read_kind="unsupported",
            status="unsupported",
            content=None,
            error_message="File type is not supported for chat reads",
        )

    cache_path = markitdown_cache_path(path)
    if cache_path.exists():
        return _LoadedReadableContent(
            read_kind="markitdown",
            status="ok",
            content=cache_path.read_text(encoding="utf-8"),
        )

    try:
        converted = convert_with_markitdown(path)
    except Exception as exc:
        return _LoadedReadableContent(
            read_kind="markitdown",
            status="error",
            content=None,
            error_message=str(exc),
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(converted, encoding="utf-8")
    return _LoadedReadableContent(
        read_kind="markitdown",
        status="ok",
        content=converted,
    )


def _count_lines(text: str, *, max_read_lines: int) -> int | None:
    if not text:
        return 0

    line_count = text.count("\n")
    if not text.endswith("\n"):
        line_count += 1
    if line_count > max_read_lines:
        return None
    return line_count


def _estimate_token_count(text: str) -> int:
    return len(tokenize(text))


def _is_within_character_limit(
    content: str | None,
    *,
    tool_limits: ChatToolLimits,
) -> bool:
    return content is not None and len(content) <= tool_limits.max_file_size_characters


def _effective_full_read_char_limit(
    session_config: ChatSessionConfig,
    tool_limits: ChatToolLimits,
) -> int:
    configured_limit = tool_limits.max_read_file_chars
    if configured_limit is not None:
        return configured_limit
    return max(1, session_config.max_context_tokens * 4)


def _normalize_range(
    *,
    start_char: int | None,
    end_char: int | None,
    character_count: int,
) -> tuple[int, int]:
    normalized_start = 0 if start_char is None else start_char
    normalized_end = character_count if end_char is None else end_char

    if normalized_start < 0:
        raise RepositoryError("start_char must be greater than or equal to 0")
    if normalized_end < 0:
        raise RepositoryError("end_char must be greater than or equal to 0")
    if end_char is not None and normalized_end <= normalized_start:
        raise RepositoryError("end_char must be greater than start_char")
    if normalized_start > character_count:
        raise RepositoryError("start_char must not exceed character_count")

    return normalized_start, min(normalized_end, character_count)


def _build_file_info_result(
    *,
    requested_path: str,
    resolved_path: str,
    candidate_file: Path,
    resolved_file: Path,
    relative_candidate_path: Path,
    session_config: ChatSessionConfig,
    tool_limits: ChatToolLimits,
    loaded_content: _LoadedReadableContent,
) -> FileInfoResult:
    full_read_char_limit = _effective_full_read_char_limit(session_config, tool_limits)
    size_bytes = resolved_file.stat().st_size
    content = loaded_content.content
    character_count = len(content) if content is not None else None
    within_size_limit = _is_within_character_limit(content, tool_limits=tool_limits)

    return FileInfoResult(
        requested_path=requested_path,
        resolved_path=resolved_path,
        name=candidate_file.name,
        size_bytes=size_bytes,
        is_hidden=_is_hidden(relative_candidate_path),
        is_symlink=candidate_file.is_symlink(),
        read_kind=loaded_content.read_kind,
        status=loaded_content.status,
        estimated_token_count=(
            _estimate_token_count(content) if content is not None else None
        ),
        character_count=character_count,
        line_count=(
            _count_lines(content, max_read_lines=tool_limits.max_read_lines)
            if content is not None
            else None
        ),
        max_file_size_characters=tool_limits.max_file_size_characters,
        within_size_limit=within_size_limit,
        full_read_char_limit=full_read_char_limit,
        can_read_full=within_size_limit
        and character_count is not None
        and character_count <= full_read_char_limit,
        error_message=loaded_content.error_message,
    )
