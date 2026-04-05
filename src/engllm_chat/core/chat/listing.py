"""Deterministic read-only directory listing tools for chat."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

from engllm_chat.core.chat._listing.content import (
    _LoadedReadableContent,
    load_readable_content,
)
from engllm_chat.core.chat._listing.ops import (
    find_files_impl,
    get_file_info_impl,
    list_directory_impl,
    list_directory_recursive_impl,
    read_file_impl,
    search_text_impl,
)
from engllm_chat.core.chat.models import (
    DirectoryListingResult,
    FileInfoBatchResult,
    FileInfoResult,
    FileReadResult,
    FileSearchResult,
    TextSearchResult,
)
from engllm_chat.domain.models import (
    ChatSessionConfig,
    ChatSourceFilters,
    ChatToolLimits,
)


def _markitdown_cache_root() -> Path:
    cache_root = Path(tempfile.gettempdir()) / "engllm-chat-markitdown-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root


def _markitdown_cache_path(path: Path) -> Path:
    stat = path.stat()
    cache_key = hashlib.sha256(
        f"{path.resolve().as_posix()}:{stat.st_size}:{stat.st_mtime_ns}".encode()
    ).hexdigest()
    return _markitdown_cache_root() / f"{cache_key}.md"


def _convert_with_markitdown(path: Path) -> str:
    try:
        from markitdown import MarkItDown
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "markitdown dependencies are unavailable for non-text file conversion"
        ) from exc

    result = MarkItDown().convert(str(path))
    if isinstance(result, str):
        return result

    for attribute in ("text_content", "markdown", "text"):
        value = getattr(result, attribute, None)
        if isinstance(value, str):
            return value

    raise RuntimeError("markitdown conversion did not return readable markdown text")


def _load_readable_content(path: Path) -> _LoadedReadableContent:
    return load_readable_content(
        path,
        markitdown_cache_path=_markitdown_cache_path,
        convert_with_markitdown=_convert_with_markitdown,
    )


def list_directory(
    root_path: Path,
    path: str = ".",
    *,
    source_filters: ChatSourceFilters,
    tool_limits: ChatToolLimits,
) -> DirectoryListingResult:
    """Return the immediate matching children of one directory under the root."""

    return list_directory_impl(
        root_path,
        path,
        source_filters=source_filters,
        tool_limits=tool_limits,
    )


def list_directory_recursive(
    root_path: Path,
    path: str = ".",
    *,
    source_filters: ChatSourceFilters,
    tool_limits: ChatToolLimits,
    max_depth: int | None = None,
) -> DirectoryListingResult:
    """Return a flat deterministic recursive listing under one directory."""

    return list_directory_recursive_impl(
        root_path,
        path,
        source_filters=source_filters,
        tool_limits=tool_limits,
        max_depth=max_depth,
    )


def find_files(
    root_path: Path,
    pattern: str,
    path: str = ".",
    *,
    source_filters: ChatSourceFilters,
    tool_limits: ChatToolLimits,
) -> FileSearchResult:
    """Return matching files beneath one root-confined directory subtree."""

    return find_files_impl(
        root_path,
        pattern,
        path,
        source_filters=source_filters,
        tool_limits=tool_limits,
    )


def search_text(
    root_path: Path,
    query: str,
    path: str = ".",
    *,
    source_filters: ChatSourceFilters,
    tool_limits: ChatToolLimits,
) -> TextSearchResult:
    """Return matching text lines within one root-confined directory or file."""

    return search_text_impl(
        root_path,
        query,
        path,
        source_filters=source_filters,
        tool_limits=tool_limits,
        load_readable_content=_load_readable_content,
    )


def get_file_info(
    root_path: Path,
    path: str | list[str],
    *,
    session_config: ChatSessionConfig,
    tool_limits: ChatToolLimits,
) -> FileInfoResult | FileInfoBatchResult:
    """Return deterministic metadata for one or more root-confined files."""

    return get_file_info_impl(
        root_path,
        path,
        session_config=session_config,
        tool_limits=tool_limits,
        load_readable_content=_load_readable_content,
    )


def read_file(
    root_path: Path,
    path: str,
    *,
    session_config: ChatSessionConfig,
    tool_limits: ChatToolLimits,
    start_char: int | None = None,
    end_char: int | None = None,
) -> FileReadResult:
    """Return bounded text or markdown content for one root-confined file."""

    return read_file_impl(
        root_path,
        path,
        session_config=session_config,
        tool_limits=tool_limits,
        start_char=start_char,
        end_char=end_char,
        load_readable_content=_load_readable_content,
    )
