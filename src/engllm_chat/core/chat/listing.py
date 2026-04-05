"""Deterministic read-only directory listing tools for chat."""

from __future__ import annotations

import fnmatch
import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from engllm_chat.core.chat.models import (
    DirectoryEntry,
    DirectoryEntryType,
    DirectoryListingResult,
    FileInfoBatchResult,
    FileInfoResult,
    FileInfoStatus,
    FileMatch,
    FileReadKind,
    FileReadResult,
    FileReadStatus,
    FileSearchResult,
    TextSearchMatch,
    TextSearchResult,
)
from engllm_chat.core.tokenize import tokenize
from engllm_chat.domain.errors import RepositoryError
from engllm_chat.domain.models import (
    ChatSessionConfig,
    ChatSourceFilters,
    ChatToolLimits,
)

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
class _ResolvedRootPath:
    """Normalized and root-confined request path metadata."""

    root: Path
    candidate: Path
    resolved: Path
    requested_path: str
    resolved_path: str


@dataclass(frozen=True, slots=True)
class _LoadedReadableContent:
    """Deterministic text representation for one readable file."""

    read_kind: FileReadKind
    status: FileInfoStatus
    content: str | None
    error_message: str | None = None


def _matches_patterns(path: Path, patterns: list[str]) -> bool:
    path_match = PurePosixPath(path.as_posix())
    return any(path_match.match(pattern) for pattern in patterns)


def _normalize_requested_path(path: str) -> str:
    cleaned = path.strip()
    if not cleaned:
        raise RepositoryError("Requested chat tool path must not be empty")
    if cleaned == ".":
        return "."
    return PurePosixPath(cleaned).as_posix()


def _normalize_required_value(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise RepositoryError(f"{field_name} must not be empty")
    return cleaned


def _normalize_required_pattern(pattern: str) -> str:
    return PurePosixPath(
        _normalize_required_value(pattern, field_name="find_files pattern")
    ).as_posix()


def _matches_path_glob(relative_path: Path, pattern: str) -> bool:
    path_parts = PurePosixPath(relative_path.as_posix()).parts
    pattern_parts = PurePosixPath(pattern).parts

    def match_parts(path_index: int, pattern_index: int) -> bool:
        if pattern_index == len(pattern_parts):
            return path_index == len(path_parts)

        pattern_part = pattern_parts[pattern_index]
        if pattern_part == "**":
            if pattern_index == len(pattern_parts) - 1:
                return True
            return any(
                match_parts(next_index, pattern_index + 1)
                for next_index in range(path_index, len(path_parts) + 1)
            )

        if path_index >= len(path_parts):
            return False
        if not fnmatch.fnmatchcase(path_parts[path_index], pattern_part):
            return False
        return match_parts(path_index + 1, pattern_index + 1)

    return match_parts(0, 0)


def _resolve_root_confined_path(
    root_path: Path,
    path: str,
    *,
    expected_kind: str,
    reject_symlink: bool,
) -> _ResolvedRootPath:
    normalized_request = _normalize_requested_path(path)
    requested_path = Path(normalized_request)
    if requested_path.is_absolute():
        raise RepositoryError("Chat tool paths must be relative to the configured root")

    resolved_root = root_path.resolve()
    candidate_target = resolved_root / requested_path
    if reject_symlink and candidate_target.is_symlink():
        raise RepositoryError(
            "Requested path must not be a symlinked "
            f"{expected_kind}: {normalized_request}"
        )

    resolved_target = candidate_target.resolve()

    try:
        resolved_relative = resolved_target.relative_to(resolved_root)
    except ValueError as exc:
        raise RepositoryError(
            "Requested chat tool path escapes the configured root"
        ) from exc

    if not resolved_target.exists():
        raise RepositoryError(
            f"Requested {expected_kind} does not exist: {normalized_request}"
        )

    kind_check = (
        resolved_target.is_dir
        if expected_kind == "directory"
        else resolved_target.is_file
    )
    if not kind_check():
        raise RepositoryError(
            f"Requested path is not a {expected_kind}: {normalized_request}"
        )

    resolved_path = (
        resolved_relative.as_posix() if resolved_relative.as_posix() else "."
    )
    return _ResolvedRootPath(
        root=resolved_root,
        candidate=candidate_target,
        resolved=resolved_target,
        requested_path=normalized_request,
        resolved_path=resolved_path,
    )


def _resolve_directory_path(root_path: Path, path: str) -> _ResolvedRootPath:
    return _resolve_root_confined_path(
        root_path,
        path,
        expected_kind="directory",
        reject_symlink=True,
    )


def _resolve_file_path(root_path: Path, path: str) -> _ResolvedRootPath:
    return _resolve_root_confined_path(
        root_path,
        path,
        expected_kind="file",
        reject_symlink=True,
    )


def _is_hidden(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts if part not in {".", ".."})


def _entry_type(path: Path) -> DirectoryEntryType:
    if path.is_symlink():
        return "symlink"
    if path.is_dir():
        return "directory"
    if path.is_file():
        return "file"
    return "other"


def _should_include_entry(
    relative_path: Path,
    *,
    source_filters: ChatSourceFilters,
) -> bool:
    if _is_hidden(relative_path) and not source_filters.include_hidden:
        return False
    if _matches_patterns(relative_path, source_filters.exclude):
        return False
    if source_filters.include and not _matches_patterns(
        relative_path, source_filters.include
    ):
        return False
    return True


def _should_prune_directory(
    relative_path: Path,
    *,
    source_filters: ChatSourceFilters,
) -> bool:
    if _is_hidden(relative_path) and not source_filters.include_hidden:
        return True
    return _matches_patterns(relative_path, source_filters.exclude)


def _build_entry(root_path: Path, path: Path, *, depth: int) -> DirectoryEntry:
    relative_path = path.relative_to(root_path)
    return DirectoryEntry(
        path=relative_path.as_posix(),
        name=path.name,
        entry_type=_entry_type(path),
        depth=depth,
        is_hidden=_is_hidden(relative_path),
        is_symlink=path.is_symlink(),
    )


def _build_file_match(root_path: Path, path: Path) -> FileMatch:
    relative_path = path.relative_to(root_path)
    parent_path = relative_path.parent.as_posix()
    return FileMatch(
        path=relative_path.as_posix(),
        name=path.name,
        parent_path="." if parent_path == "." else parent_path,
        is_hidden=_is_hidden(relative_path),
    )


def _build_text_search_match(
    root_path: Path,
    path: Path,
    *,
    line_number: int,
    line_text: str,
) -> TextSearchMatch:
    relative_path = path.relative_to(root_path)
    return TextSearchMatch(
        path=relative_path.as_posix(),
        line_number=line_number,
        line_text=line_text,
        is_hidden=_is_hidden(relative_path),
    )


def _read_searchable_text(path: Path) -> str | None:
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None

    if "\x00" in content:
        return None
    return content


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


def _read_markitdown_content(path: Path) -> _LoadedReadableContent:
    cache_path = _markitdown_cache_path(path)
    if cache_path.exists():
        return _LoadedReadableContent(
            read_kind="markitdown",
            status="ok",
            content=cache_path.read_text(encoding="utf-8"),
        )

    try:
        converted = _convert_with_markitdown(path)
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


def _load_readable_content(path: Path) -> _LoadedReadableContent:
    text_content = _read_searchable_text(path)
    if text_content is not None:
        return _LoadedReadableContent(
            read_kind="text",
            status="ok",
            content=text_content,
        )

    if path.suffix.lower() in _MARKITDOWN_EXTENSIONS:
        return _read_markitdown_content(path)

    return _LoadedReadableContent(
        read_kind="unsupported",
        status="unsupported",
        content=None,
        error_message="File type is not supported for chat reads",
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


def list_directory(
    root_path: Path,
    path: str = ".",
    *,
    source_filters: ChatSourceFilters,
    tool_limits: ChatToolLimits,
) -> DirectoryListingResult:
    """Return the immediate matching children of one directory under the root."""

    resolved_request = _resolve_directory_path(root_path, path)
    entries: list[DirectoryEntry] = []
    truncated = False

    for child in sorted(
        resolved_request.resolved.iterdir(), key=lambda item: item.name
    ):
        relative_child = child.relative_to(resolved_request.root)
        if not _should_include_entry(relative_child, source_filters=source_filters):
            continue
        if len(entries) >= tool_limits.max_entries_per_call:
            truncated = True
            break
        entries.append(_build_entry(resolved_request.root, child, depth=1))

    return DirectoryListingResult(
        requested_path=resolved_request.requested_path,
        resolved_path=resolved_request.resolved_path,
        recursive=False,
        max_depth_applied=1,
        entries=entries,
        truncated=truncated,
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

    if max_depth is not None and max_depth <= 0:
        raise RepositoryError("max_depth must be greater than 0 when provided")

    resolved_request = _resolve_directory_path(root_path, path)
    applied_max_depth = min(
        max_depth if max_depth is not None else tool_limits.max_recursive_depth,
        tool_limits.max_recursive_depth,
    )
    entries: list[DirectoryEntry] = []
    truncated = False

    def walk_directory(directory: Path, *, depth: int) -> bool:
        nonlocal truncated

        for child in sorted(directory.iterdir(), key=lambda item: item.name):
            relative_child = child.relative_to(resolved_request.root)

            if _should_include_entry(relative_child, source_filters=source_filters):
                if len(entries) >= tool_limits.max_entries_per_call:
                    truncated = True
                    return False
                entries.append(_build_entry(resolved_request.root, child, depth=depth))

            if depth >= applied_max_depth:
                continue
            if child.is_symlink() or not child.is_dir():
                continue
            if _should_prune_directory(relative_child, source_filters=source_filters):
                continue
            if not walk_directory(child, depth=depth + 1):
                return False

        return True

    walk_directory(resolved_request.resolved, depth=1)
    return DirectoryListingResult(
        requested_path=resolved_request.requested_path,
        resolved_path=resolved_request.resolved_path,
        recursive=True,
        max_depth_applied=applied_max_depth,
        entries=entries,
        truncated=truncated,
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

    normalized_pattern = _normalize_required_pattern(pattern)
    resolved_request = _resolve_directory_path(root_path, path)
    matches: list[FileMatch] = []
    truncated = False

    def walk_directory(directory: Path) -> bool:
        nonlocal truncated

        for child in sorted(directory.iterdir(), key=lambda item: item.name):
            relative_child = child.relative_to(resolved_request.root)

            if child.is_symlink():
                continue

            if child.is_dir():
                if _should_prune_directory(
                    relative_child, source_filters=source_filters
                ):
                    continue
                if not walk_directory(child):
                    return False
                continue

            if not child.is_file():
                continue
            if not _should_include_entry(relative_child, source_filters=source_filters):
                continue
            if not _matches_path_glob(relative_child, normalized_pattern):
                continue
            if len(matches) >= tool_limits.max_entries_per_call:
                truncated = True
                return False
            matches.append(_build_file_match(resolved_request.root, child))

        return True

    walk_directory(resolved_request.resolved)
    return FileSearchResult(
        requested_path=resolved_request.requested_path,
        resolved_path=resolved_request.resolved_path,
        pattern=normalized_pattern,
        matches=matches,
        truncated=truncated,
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

    normalized_query = _normalize_required_value(query, field_name="search_text query")
    normalized_path = _normalize_requested_path(path)
    requested_path = Path(normalized_path)
    if requested_path.is_absolute():
        raise RepositoryError("Chat tool paths must be relative to the configured root")

    resolved_root = root_path.resolve()
    candidate_target = resolved_root / requested_path
    resolved_target = candidate_target.resolve()
    try:
        resolved_relative = resolved_target.relative_to(resolved_root)
    except ValueError as exc:
        raise RepositoryError(
            "Requested chat tool path escapes the configured root"
        ) from exc

    if candidate_target.is_symlink():
        symlink_kind = (
            "directory"
            if resolved_target.exists() and resolved_target.is_dir()
            else "file"
        )
        raise RepositoryError(
            f"Requested path must not be a symlinked {symlink_kind}: {normalized_path}"
        )
    if not candidate_target.exists():
        raise RepositoryError(
            f"Requested file or directory does not exist: {normalized_path}"
        )

    if resolved_target.is_file():
        resolved_path = (
            resolved_relative.as_posix() if resolved_relative.as_posix() else "."
        )
        resolved_request = _ResolvedRootPath(
            root=resolved_root,
            candidate=candidate_target,
            resolved=resolved_target,
            requested_path=normalized_path,
            resolved_path=resolved_path,
        )
        loaded_content = _load_readable_content(resolved_request.resolved)
        if loaded_content.status != "ok" or loaded_content.content is None:
            return TextSearchResult(
                requested_path=resolved_request.requested_path,
                resolved_path=resolved_request.resolved_path,
                query=normalized_query,
                matches=[],
                truncated=False,
            )
        if not _is_within_character_limit(
            loaded_content.content, tool_limits=tool_limits
        ):
            return TextSearchResult(
                requested_path=resolved_request.requested_path,
                resolved_path=resolved_request.resolved_path,
                query=normalized_query,
                matches=[],
                truncated=False,
            )

        file_matches: list[TextSearchMatch] = []
        truncated = False
        for line_number, line_text in enumerate(
            loaded_content.content.splitlines(), start=1
        ):
            if normalized_query not in line_text:
                continue
            if len(file_matches) >= tool_limits.max_search_matches:
                truncated = True
                break
            file_matches.append(
                _build_text_search_match(
                    resolved_request.root,
                    resolved_request.resolved,
                    line_number=line_number,
                    line_text=line_text,
                )
            )

        return TextSearchResult(
            requested_path=resolved_request.requested_path,
            resolved_path=resolved_request.resolved_path,
            query=normalized_query,
            matches=file_matches,
            truncated=truncated,
        )

    resolved_request = _resolve_directory_path(root_path, path)
    matches: list[TextSearchMatch] = []
    truncated = False

    def walk_directory(directory: Path) -> bool:
        nonlocal truncated

        for child in sorted(directory.iterdir(), key=lambda item: item.name):
            relative_child = child.relative_to(resolved_request.root)

            if child.is_symlink():
                continue

            if child.is_dir():
                if _should_prune_directory(
                    relative_child, source_filters=source_filters
                ):
                    continue
                if not walk_directory(child):
                    return False
                continue

            if not child.is_file():
                continue
            if not _should_include_entry(relative_child, source_filters=source_filters):
                continue

            loaded_content = _load_readable_content(child)
            if loaded_content.status != "ok" or loaded_content.content is None:
                continue
            if not _is_within_character_limit(
                loaded_content.content, tool_limits=tool_limits
            ):
                continue

            for line_number, line_text in enumerate(
                loaded_content.content.splitlines(), start=1
            ):
                if normalized_query not in line_text:
                    continue
                if len(matches) >= tool_limits.max_search_matches:
                    truncated = True
                    return False
                matches.append(
                    _build_text_search_match(
                        resolved_request.root,
                        child,
                        line_number=line_number,
                        line_text=line_text,
                    )
                )

        return True

    walk_directory(resolved_request.resolved)
    return TextSearchResult(
        requested_path=resolved_request.requested_path,
        resolved_path=resolved_request.resolved_path,
        query=normalized_query,
        matches=matches,
        truncated=truncated,
    )


def _get_single_file_info(
    root_path: Path,
    path: str,
    *,
    session_config: ChatSessionConfig,
    tool_limits: ChatToolLimits,
) -> FileInfoResult:
    resolved_request = _resolve_file_path(root_path, path)
    loaded_content = _load_readable_content(resolved_request.resolved)
    return _build_file_info_result(
        requested_path=resolved_request.requested_path,
        resolved_path=resolved_request.resolved_path,
        candidate_file=resolved_request.candidate,
        resolved_file=resolved_request.resolved,
        relative_candidate_path=resolved_request.candidate.relative_to(
            resolved_request.root
        ),
        session_config=session_config,
        tool_limits=tool_limits,
        loaded_content=loaded_content,
    )


def get_file_info(
    root_path: Path,
    path: str | list[str],
    *,
    session_config: ChatSessionConfig,
    tool_limits: ChatToolLimits,
) -> FileInfoResult | FileInfoBatchResult:
    """Return deterministic metadata for one or more root-confined files."""

    if isinstance(path, str):
        return _get_single_file_info(
            root_path,
            path,
            session_config=session_config,
            tool_limits=tool_limits,
        )

    results: list[FileInfoResult] = []
    for requested_path in path:
        try:
            result = _get_single_file_info(
                root_path,
                requested_path,
                session_config=session_config,
                tool_limits=tool_limits,
            )
        except RepositoryError as exc:
            cleaned_path = requested_path.strip()
            results.append(
                FileInfoResult(
                    requested_path=cleaned_path or requested_path,
                    resolved_path="",
                    name=Path(cleaned_path or requested_path).name,
                    size_bytes=0,
                    is_hidden=False,
                    is_symlink=False,
                    read_kind="unsupported",
                    status="error",
                    estimated_token_count=None,
                    character_count=None,
                    line_count=None,
                    max_file_size_characters=tool_limits.max_file_size_characters,
                    within_size_limit=False,
                    full_read_char_limit=_effective_full_read_char_limit(
                        session_config,
                        tool_limits,
                    ),
                    can_read_full=False,
                    error_message=str(exc),
                )
            )
            continue
        results.append(result)

    return FileInfoBatchResult(results=results)


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

    resolved_request = _resolve_file_path(root_path, path)
    loaded_content = _load_readable_content(resolved_request.resolved)
    file_info = _build_file_info_result(
        requested_path=resolved_request.requested_path,
        resolved_path=resolved_request.resolved_path,
        candidate_file=resolved_request.candidate,
        resolved_file=resolved_request.resolved,
        relative_candidate_path=resolved_request.candidate.relative_to(
            resolved_request.root
        ),
        session_config=session_config,
        tool_limits=tool_limits,
        loaded_content=loaded_content,
    )

    if file_info.status != "ok" or loaded_content.content is None:
        failure_status: FileReadStatus = (
            "unsupported" if file_info.status == "unsupported" else "error"
        )
        return FileReadResult(
            requested_path=resolved_request.requested_path,
            resolved_path=resolved_request.resolved_path,
            read_kind=file_info.read_kind,
            status=failure_status,
            content=None,
            file_size_bytes=file_info.size_bytes,
            max_file_size_characters=file_info.max_file_size_characters,
            full_read_char_limit=file_info.full_read_char_limit,
            character_count=file_info.character_count,
            estimated_token_count=file_info.estimated_token_count,
            error_message=file_info.error_message,
        )

    if not file_info.within_size_limit:
        return FileReadResult(
            requested_path=resolved_request.requested_path,
            resolved_path=resolved_request.resolved_path,
            read_kind=file_info.read_kind,
            status="too_large",
            content=None,
            file_size_bytes=file_info.size_bytes,
            max_file_size_characters=file_info.max_file_size_characters,
            full_read_char_limit=file_info.full_read_char_limit,
            character_count=file_info.character_count,
            estimated_token_count=file_info.estimated_token_count,
            error_message=(
                "Readable file content exceeds max_file_size_characters and cannot "
                "be returned by chat tools"
            ),
        )

    if start_char is None and end_char is None and not file_info.can_read_full:
        return FileReadResult(
            requested_path=resolved_request.requested_path,
            resolved_path=resolved_request.resolved_path,
            read_kind=file_info.read_kind,
            status="too_large",
            content=None,
            file_size_bytes=file_info.size_bytes,
            max_file_size_characters=file_info.max_file_size_characters,
            full_read_char_limit=file_info.full_read_char_limit,
            character_count=file_info.character_count,
            estimated_token_count=file_info.estimated_token_count,
            error_message=(
                "Full file read exceeds the configured max_read_file_chars limit; "
                "request a character range instead"
            ),
        )

    character_count = len(loaded_content.content)
    normalized_start, normalized_end = _normalize_range(
        start_char=start_char,
        end_char=end_char,
        character_count=character_count,
    )
    selected_content = loaded_content.content[normalized_start:normalized_end]
    truncated = False
    returned_end = normalized_end
    if len(selected_content) > tool_limits.max_tool_result_chars:
        selected_content = selected_content[: tool_limits.max_tool_result_chars]
        truncated = True
        returned_end = normalized_start + len(selected_content)

    return FileReadResult(
        requested_path=resolved_request.requested_path,
        resolved_path=resolved_request.resolved_path,
        read_kind=file_info.read_kind,
        status="ok",
        content=selected_content,
        truncated=truncated,
        content_char_count=len(selected_content),
        character_count=character_count,
        start_char=normalized_start,
        end_char=returned_end,
        file_size_bytes=file_info.size_bytes,
        max_file_size_characters=file_info.max_file_size_characters,
        full_read_char_limit=file_info.full_read_char_limit,
        estimated_token_count=file_info.estimated_token_count,
        error_message=None,
    )
