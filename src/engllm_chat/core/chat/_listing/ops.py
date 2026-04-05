"""Operational implementations for deterministic chat filesystem tools."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from engllm_chat.core.chat.models import (
    DirectoryEntry,
    DirectoryListingResult,
    FileInfoBatchResult,
    FileInfoResult,
    FileMatch,
    FileReadResult,
    FileReadStatus,
    FileSearchResult,
    TextSearchMatch,
    TextSearchResult,
)
from engllm_chat.domain.errors import RepositoryError
from engllm_chat.domain.models import (
    ChatSessionConfig,
    ChatSourceFilters,
    ChatToolLimits,
)

from .content import (
    _build_file_info_result,
    _effective_full_read_char_limit,
    _is_within_character_limit,
    _LoadedReadableContent,
    _normalize_range,
)
from .paths import (
    _build_entry,
    _build_file_match,
    _build_text_search_match,
    _matches_path_glob,
    _normalize_requested_path,
    _normalize_required_pattern,
    _normalize_required_value,
    _resolve_directory_path,
    _resolve_file_path,
    _ResolvedRootPath,
    _should_include_entry,
    _should_prune_directory,
)

ReadableContentLoader = Callable[[Path], _LoadedReadableContent]


def list_directory_impl(
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


def list_directory_recursive_impl(
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


def find_files_impl(
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


def _resolve_search_file_or_directory(
    root_path: Path,
    path: str,
) -> tuple[str, Path, Path, Path]:
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

    return normalized_path, resolved_root, candidate_target, resolved_relative


def search_text_impl(
    root_path: Path,
    query: str,
    path: str = ".",
    *,
    source_filters: ChatSourceFilters,
    tool_limits: ChatToolLimits,
    load_readable_content: ReadableContentLoader,
) -> TextSearchResult:
    """Return matching text lines within one root-confined directory or file."""

    normalized_query = _normalize_required_value(query, field_name="search_text query")
    normalized_path, resolved_root, candidate_target, resolved_relative = (
        _resolve_search_file_or_directory(root_path, path)
    )

    if candidate_target.resolve().is_file():
        resolved_path = (
            resolved_relative.as_posix() if resolved_relative.as_posix() else "."
        )
        resolved_request = _ResolvedRootPath(
            root=resolved_root,
            candidate=candidate_target,
            resolved=candidate_target.resolve(),
            requested_path=normalized_path,
            resolved_path=resolved_path,
        )
        loaded_content = load_readable_content(resolved_request.resolved)
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

            loaded_content = load_readable_content(child)
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


def _get_single_file_info_impl(
    root_path: Path,
    path: str,
    *,
    session_config: ChatSessionConfig,
    tool_limits: ChatToolLimits,
    load_readable_content: ReadableContentLoader,
) -> FileInfoResult:
    resolved_request = _resolve_file_path(root_path, path)
    loaded_content = load_readable_content(resolved_request.resolved)
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


def get_file_info_impl(
    root_path: Path,
    path: str | list[str],
    *,
    session_config: ChatSessionConfig,
    tool_limits: ChatToolLimits,
    load_readable_content: ReadableContentLoader,
) -> FileInfoResult | FileInfoBatchResult:
    """Return deterministic metadata for one or more root-confined files."""

    # `get_file_info` is intentionally separate from `read_file`: the model can
    # inspect size/readability metadata first and choose a safer next step
    # before asking for the actual content.
    if isinstance(path, str):
        return _get_single_file_info_impl(
            root_path,
            path,
            session_config=session_config,
            tool_limits=tool_limits,
            load_readable_content=load_readable_content,
        )

    results: list[FileInfoResult] = []
    for requested_path in path:
        try:
            result = _get_single_file_info_impl(
                root_path,
                requested_path,
                session_config=session_config,
                tool_limits=tool_limits,
                load_readable_content=load_readable_content,
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


def read_file_impl(
    root_path: Path,
    path: str,
    *,
    session_config: ChatSessionConfig,
    tool_limits: ChatToolLimits,
    start_char: int | None = None,
    end_char: int | None = None,
    load_readable_content: ReadableContentLoader,
) -> FileReadResult:
    """Return bounded text or markdown content for one root-confined file."""

    # `read_file` is the content-returning tool. It depends on the prior
    # metadata and readable-content checks so full reads and ranged reads stay
    # bounded and predictable.
    resolved_request = _resolve_file_path(root_path, path)
    loaded_content = load_readable_content(resolved_request.resolved)
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
