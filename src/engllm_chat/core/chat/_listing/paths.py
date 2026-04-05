"""Shared path and filter helpers for deterministic chat filesystem tools."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from engllm_chat.core.chat.models import (
    DirectoryEntry,
    DirectoryEntryType,
    FileMatch,
    TextSearchMatch,
)
from engllm_chat.domain.errors import RepositoryError
from engllm_chat.domain.models import ChatSourceFilters


@dataclass(frozen=True, slots=True)
class _ResolvedRootPath:
    """Normalized and root-confined request path metadata."""

    root: Path
    candidate: Path
    resolved: Path
    requested_path: str
    resolved_path: str


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
