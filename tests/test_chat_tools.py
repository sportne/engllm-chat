"""Tests for deterministic chat filesystem tools."""

from __future__ import annotations

from pathlib import Path

import pytest

import engllm_chat.core.chat.listing as chat_listing
from engllm_chat.core.chat import (
    find_files,
    get_file_info,
    list_directory,
    list_directory_recursive,
    read_file,
    search_text,
)
from engllm_chat.domain.errors import RepositoryError
from engllm_chat.domain.models import (
    ChatSessionConfig,
    ChatSourceFilters,
    ChatToolLimits,
)


def _write(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _make_symlink(link_path: Path, target: Path) -> None:
    try:
        link_path.symlink_to(target, target_is_directory=target.is_dir())
    except OSError as exc:
        pytest.skip(f"symlinks are not available in this environment: {exc}")


def test_list_directory_returns_immediate_entries_in_deterministic_order(
    tmp_path: Path,
) -> None:
    (tmp_path / "b_dir").mkdir()
    _write(tmp_path / "a.txt")
    _write(tmp_path / "b_dir" / "nested.py")

    result = list_directory(
        tmp_path,
        source_filters=ChatSourceFilters(),
        tool_limits=ChatToolLimits(),
    )

    assert result.requested_path == "."
    assert result.resolved_path == "."
    assert result.recursive is False
    assert result.max_depth_applied == 1
    assert [entry.path for entry in result.entries] == ["a.txt", "b_dir"]
    assert [entry.entry_type for entry in result.entries] == ["file", "directory"]
    assert all(entry.depth == 1 for entry in result.entries)


def test_list_directory_hidden_entries_respect_include_hidden(tmp_path: Path) -> None:
    _write(tmp_path / ".env")
    _write(tmp_path / "visible.txt")

    default_result = list_directory(
        tmp_path,
        source_filters=ChatSourceFilters(),
        tool_limits=ChatToolLimits(),
    )
    visible_result = list_directory(
        tmp_path,
        source_filters=ChatSourceFilters(include_hidden=True),
        tool_limits=ChatToolLimits(),
    )

    assert [entry.path for entry in default_result.entries] == ["visible.txt"]
    assert [entry.path for entry in visible_result.entries] == [".env", "visible.txt"]
    assert visible_result.entries[0].is_hidden is True


def test_listing_filters_affect_results_and_recursive_descent(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "pkg" / "module.py")
    _write(tmp_path / "src" / "pkg" / "module.txt")
    _write(tmp_path / "skip_me" / "ignored.py")

    recursive_result = list_directory_recursive(
        tmp_path,
        source_filters=ChatSourceFilters(
            include=["src/**/*.py", "src", "src/pkg"],
            exclude=["skip_me", "skip_me/**"],
        ),
        tool_limits=ChatToolLimits(),
    )
    direct_result = list_directory(
        tmp_path,
        source_filters=ChatSourceFilters(include=["src"]),
        tool_limits=ChatToolLimits(),
    )

    assert [entry.path for entry in direct_result.entries] == ["src"]
    assert [entry.path for entry in recursive_result.entries] == [
        "src",
        "src/pkg",
        "src/pkg/module.py",
    ]


def test_list_directory_recursive_returns_flat_depth_first_results(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "alpha" / "beta" / "c.txt")
    _write(tmp_path / "alpha" / "a.txt")
    _write(tmp_path / "z.txt")

    result = list_directory_recursive(
        tmp_path,
        source_filters=ChatSourceFilters(),
        tool_limits=ChatToolLimits(),
    )

    assert [entry.path for entry in result.entries] == [
        "alpha",
        "alpha/a.txt",
        "alpha/beta",
        "alpha/beta/c.txt",
        "z.txt",
    ]
    assert [entry.depth for entry in result.entries] == [1, 2, 2, 3, 1]


def test_recursive_listing_respects_depth_limits(tmp_path: Path) -> None:
    _write(tmp_path / "a" / "b" / "c" / "deep.txt")

    config_limited = list_directory_recursive(
        tmp_path,
        source_filters=ChatSourceFilters(),
        tool_limits=ChatToolLimits(max_recursive_depth=2),
    )
    override_limited = list_directory_recursive(
        tmp_path,
        source_filters=ChatSourceFilters(),
        tool_limits=ChatToolLimits(max_recursive_depth=5),
        max_depth=2,
    )

    assert config_limited.max_depth_applied == 2
    assert [entry.path for entry in config_limited.entries] == ["a", "a/b"]
    assert override_limited.max_depth_applied == 2
    assert [entry.path for entry in override_limited.entries] == ["a", "a/b"]


def test_recursive_listing_lists_symlink_but_does_not_descend(tmp_path: Path) -> None:
    (tmp_path / "real_dir").mkdir()
    _write(tmp_path / "real_dir" / "inside.txt")
    _make_symlink(tmp_path / "linked_dir", tmp_path / "real_dir")

    result = list_directory_recursive(
        tmp_path,
        source_filters=ChatSourceFilters(),
        tool_limits=ChatToolLimits(),
    )

    assert [entry.path for entry in result.entries] == [
        "linked_dir",
        "real_dir",
        "real_dir/inside.txt",
    ]
    assert result.entries[0].entry_type == "symlink"
    assert result.entries[0].is_symlink is True


def test_listing_truncates_direct_and_recursive_results(tmp_path: Path) -> None:
    _write(tmp_path / "a.txt")
    _write(tmp_path / "b.txt")
    _write(tmp_path / "dir" / "c.txt")

    direct_result = list_directory(
        tmp_path,
        source_filters=ChatSourceFilters(),
        tool_limits=ChatToolLimits(max_entries_per_call=1),
    )
    recursive_result = list_directory_recursive(
        tmp_path,
        source_filters=ChatSourceFilters(),
        tool_limits=ChatToolLimits(max_entries_per_call=2),
    )

    assert [entry.path for entry in direct_result.entries] == ["a.txt"]
    assert direct_result.truncated is True
    assert [entry.path for entry in recursive_result.entries] == ["a.txt", "b.txt"]
    assert recursive_result.truncated is True


def test_listing_rejects_path_escape(tmp_path: Path) -> None:
    with pytest.raises(
        RepositoryError, match="Requested chat tool path escapes the configured root"
    ):
        list_directory(
            tmp_path,
            "../outside",
            source_filters=ChatSourceFilters(),
            tool_limits=ChatToolLimits(),
        )


def test_listing_rejects_nonexistent_path(tmp_path: Path) -> None:
    with pytest.raises(RepositoryError, match="Requested directory does not exist"):
        list_directory(
            tmp_path,
            "missing",
            source_filters=ChatSourceFilters(),
            tool_limits=ChatToolLimits(),
        )


def test_listing_rejects_file_path_and_invalid_max_depth(tmp_path: Path) -> None:
    _write(tmp_path / "file.txt")

    with pytest.raises(RepositoryError, match="Requested path is not a directory"):
        list_directory(
            tmp_path,
            "file.txt",
            source_filters=ChatSourceFilters(),
            tool_limits=ChatToolLimits(),
        )

    with pytest.raises(RepositoryError, match="max_depth must be greater than 0"):
        list_directory_recursive(
            tmp_path,
            source_filters=ChatSourceFilters(),
            tool_limits=ChatToolLimits(),
            max_depth=0,
        )


def test_find_files_returns_recursive_glob_matches_in_deterministic_order(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "src" / "a.py")
    _write(tmp_path / "src" / "nested" / "b.py")
    _write(tmp_path / "src" / "nested" / "c.txt")

    result = find_files(
        tmp_path,
        "**/*.py",
        source_filters=ChatSourceFilters(),
        tool_limits=ChatToolLimits(),
    )

    assert result.requested_path == "."
    assert result.resolved_path == "."
    assert result.pattern == "**/*.py"
    assert [match.path for match in result.matches] == ["src/a.py", "src/nested/b.py"]
    assert [match.parent_path for match in result.matches] == ["src", "src/nested"]


def test_find_files_supports_subtree_scoped_search(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "pkg" / "module.py")
    _write(tmp_path / "tests" / "test_module.py")

    result = find_files(
        tmp_path,
        "**/*.py",
        path="src",
        source_filters=ChatSourceFilters(),
        tool_limits=ChatToolLimits(),
    )

    assert result.requested_path == "src"
    assert result.resolved_path == "src"
    assert [match.path for match in result.matches] == ["src/pkg/module.py"]


def test_find_files_matches_root_relative_paths_not_basenames(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "module.py")
    _write(tmp_path / "module.py")

    result = find_files(
        tmp_path,
        "src/*.py",
        source_filters=ChatSourceFilters(),
        tool_limits=ChatToolLimits(),
    )

    assert [match.path for match in result.matches] == ["src/module.py"]


def test_find_files_respects_hidden_include_exclude_and_symlink_rules(
    tmp_path: Path,
) -> None:
    _write(tmp_path / ".hidden.py")
    _write(tmp_path / "visible.py")
    _write(tmp_path / "skip_me" / "ignored.py")
    (tmp_path / "real_dir").mkdir()
    _write(tmp_path / "real_dir" / "real.py")
    _make_symlink(tmp_path / "linked_dir", tmp_path / "real_dir")

    default_result = find_files(
        tmp_path,
        "**/*.py",
        source_filters=ChatSourceFilters(exclude=["skip_me", "skip_me/**"]),
        tool_limits=ChatToolLimits(),
    )
    hidden_result = find_files(
        tmp_path,
        "**/*.py",
        source_filters=ChatSourceFilters(
            include_hidden=True,
            exclude=["skip_me", "skip_me/**"],
        ),
        tool_limits=ChatToolLimits(),
    )

    assert [match.path for match in default_result.matches] == [
        "real_dir/real.py",
        "visible.py",
    ]
    assert [match.path for match in hidden_result.matches] == [
        ".hidden.py",
        "real_dir/real.py",
        "visible.py",
    ]

    hidden_root_result = find_files(
        tmp_path,
        "*.py",
        source_filters=ChatSourceFilters(include_hidden=True),
        tool_limits=ChatToolLimits(),
    )
    assert [match.path for match in hidden_root_result.matches] == [
        ".hidden.py",
        "visible.py",
    ]


def test_find_files_truncates_results(tmp_path: Path) -> None:
    _write(tmp_path / "a.py")
    _write(tmp_path / "b.py")
    _write(tmp_path / "c.py")

    result = find_files(
        tmp_path,
        "*.py",
        source_filters=ChatSourceFilters(),
        tool_limits=ChatToolLimits(max_entries_per_call=2),
    )

    assert [match.path for match in result.matches] == ["a.py", "b.py"]
    assert result.truncated is True


def test_find_files_rejects_escape_missing_file_root_and_blank_pattern(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "file.txt")

    with pytest.raises(
        RepositoryError, match="Requested chat tool path escapes the configured root"
    ):
        find_files(
            tmp_path,
            "*.py",
            "../outside",
            source_filters=ChatSourceFilters(),
            tool_limits=ChatToolLimits(),
        )

    with pytest.raises(RepositoryError, match="Requested directory does not exist"):
        find_files(
            tmp_path,
            "*.py",
            "missing",
            source_filters=ChatSourceFilters(),
            tool_limits=ChatToolLimits(),
        )

    with pytest.raises(RepositoryError, match="Requested path is not a directory"):
        find_files(
            tmp_path,
            "*.py",
            "file.txt",
            source_filters=ChatSourceFilters(),
            tool_limits=ChatToolLimits(),
        )

    with pytest.raises(RepositoryError, match="find_files pattern must not be empty"):
        find_files(
            tmp_path,
            "   ",
            source_filters=ChatSourceFilters(),
            tool_limits=ChatToolLimits(),
        )


def test_search_text_returns_literal_matches_in_deterministic_order(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "src" / "a.py", "needle first\nmiss\nneedle again\n")
    _write(tmp_path / "src" / "nested" / "b.py", "zzz\nneedle nested\n")

    result = search_text(
        tmp_path,
        "needle",
        source_filters=ChatSourceFilters(),
        tool_limits=ChatToolLimits(),
    )

    assert result.requested_path == "."
    assert result.resolved_path == "."
    assert result.query == "needle"
    assert [(match.path, match.line_number) for match in result.matches] == [
        ("src/a.py", 1),
        ("src/a.py", 3),
        ("src/nested/b.py", 2),
    ]
    assert [match.line_text for match in result.matches] == [
        "needle first",
        "needle again",
        "needle nested",
    ]


def test_search_text_supports_subtree_scoped_search(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "pkg" / "module.py", "needle\n")
    _write(tmp_path / "tests" / "test_module.py", "needle\n")

    result = search_text(
        tmp_path,
        "needle",
        path="src",
        source_filters=ChatSourceFilters(),
        tool_limits=ChatToolLimits(),
    )

    assert result.requested_path == "src"
    assert result.resolved_path == "src"
    assert [match.path for match in result.matches] == ["src/pkg/module.py"]


def test_search_text_is_case_sensitive_and_returns_matching_line_only(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "sample.txt", "Needle\nneedle here\nneedle there\n")

    result = search_text(
        tmp_path,
        "needle",
        source_filters=ChatSourceFilters(),
        tool_limits=ChatToolLimits(),
    )

    assert [(match.line_number, match.line_text) for match in result.matches] == [
        (2, "needle here"),
        (3, "needle there"),
    ]


def test_search_text_respects_hidden_include_exclude_and_symlink_rules(
    tmp_path: Path,
) -> None:
    _write(tmp_path / ".hidden.txt", "needle hidden\n")
    _write(tmp_path / "visible.txt", "needle visible\n")
    _write(tmp_path / "skip_me" / "ignored.txt", "needle skip\n")
    (tmp_path / "real_dir").mkdir()
    _write(tmp_path / "real_dir" / "real.txt", "needle real\n")
    _make_symlink(tmp_path / "linked_dir", tmp_path / "real_dir")

    default_result = search_text(
        tmp_path,
        "needle",
        source_filters=ChatSourceFilters(exclude=["skip_me", "skip_me/**"]),
        tool_limits=ChatToolLimits(),
    )
    hidden_result = search_text(
        tmp_path,
        "needle",
        source_filters=ChatSourceFilters(
            include_hidden=True,
            exclude=["skip_me", "skip_me/**"],
        ),
        tool_limits=ChatToolLimits(),
    )

    assert [match.path for match in default_result.matches] == [
        "real_dir/real.txt",
        "visible.txt",
    ]
    assert [match.path for match in hidden_result.matches] == [
        ".hidden.txt",
        "real_dir/real.txt",
        "visible.txt",
    ]


def test_search_text_include_filters_gate_files_without_pruning_descent(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "src" / "pkg" / "match.py", "needle\n")
    _write(tmp_path / "src" / "pkg" / "skip.txt", "needle\n")

    result = search_text(
        tmp_path,
        "needle",
        source_filters=ChatSourceFilters(include=["src/**/*.py"]),
        tool_limits=ChatToolLimits(),
    )

    assert [match.path for match in result.matches] == ["src/pkg/match.py"]


def test_search_text_skips_non_text_and_unreadable_files(tmp_path: Path) -> None:
    _write(tmp_path / "visible.txt", "needle visible\n")
    (tmp_path / "binary.bin").write_bytes(b"\xff\xfe\xfd")
    _write(tmp_path / "blocked.txt", "needle blocked\n")

    original_read_text = Path.read_text

    def fake_read_text(self: Path, *args: object, **kwargs: object) -> str:
        if self.name == "blocked.txt":
            raise OSError("permission denied")
        return original_read_text(self, *args, **kwargs)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(Path, "read_text", fake_read_text)
    try:
        result = search_text(
            tmp_path,
            "needle",
            source_filters=ChatSourceFilters(),
            tool_limits=ChatToolLimits(),
        )
    finally:
        monkeypatch.undo()

    assert [match.path for match in result.matches] == ["visible.txt"]


def test_search_text_truncates_results(tmp_path: Path) -> None:
    _write(tmp_path / "a.txt", "needle one\n")
    _write(tmp_path / "b.txt", "needle two\n")
    _write(tmp_path / "c.txt", "needle three\n")

    result = search_text(
        tmp_path,
        "needle",
        source_filters=ChatSourceFilters(),
        tool_limits=ChatToolLimits(max_search_matches=2),
    )

    assert [match.path for match in result.matches] == ["a.txt", "b.txt"]
    assert result.truncated is True


def test_search_text_rejects_escape_missing_file_root_and_blank_query(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "file.txt", "needle\n")

    with pytest.raises(
        RepositoryError, match="Requested chat tool path escapes the configured root"
    ):
        search_text(
            tmp_path,
            "needle",
            "../outside",
            source_filters=ChatSourceFilters(),
            tool_limits=ChatToolLimits(),
        )

    with pytest.raises(RepositoryError, match="Requested directory does not exist"):
        search_text(
            tmp_path,
            "needle",
            "missing",
            source_filters=ChatSourceFilters(),
            tool_limits=ChatToolLimits(),
        )

    with pytest.raises(RepositoryError, match="Requested path is not a directory"):
        search_text(
            tmp_path,
            "needle",
            "file.txt",
            source_filters=ChatSourceFilters(),
            tool_limits=ChatToolLimits(),
        )

    with pytest.raises(RepositoryError, match="search_text query must not be empty"):
        search_text(
            tmp_path,
            "   ",
            source_filters=ChatSourceFilters(),
            tool_limits=ChatToolLimits(),
        )


def test_get_file_info_returns_text_metadata_and_context_derived_limit(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "notes.txt", "alpha beta\nsecond line\n")

    result = get_file_info(
        tmp_path,
        "notes.txt",
        session_config=ChatSessionConfig(max_context_tokens=10),
        tool_limits=ChatToolLimits(),
    )

    assert result.requested_path == "notes.txt"
    assert result.resolved_path == "notes.txt"
    assert result.read_kind == "text"
    assert result.status == "ok"
    assert result.size_bytes == len(b"alpha beta\nsecond line\n")
    assert result.character_count == len("alpha beta\nsecond line\n")
    assert result.line_count == 2
    assert result.estimated_token_count == 4
    assert result.max_file_size_characters == 262144
    assert result.within_size_limit is True
    assert result.full_read_char_limit == 40
    assert result.can_read_full is True


def test_get_file_info_respects_read_char_override(tmp_path: Path) -> None:
    _write(tmp_path / "notes.txt", "abcdef")

    result = get_file_info(
        tmp_path,
        "notes.txt",
        session_config=ChatSessionConfig(max_context_tokens=100),
        tool_limits=ChatToolLimits(max_read_file_chars=4),
    )

    assert result.full_read_char_limit == 4
    assert result.can_read_full is False


def test_get_file_info_reports_when_file_exceeds_character_cap(tmp_path: Path) -> None:
    _write(tmp_path / "notes.txt", "abcdef")

    result = get_file_info(
        tmp_path,
        "notes.txt",
        session_config=ChatSessionConfig(max_context_tokens=100),
        tool_limits=ChatToolLimits(max_file_size_characters=4),
    )

    assert result.max_file_size_characters == 4
    assert result.within_size_limit is False
    assert result.can_read_full is False


def test_get_file_info_reports_unsupported_binary_files(tmp_path: Path) -> None:
    (tmp_path / "blob.bin").write_bytes(b"\xff\xfe\x00\x01")

    result = get_file_info(
        tmp_path,
        "blob.bin",
        session_config=ChatSessionConfig(),
        tool_limits=ChatToolLimits(),
    )

    assert result.read_kind == "unsupported"
    assert result.status == "unsupported"
    assert result.character_count is None
    assert result.estimated_token_count is None
    assert result.within_size_limit is False
    assert result.can_read_full is False


def test_read_file_returns_full_text_for_small_files(tmp_path: Path) -> None:
    _write(tmp_path / "notes.txt", "alpha beta\nsecond line\n")

    result = read_file(
        tmp_path,
        "notes.txt",
        session_config=ChatSessionConfig(max_context_tokens=100),
        tool_limits=ChatToolLimits(),
    )

    assert result.status == "ok"
    assert result.read_kind == "text"
    assert result.content == "alpha beta\nsecond line\n"
    assert result.start_char == 0
    assert result.end_char == len("alpha beta\nsecond line\n")
    assert result.content_char_count == len("alpha beta\nsecond line\n")
    assert result.truncated is False


def test_read_file_supports_character_ranges_for_text_files(tmp_path: Path) -> None:
    _write(tmp_path / "notes.txt", "0123456789")

    result = read_file(
        tmp_path,
        "notes.txt",
        session_config=ChatSessionConfig(),
        tool_limits=ChatToolLimits(),
        start_char=2,
        end_char=7,
    )

    assert result.status == "ok"
    assert result.content == "23456"
    assert result.start_char == 2
    assert result.end_char == 7
    assert result.character_count == 10


def test_read_file_reports_too_large_but_allows_ranges(tmp_path: Path) -> None:
    _write(tmp_path / "big.txt", "0123456789" * 6)

    oversized = read_file(
        tmp_path,
        "big.txt",
        session_config=ChatSessionConfig(max_context_tokens=5),
        tool_limits=ChatToolLimits(),
    )
    ranged = read_file(
        tmp_path,
        "big.txt",
        session_config=ChatSessionConfig(max_context_tokens=5),
        tool_limits=ChatToolLimits(),
        start_char=10,
        end_char=20,
    )

    assert oversized.status == "too_large"
    assert oversized.content is None
    assert oversized.character_count == 60
    assert ranged.status == "ok"
    assert ranged.content == "0123456789"
    assert ranged.start_char == 10
    assert ranged.end_char == 20


def test_read_file_rejects_files_over_source_character_cap(tmp_path: Path) -> None:
    _write(tmp_path / "big.txt", "0123456789")

    full_result = read_file(
        tmp_path,
        "big.txt",
        session_config=ChatSessionConfig(max_context_tokens=100),
        tool_limits=ChatToolLimits(max_file_size_characters=5),
    )
    ranged_result = read_file(
        tmp_path,
        "big.txt",
        session_config=ChatSessionConfig(max_context_tokens=100),
        tool_limits=ChatToolLimits(max_file_size_characters=5),
        start_char=1,
        end_char=3,
    )

    assert full_result.status == "too_large"
    assert full_result.max_file_size_characters == 5
    assert "max_file_size_characters" in (full_result.error_message or "")
    assert ranged_result.status == "too_large"
    assert ranged_result.content is None


def test_read_file_truncates_to_tool_result_chars(tmp_path: Path) -> None:
    _write(tmp_path / "notes.txt", "abcdefghij")

    result = read_file(
        tmp_path,
        "notes.txt",
        session_config=ChatSessionConfig(),
        tool_limits=ChatToolLimits(max_tool_result_chars=4),
    )

    assert result.status == "ok"
    assert result.content == "abcd"
    assert result.truncated is True
    assert result.start_char == 0
    assert result.end_char == 4


def test_search_text_skips_files_over_character_cap(tmp_path: Path) -> None:
    _write(tmp_path / "small.txt", "needle ok\n")
    _write(tmp_path / "big.txt", "needle " + ("x" * 20))

    result = search_text(
        tmp_path,
        "needle",
        source_filters=ChatSourceFilters(),
        tool_limits=ChatToolLimits(max_file_size_characters=10),
    )

    assert [match.path for match in result.matches] == ["small.txt"]


def test_markitdown_reads_support_metadata_ranges_and_cache_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sample_file = tmp_path / "report.pdf"
    sample_file.write_bytes(b"%PDF-\xff binary")
    cache_root = tmp_path / "cache"
    calls: list[str] = []

    def fake_convert(path: Path) -> str:
        calls.append(path.name)
        return "# Converted\n\nThis is cached markdown."

    monkeypatch.setattr(chat_listing, "_convert_with_markitdown", fake_convert)
    monkeypatch.setattr(chat_listing, "_markitdown_cache_root", lambda: cache_root)

    info = get_file_info(
        tmp_path,
        "report.pdf",
        session_config=ChatSessionConfig(max_context_tokens=100),
        tool_limits=ChatToolLimits(),
    )
    full = read_file(
        tmp_path,
        "report.pdf",
        session_config=ChatSessionConfig(max_context_tokens=100),
        tool_limits=ChatToolLimits(),
    )
    ranged = read_file(
        tmp_path,
        "report.pdf",
        session_config=ChatSessionConfig(max_context_tokens=100),
        tool_limits=ChatToolLimits(),
        start_char=2,
        end_char=11,
    )

    assert info.read_kind == "markitdown"
    assert info.status == "ok"
    assert info.character_count == len("# Converted\n\nThis is cached markdown.")
    assert full.content == "# Converted\n\nThis is cached markdown."
    assert ranged.content == "Converted"
    assert calls == ["report.pdf"]
    assert any(cache_root.iterdir())


def test_search_text_uses_markitdown_content_when_within_character_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sample_file = tmp_path / "report.pdf"
    sample_file.write_bytes(b"%PDF-\xff binary")
    monkeypatch.setattr(
        chat_listing, "_convert_with_markitdown", lambda path: "needle in markdown"
    )
    monkeypatch.setattr(chat_listing, "_markitdown_cache_root", lambda: tmp_path / "c")

    result = search_text(
        tmp_path,
        "needle",
        source_filters=ChatSourceFilters(),
        tool_limits=ChatToolLimits(),
    )

    assert [match.path for match in result.matches] == ["report.pdf"]


def test_directory_tools_reject_requested_symlinked_directory(
    tmp_path: Path,
) -> None:
    (tmp_path / "real_dir").mkdir()
    _write(tmp_path / "real_dir" / "inside.txt", "needle\n")
    _make_symlink(tmp_path / "linked_dir", tmp_path / "real_dir")

    with pytest.raises(RepositoryError, match="symlinked directory"):
        list_directory(
            tmp_path,
            "linked_dir",
            source_filters=ChatSourceFilters(),
            tool_limits=ChatToolLimits(),
        )

    with pytest.raises(RepositoryError, match="symlinked directory"):
        list_directory_recursive(
            tmp_path,
            "linked_dir",
            source_filters=ChatSourceFilters(),
            tool_limits=ChatToolLimits(),
        )

    with pytest.raises(RepositoryError, match="symlinked directory"):
        find_files(
            tmp_path,
            "**/*.txt",
            "linked_dir",
            source_filters=ChatSourceFilters(),
            tool_limits=ChatToolLimits(),
        )

    with pytest.raises(RepositoryError, match="symlinked directory"):
        search_text(
            tmp_path,
            "needle",
            "linked_dir",
            source_filters=ChatSourceFilters(),
            tool_limits=ChatToolLimits(),
        )


def test_file_tools_reject_requested_symlinked_file(tmp_path: Path) -> None:
    _write(tmp_path / "real.txt", "hello")
    _make_symlink(tmp_path / "linked.txt", tmp_path / "real.txt")

    with pytest.raises(RepositoryError, match="symlinked file"):
        get_file_info(
            tmp_path,
            "linked.txt",
            session_config=ChatSessionConfig(),
            tool_limits=ChatToolLimits(),
        )

    with pytest.raises(RepositoryError, match="symlinked file"):
        read_file(
            tmp_path,
            "linked.txt",
            session_config=ChatSessionConfig(),
            tool_limits=ChatToolLimits(),
        )


def test_markitdown_conversion_failures_return_structured_error_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sample_file = tmp_path / "report.pdf"
    sample_file.write_bytes(b"%PDF-\xff binary")
    monkeypatch.setattr(
        chat_listing,
        "_convert_with_markitdown",
        lambda path: (_ for _ in ()).throw(RuntimeError("conversion failed")),
    )
    monkeypatch.setattr(chat_listing, "_markitdown_cache_root", lambda: tmp_path / "c")

    info = get_file_info(
        tmp_path,
        "report.pdf",
        session_config=ChatSessionConfig(),
        tool_limits=ChatToolLimits(),
    )
    result = read_file(
        tmp_path,
        "report.pdf",
        session_config=ChatSessionConfig(),
        tool_limits=ChatToolLimits(),
    )

    assert info.status == "error"
    assert "conversion failed" in (info.error_message or "")
    assert result.status == "error"
    assert "conversion failed" in (result.error_message or "")


def test_read_file_rejects_invalid_ranges_and_bad_paths(tmp_path: Path) -> None:
    _write(tmp_path / "notes.txt", "abcdef")

    with pytest.raises(RepositoryError, match="start_char must be greater than"):
        read_file(
            tmp_path,
            "notes.txt",
            session_config=ChatSessionConfig(),
            tool_limits=ChatToolLimits(),
            start_char=-1,
        )

    with pytest.raises(RepositoryError, match="end_char must be greater than"):
        read_file(
            tmp_path,
            "notes.txt",
            session_config=ChatSessionConfig(),
            tool_limits=ChatToolLimits(),
            start_char=3,
            end_char=3,
        )

    with pytest.raises(RepositoryError, match="start_char must not exceed"):
        read_file(
            tmp_path,
            "notes.txt",
            session_config=ChatSessionConfig(),
            tool_limits=ChatToolLimits(),
            start_char=20,
        )

    with pytest.raises(
        RepositoryError, match="Requested chat tool path escapes the configured root"
    ):
        get_file_info(
            tmp_path,
            "../outside",
            session_config=ChatSessionConfig(),
            tool_limits=ChatToolLimits(),
        )

    with pytest.raises(RepositoryError, match="Requested file does not exist"):
        read_file(
            tmp_path,
            "missing.txt",
            session_config=ChatSessionConfig(),
            tool_limits=ChatToolLimits(),
        )

    with pytest.raises(RepositoryError, match="Requested path is not a file"):
        read_file(
            tmp_path,
            ".",
            session_config=ChatSessionConfig(),
            tool_limits=ChatToolLimits(),
        )
