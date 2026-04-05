"""Focused tests for internal chat listing helper modules."""

from __future__ import annotations

from pathlib import Path

from engllm_chat.core.chat._listing import content, ops, paths
from engllm_chat.domain.models import (
    ChatSessionConfig,
    ChatSourceFilters,
    ChatToolLimits,
)


def _write(path: Path, content_text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content_text, encoding="utf-8")


def test_paths_module_resolves_directory_and_filters_entries(tmp_path: Path) -> None:
    _write(tmp_path / ".hidden" / "nested.txt")
    _write(tmp_path / "src" / "keep.py")
    _write(tmp_path / "src" / "skip.txt")

    resolved = paths._resolve_directory_path(tmp_path, "src")
    include_filters = ChatSourceFilters(
        include=["src", "src/*.py"],
        exclude=["skip/**"],
    )

    assert resolved.requested_path == "src"
    assert resolved.resolved_path == "src"
    assert (
        paths._should_include_entry(Path("src/keep.py"), source_filters=include_filters)
        is True
    )
    assert (
        paths._should_include_entry(
            Path("src/skip.txt"), source_filters=include_filters
        )
        is False
    )
    assert (
        paths._should_prune_directory(
            Path(".hidden"), source_filters=ChatSourceFilters()
        )
        is True
    )


def test_paths_module_matches_recursive_globs() -> None:
    assert paths._matches_path_glob(Path("src/app/main.py"), "**/*.py") is True
    assert paths._matches_path_glob(Path("src/app/main.py"), "src/*/*.py") is True
    assert paths._matches_path_glob(Path("src/app/main.py"), "src/*.py") is False


def test_content_module_loads_markitdown_using_injected_hooks(tmp_path: Path) -> None:
    sample_file = tmp_path / "report.pdf"
    sample_file.write_bytes(b"%PDF-\xff binary")
    cache_path = tmp_path / "cache" / "report.md"
    calls: list[str] = []

    loaded = content.load_readable_content(
        sample_file,
        markitdown_cache_path=lambda path: cache_path,
        convert_with_markitdown=lambda path: calls.append(path.name)
        or "# Converted\n\nBody",
    )
    cached = content.load_readable_content(
        sample_file,
        markitdown_cache_path=lambda path: cache_path,
        convert_with_markitdown=lambda path: (_ for _ in ()).throw(
            AssertionError("cache should have been reused")
        ),
    )

    assert loaded.read_kind == "markitdown"
    assert loaded.content == "# Converted\n\nBody"
    assert cached.content == "# Converted\n\nBody"
    assert calls == ["report.pdf"]


def test_content_module_builds_file_metadata_and_normalizes_ranges(
    tmp_path: Path,
) -> None:
    sample_file = tmp_path / "notes.txt"
    _write(sample_file, "line one\nline two")
    loaded = content._LoadedReadableContent(
        read_kind="text",
        status="ok",
        content="line one\nline two",
    )

    info = content._build_file_info_result(
        requested_path="notes.txt",
        resolved_path="notes.txt",
        candidate_file=sample_file,
        resolved_file=sample_file,
        relative_candidate_path=Path("notes.txt"),
        session_config=ChatSessionConfig(max_context_tokens=10),
        tool_limits=ChatToolLimits(max_read_lines=10),
        loaded_content=loaded,
    )

    assert info.character_count == len("line one\nline two")
    assert info.line_count == 2
    assert info.can_read_full is True
    assert content._normalize_range(
        start_char=5,
        end_char=999,
        character_count=len("line one\nline two"),
    ) == (5, len("line one\nline two"))


def test_ops_search_text_impl_supports_single_file_mode(tmp_path: Path) -> None:
    sample_file = tmp_path / "notes.txt"
    _write(sample_file, "needle here\nother line\nneedle twice\n")

    result = ops.search_text_impl(
        tmp_path,
        "needle",
        "notes.txt",
        source_filters=ChatSourceFilters(),
        tool_limits=ChatToolLimits(max_search_matches=1),
        load_readable_content=lambda path: content._LoadedReadableContent(
            read_kind="text",
            status="ok",
            content=path.read_text(encoding="utf-8"),
        ),
    )

    assert result.resolved_path == "notes.txt"
    assert [match.line_number for match in result.matches] == [1]
    assert result.truncated is True


def test_ops_get_file_info_impl_batches_repository_errors(tmp_path: Path) -> None:
    _write(tmp_path / "good.txt", "ok")

    result = ops.get_file_info_impl(
        tmp_path,
        ["good.txt", "missing.txt"],
        session_config=ChatSessionConfig(max_context_tokens=10),
        tool_limits=ChatToolLimits(),
        load_readable_content=lambda path: content._LoadedReadableContent(
            read_kind="text",
            status="ok",
            content=path.read_text(encoding="utf-8"),
        ),
    )

    assert [item.requested_path for item in result.results] == [
        "good.txt",
        "missing.txt",
    ]
    assert result.results[0].status == "ok"
    assert result.results[1].status == "error"
    assert "Requested file does not exist" in (result.results[1].error_message or "")
