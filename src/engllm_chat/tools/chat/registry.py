"""Internal registry and dispatch helpers for chat tools."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError as PydanticValidationError

from engllm_chat.core.chat import (
    find_files,
    get_file_info,
    list_directory,
    list_directory_recursive,
    read_file,
    search_text,
)
from engllm_chat.domain.models import (
    ChatConfig,
    ChatToolCall,
    ChatToolResult,
    DomainModel,
)
from engllm_chat.llm.base import ChatToolDefinition
from engllm_chat.tools.chat.models import (
    FindFilesArgs,
    GetFileInfoArgs,
    ListDirectoryArgs,
    ListDirectoryRecursiveArgs,
    ReadFileArgs,
    SearchTextArgs,
)


@dataclass(frozen=True, slots=True)
class ChatToolSpec:
    """One orchestrated chat tool definition and executor."""

    name: str
    description: str
    argument_model: type[DomainModel]
    runner: Callable[[DomainModel, Path, ChatConfig], DomainModel]


def _run_list_directory(
    args: DomainModel,
    root_path: Path,
    config: ChatConfig,
) -> DomainModel:
    typed_args = ListDirectoryArgs.model_validate(args.model_dump())
    return list_directory(
        root_path,
        typed_args.path,
        source_filters=config.source_filters,
        tool_limits=config.tool_limits,
    )


def _run_list_directory_recursive(
    args: DomainModel,
    root_path: Path,
    config: ChatConfig,
) -> DomainModel:
    typed_args = ListDirectoryRecursiveArgs.model_validate(args.model_dump())
    return list_directory_recursive(
        root_path,
        typed_args.path,
        source_filters=config.source_filters,
        tool_limits=config.tool_limits,
        max_depth=typed_args.max_depth,
    )


def _run_find_files(
    args: DomainModel,
    root_path: Path,
    config: ChatConfig,
) -> DomainModel:
    typed_args = FindFilesArgs.model_validate(args.model_dump())
    return find_files(
        root_path,
        typed_args.pattern,
        typed_args.path,
        source_filters=config.source_filters,
        tool_limits=config.tool_limits,
    )


def _run_search_text(
    args: DomainModel,
    root_path: Path,
    config: ChatConfig,
) -> DomainModel:
    typed_args = SearchTextArgs.model_validate(args.model_dump())
    return search_text(
        root_path,
        typed_args.query,
        typed_args.path,
        source_filters=config.source_filters,
        tool_limits=config.tool_limits,
    )


def _run_get_file_info(
    args: DomainModel,
    root_path: Path,
    config: ChatConfig,
) -> DomainModel:
    typed_args = GetFileInfoArgs.model_validate(args.model_dump())
    return get_file_info(
        root_path,
        typed_args.path if typed_args.path is not None else typed_args.paths or [],
        session_config=config.session,
        tool_limits=config.tool_limits,
    )


def _run_read_file(
    args: DomainModel,
    root_path: Path,
    config: ChatConfig,
) -> DomainModel:
    typed_args = ReadFileArgs.model_validate(args.model_dump())
    return read_file(
        root_path,
        typed_args.path,
        session_config=config.session,
        tool_limits=config.tool_limits,
        start_char=typed_args.start_char,
        end_char=typed_args.end_char,
    )


_CHAT_TOOL_SPECS: tuple[ChatToolSpec, ...] = (
    ChatToolSpec(
        name="list_directory",
        description="List immediate children of one directory.",
        argument_model=ListDirectoryArgs,
        runner=_run_list_directory,
    ),
    ChatToolSpec(
        name="list_directory_recursive",
        description="List one directory subtree as a flat depth-first result.",
        argument_model=ListDirectoryRecursiveArgs,
        runner=_run_list_directory_recursive,
    ),
    ChatToolSpec(
        name="find_files",
        description="Find files by root-relative glob pattern.",
        argument_model=FindFilesArgs,
        runner=_run_find_files,
    ),
    ChatToolSpec(
        name="search_text",
        description="Search readable file content for a literal substring.",
        argument_model=SearchTextArgs,
        runner=_run_search_text,
    ),
    ChatToolSpec(
        name="get_file_info",
        description=(
            "Inspect one file or a small batch of files before deciding whether "
            "or how to read them."
        ),
        argument_model=GetFileInfoArgs,
        runner=_run_get_file_info,
    ),
    ChatToolSpec(
        name="read_file",
        description=(
            "Read text or converted markdown content, optionally using "
            "start_char/end_char."
        ),
        argument_model=ReadFileArgs,
        runner=_run_read_file,
    ),
)

_CHAT_TOOL_SPEC_BY_NAME = {spec.name: spec for spec in _CHAT_TOOL_SPECS}


def get_chat_tool_specs() -> tuple[ChatToolSpec, ...]:
    """Return the ordered internal chat-tool registry."""

    return _CHAT_TOOL_SPECS


def get_chat_tool_spec(name: str) -> ChatToolSpec | None:
    """Return one chat-tool spec by name when present."""

    return _CHAT_TOOL_SPEC_BY_NAME.get(name)


def build_chat_tool_definitions() -> list[ChatToolDefinition]:
    """Return provider-facing tool definitions derived from the registry."""

    return [
        ChatToolDefinition(
            name=spec.name,
            description=spec.description,
            input_schema=spec.argument_model.model_json_schema(),
            argument_model=spec.argument_model,
        )
        for spec in _CHAT_TOOL_SPECS
    ]


def execute_chat_tool_call(
    tool_call: ChatToolCall,
    *,
    root_path: Path,
    config: ChatConfig,
) -> ChatToolResult:
    """Execute one tool call against the internal chat-tool registry."""

    spec = get_chat_tool_spec(tool_call.tool_name)
    if spec is None:
        return ChatToolResult(
            call_id=tool_call.call_id,
            tool_name=tool_call.tool_name,
            status="error",
            payload={},
            error_message=f"Unknown chat tool '{tool_call.tool_name}'",
        )

    try:
        arguments = spec.argument_model.model_validate(tool_call.arguments)
        payload_model = spec.runner(arguments, root_path, config)
        return ChatToolResult(
            call_id=tool_call.call_id,
            tool_name=tool_call.tool_name,
            status="ok",
            payload=payload_model.model_dump(mode="json"),
        )
    except (PydanticValidationError, ValueError, TypeError, Exception) as exc:
        return ChatToolResult(
            call_id=tool_call.call_id,
            tool_name=tool_call.tool_name,
            status="error",
            payload={},
            error_message=str(exc),
        )
