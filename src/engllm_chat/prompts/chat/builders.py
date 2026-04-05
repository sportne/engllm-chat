"""Deterministic prompt builders for the interactive chat tool."""

from __future__ import annotations

from pydantic import BaseModel

from engllm_chat.domain.models import ChatFinalResponse, ChatToolLimits
from engllm_chat.prompts.chat.templates import CHAT_SYSTEM_PROMPT_PREAMBLE

_TOOL_DESCRIPTIONS: tuple[tuple[str, str], ...] = (
    ("list_directory", "List immediate children of one directory."),
    (
        "list_directory_recursive",
        "List one directory subtree as a flat depth-first result.",
    ),
    ("find_files", "Find files by root-relative glob pattern."),
    (
        "search_text",
        "Search readable file content for a literal substring in one directory tree "
        "or one file.",
    ),
    (
        "get_file_info",
        "Inspect one file or a small batch of files before deciding whether or how "
        "to read them.",
    ),
    (
        "read_file",
        "Read text or converted markdown content, optionally using "
        "start_char/end_char.",
    ),
)

_FIELD_GUIDANCE: dict[str, str] = {
    "answer": "Main grounded answer. Keep it concise and avoid unsupported claims.",
    "citations": (
        "Support material claims with file paths and line ranges when available."
    ),
    "confidence": (
        "Optional 0.0-1.0 confidence based on evidence quality and completeness."
    ),
    "uncertainty": (
        "List caveats, ambiguity, or places where the evidence is incomplete."
    ),
    "missing_information": (
        "List explicit gaps or TBD items that blocked a stronger answer."
    ),
    "follow_up_suggestions": (
        "Suggest useful next questions or tool-driven follow-up steps."
    ),
}

_DEFAULT_FIELD_GUIDANCE = "Return a valid value that matches the response schema."


def build_chat_system_prompt(
    *,
    tool_limits: ChatToolLimits,
    response_model: type[BaseModel] = ChatFinalResponse,
) -> str:
    """Return the interactive chat system prompt."""

    tool_catalog = "\n".join(
        f"- {tool_name}: {description}" for tool_name, description in _TOOL_DESCRIPTIONS
    )
    usage_examples = "\n".join(
        (
            (
                "- To search the entire repo by file contents: "
                "search_text(path='.', query='provider')"
            ),
            (
                "- To locate candidate files first: "
                "find_files(path='src', pattern='**/*.py')"
            ),
            (
                "- To inspect a directory tree before reading: "
                "list_directory_recursive(path='src')"
            ),
            (
                "- To find specific evidence in text: "
                "search_text(path='src', query='MySymbol')"
            ),
            (
                "- To search one file directly: "
                "search_text(path='src/app.py', query='MySymbol')"
            ),
            (
                "- To inspect a possibly large file first: "
                "get_file_info(path='src/app.py')"
            ),
            (
                "- To compare several candidate files first: "
                "get_file_info(paths=['src/app.py', 'src/config.py'])"
            ),
            (
                "- To read only part of a file: "
                "read_file(path='src/app.py', start_char=0, end_char=4000)"
            ),
        )
    )
    visible_limits = "\n".join(
        (
            (
                "- search_text returns at most "
                f"{tool_limits.max_search_matches} matches per call."
            ),
            (
                "- read-only file tools reject readable content over "
                f"{tool_limits.max_file_size_characters} characters."
            ),
            (
                "- full-file reads are further limited by "
                "max_read_file_chars when it is set; "
                "otherwise orchestration may derive a limit from the "
                "session context window."
            ),
            (
                "- any single tool result may be truncated near "
                f"{tool_limits.max_tool_result_chars} characters."
            ),
        )
    )
    response_fields = "\n".join(
        f"- {field_name}: "
        f"{_FIELD_GUIDANCE.get(field_name, _DEFAULT_FIELD_GUIDANCE)}"
        for field_name in response_model.model_fields
    )

    return (
        f"{CHAT_SYSTEM_PROMPT_PREAMBLE}\n\n"
        "Available tools:\n"
        f"{tool_catalog}\n\n"
        "Required action format:\n"
        "- On every turn, return exactly one structured action.\n"
        "- If you need more evidence, return action.kind='tool_request' with one "
        "tool_name and one arguments object.\n"
        "- If you can answer, return action.kind='final_response' with response "
        "matching the final-answer schema.\n"
        "- Request at most one tool per turn.\n"
        "- Tool results will be supplied back to you as plain conversation "
        "messages; use them before deciding the next action.\n\n"
        "Tool usage examples:\n"
        f"{usage_examples}\n\n"
        "Operational rules:\n"
        "- path must never be blank. Use path='.' to operate on the entire "
        "configured root.\n"
        "- list_directory, list_directory_recursive, and find_files expect "
        "directory paths.\n"
        "- search_text accepts either a directory path or a single file path.\n"
        "- get_file_info accepts either one file path or a list of file paths.\n"
        "- read_file expects one file path.\n"
        "- Use find_files for matching file paths or file names.\n"
        "- Use search_text for searching inside file contents.\n"
        "- Prefer find_files or search_text before broad file reads.\n"
        "- For questions about code behavior or provider support, usually start "
        "with search_text(path='.', query='...') or a targeted "
        "list_directory_recursive(path='src').\n"
        "- Use get_file_info before read_file when a file may be large.\n"
        "- Use start_char and end_char for partial reads when a full file is "
        "unnecessary.\n"
        "- If a tool call fails because of its arguments or target path, do not "
        "repeat the same failing call. Correct the arguments or choose a better "
        "tool.\n"
        "- Answer conservatively and cite the evidence you actually have.\n\n"
        "Relevant limits:\n"
        f"{visible_limits}\n\n"
        "Final response fields:\n"
        f"{response_fields}\n"
    )
