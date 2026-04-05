# Deterministic Tools in `engllm-chat`

This document explains the read-only tool layer used by `engllm-chat`.

If you are studying this project as a reference implementation, this is one of
the most important parts of the architecture. The deterministic tools are what
let the model explore a codebase without giving it direct shell access or
arbitrary file access.

## What “Deterministic” Means Here

In this project, a deterministic tool is a function whose behavior is defined
by normal application code rather than by the model.

That means:

- the tool has a fixed input shape
- the tool has a fixed output shape
- the same input produces the same result, assuming the underlying files are
  unchanged
- the tool does not improvise or “interpret” the request the way a model would
- the tool can be tested directly without calling a provider

The model decides **which** tool to request. The application decides **how**
that tool actually behaves.

## Why This Layer Exists

Without a deterministic tool layer, a project like this would usually drift
toward unsafe or hard-to-test patterns, such as:

- letting the model inspect the filesystem directly
- letting the model compose shell commands
- mixing provider behavior with local file access rules
- returning loosely shaped strings instead of typed results

This project avoids that by keeping the filesystem rules inside normal Python
code and exposing them through a small safe interface.

## The Current Tool Catalog

The model can currently request these tools:

- `list_directory(path=".")`
- `list_directory_recursive(path=".", max_depth=None)`
- `find_files(pattern, path=".")`
- `search_text(query, path=".")`
- `get_file_info(path=...)` or `get_file_info(paths=[...])`
- `read_file(path, start_char=None, end_char=None)`

Each tool has:

- typed arguments
- deterministic implementation code
- a typed result model

The registry in `tools/chat/registry.py` is what connects the model-facing tool
definition to the real implementation in `core/chat/`.

## The Main Safety Rules

These tools are intentionally constrained. The important rules are:

### Root confinement

All tool paths are interpreted relative to the configured root directory.

That means:

- the model cannot ask for arbitrary absolute paths
- `..` escapes are rejected if they leave the configured root
- the selected root is the boundary for all filesystem access

This is one of the main reasons the tool layer is safe enough for local use.

### Blank paths are invalid

The tools do not treat blank strings as “probably the root”. Blank paths are
rejected.

If the model wants to operate on the whole repo, it should use:

```text
path="."
```

### Direct symlink targets are rejected

The tools do not allow the model to directly target a symlinked file or
directory. This helps avoid surprising filesystem traversal behavior.

### Read-only behavior

The tool layer never mutates files. It only:

- lists
- searches
- inspects
- reads

There are no write, edit, delete, or shell-execution tools in this system.

### Bounded outputs

The tools enforce limits such as:

- maximum entries per call
- maximum recursive depth
- maximum search matches
- maximum readable file size
- maximum tool result size
- optional maximum full-read size before a ranged read is required

These limits help keep the workflow safe, predictable, and small enough to fit
within the chat context window.

## How Readable Content Works

The project treats readable content carefully instead of assuming every file can
be decoded as plain text.

At a high level, the tool layer tries these steps:

1. try to read the file as UTF-8 text
2. reject obviously non-text content such as content containing null bytes
3. if the file type is supported for conversion, use `markitdown`
4. cache converted markdown text for reuse
5. apply the same character-based rules to the resulting readable text

This matters because many repository questions involve documents as well as code
files.

## Why Character Limits Matter More Than Raw Bytes

The tool layer is designed for what the model will actually read, which is
text. For that reason, character-based limits are often more important than raw
byte size.

For example:

- a PDF might be large on disk, but the relevant converted markdown might still
  be small enough to inspect
- a text file might be valid UTF-8 but still be too large to send back as one
  full read

So the system asks questions like:

- can this content be turned into readable text?
- how many characters would the readable form contain?
- is it small enough to search or read safely?

## When To Use Each Tool

The model gets better results when it uses the right tool for the job.

### `find_files`

Use `find_files` when you are trying to locate files by name or path pattern.

Examples:

- find Python files under `src`
- find files named `workflow.py`
- find test files that match `test_*.py`

Good mental model:

- `find_files` is for filenames and paths
- it is **not** for searching file contents

### `search_text`

Use `search_text` when you need to look inside readable file contents.

Examples:

- search for a class name
- search for a function name
- search for a config key or literal string

Good mental model:

- `search_text` is for content
- it can target either a whole directory tree or one specific file

### `get_file_info`

Use `get_file_info` before reading when you need metadata about one or more
files.

It is useful for questions like:

- is this file readable as text?
- how large is the readable content?
- would a full read be allowed?
- should the next step be a ranged read instead of a full read?

It also supports batching, which makes it useful when comparing multiple files.

### `read_file`

Use `read_file` when you actually need the text itself.

This tool:

- reads one file only
- supports character ranges
- enforces readable-content limits and result-size limits

Good mental model:

- `get_file_info` helps you decide **whether** and **how** to read
- `read_file` actually returns the content

### `list_directory` and `list_directory_recursive`

Use these when you need structure rather than content.

Examples:

- understand the top-level layout
- inspect a subtree before choosing files to read
- find likely directories to search next

`list_directory` is the shallow view.

`list_directory_recursive` is the deeper tree walk with bounded recursion.

## A Typical Investigation Pattern

In practice, implementation questions often work best as a sequence like this:

1. `list_directory` or `find_files` to locate candidate files
2. `search_text` to find the specific symbol or phrase
3. `get_file_info` if the file might be large or unreadable
4. `read_file` to inspect the exact content

This is a big part of why the tool layer is useful as a reference
implementation: the workflow encourages evidence gathering in small, typed,
reviewable steps.

## Where This Lives in the Code

The important code locations are:

- `src/engllm_chat/tools/chat/registry.py`
  - the bridge between model tool requests and deterministic execution
- `src/engllm_chat/core/chat/listing.py`
  - the public façade for the deterministic filesystem tools
- `src/engllm_chat/core/chat/_listing/`
  - the internal implementation split for path handling, readable content, and
    tool operations
- `src/engllm_chat/core/chat/models.py`
  - typed result models for the deterministic tools
- `src/engllm_chat/tools/chat/models.py`
  - typed argument models for the chat tool interface

## What This Project Is Trying To Demonstrate

As a reference implementation, this tool layer demonstrates a few useful
patterns:

- keep the model away from direct filesystem or shell access
- make tools small, typed, and testable
- keep safety rules in normal application code
- separate tool metadata from tool implementation
- return typed results that can be fed back into the workflow consistently

If you are building a similar system, this is the pattern to copy before you
start thinking about richer agents or more powerful tools.

## Suggested Next Reading

The best companion docs to this one are:

- `docs/ARCHITECTURE.md`
- `docs/LLM_STRUCTURED_CALLS.md`
- `docs/CONTRIBUTING_CHAT_ARCHITECTURE.md`
- `docs/CHAT_USAGE.md`

Those cover the larger system map, the provider interaction pattern, and the
operator-facing usage layer around these tools, plus where to make changes
safely when extending the system.
