# EngLLM Chat Usage Guide

This guide covers how to start and use the interactive directory chat feature:

```bash
engllm-chat interactive <directory> --config <path>
```

The chat tool is local-first, directory-scoped, and read-only. The model can
inspect files only through the built-in safe toolset; it cannot execute shell
commands or modify your repository.

## What It Does

`engllm-chat interactive` starts a Textual terminal app over one directory root.
During a session, the model can:

- list files and directories
- search for matching files
- search text inside readable files
- inspect file metadata before reading
- read bounded slices of text or converted markdown

All file access is confined to the directory you pass on the command line.

## Supported Providers

The current chat implementation supports:

- `ollama`
- `mock`
- `openai`
- `xai`
- `anthropic`
- `gemini`

Use `mock` for offline testing and deterministic behavior. Use `ollama` for a
real local model. Use the hosted providers when you want to connect to their
OpenAI-compatible chat endpoints with provider-specific API key defaults.

Hosted-provider defaults:

- `openai`: `OPENAI_API_KEY`, `https://api.openai.com/v1`
- `xai`: `XAI_API_KEY`, `https://api.x.ai/v1`
- `anthropic`: `ANTHROPIC_API_KEY`, `https://api.anthropic.com/v1/`
- `gemini`: `GEMINI_API_KEY`,
  `https://generativelanguage.googleapis.com/v1beta/openai/`

## Setup

From the repository root:

```bash
make setup-venv
make install-dev
```

If you want to use Ollama, make sure your local Ollama server is running first.
The default base URL is `http://127.0.0.1:11434`.

If you want to use a hosted provider, export the matching API key before you
launch the app:

```bash
export OPENAI_API_KEY=...
export XAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
```

## Minimal Chat Config

Create a standalone chat config file such as `chat.yaml`:

```yaml
llm:
  provider: ollama
  model_name: qwen2.5:7b-instruct
  temperature: 0.1
  ollama_base_url: http://127.0.0.1:11434
  api_base_url: null
  timeout_seconds: 60.0
  api_key_env_var: null
  prompt_for_api_key_if_missing: true

source_filters:
  include: []
  exclude: []
  include_hidden: false

session:
  max_context_tokens: 24000
  max_tool_round_trips: 8
  max_tool_calls_per_round: 4
  max_total_tool_calls_per_turn: 12

tool_limits:
  max_entries_per_call: 200
  max_recursive_depth: 12
  max_search_matches: 50
  max_read_lines: 200
  max_file_size_characters: 262144
  max_read_file_chars: null
  max_tool_result_chars: 24000

ui:
  show_token_usage: true
  show_footer_help: true
```

## Starting A Session

Run chat against the directory you want to inspect:

```bash
.venv/bin/python -m engllm_chat interactive . --config chat.yaml
```

Or, after installation:

```bash
engllm-chat interactive . --config chat.yaml
```

You can point the command at any directory:

```bash
engllm-chat interactive src --config chat.yaml
engllm-chat interactive /path/to/repo/subtree --config chat.yaml
```

## Common Runtime Overrides

CLI overrides take precedence over config values. The most useful overrides are:

```bash
engllm-chat interactive . --config chat.yaml \
  --provider ollama \
  --model qwen2.5:14b-instruct-q4_K_M \
  --temperature 0.2 \
  --ollama-base-url http://127.0.0.1:11434
```

Hosted providers can override the OpenAI-compatible endpoint explicitly:

```bash
engllm-chat interactive . --config chat.yaml \
  --provider xai \
  --model grok-4-fast-reasoning \
  --api-base-url https://api.x.ai/v1
```

You can also override session and tool limits at runtime:

```bash
engllm-chat interactive . --config chat.yaml \
  --max-context-tokens 32000 \
  --max-tool-round-trips 10 \
  --max-tool-calls-per-round 4 \
  --max-total-tool-calls-per-turn 12 \
  --max-entries-per-call 300 \
  --max-recursive-depth 16 \
  --max-search-matches 80 \
  --max-read-lines 300 \
  --max-file-size-characters 400000 \
  --max-tool-result-chars 30000
```

## Using The Textual UI

The chat UI has four main regions:

- a scrollable transcript
- a multiline composer
- a transient status row
- a persistent footer row

Key behavior:

- `Enter` sends the current draft
- `Shift+Enter` inserts a newline
- `quit` or `exit` closes the session
- `/help` shows a short in-app help message

While a turn is running:

- the footer shows active token/help hints
- the `Stop` button becomes available
- sending another message opens interrupt confirmation

If you interrupt a running answer, the partial assistant output remains visible
and is marked as interrupted.

## Practical Usage Patterns

Good questions:

- `Where is the chat workflow implemented?`
- `How does read_file handle large files?`
- `Which tests cover the Ollama streaming adapter?`
- `What changed between the streaming workflow and the non-streaming one?`

For large files, the model will usually:

1. inspect the file with `get_file_info`
2. decide whether the file is small enough to read fully
3. read only a character range when the file is too large

That means it is normal for the model to explore in a few steps before it gives
you a final answer.

## Mock Provider Example

Use the mock provider to test the UI and workflow without a live model:

```bash
engllm-chat interactive . --config chat.yaml --provider mock --model mock-chat
```

This is useful for local smoke testing, UI checks, and deterministic debugging.

## Ollama Example

Use a local Ollama model:

```bash
engllm-chat interactive . --config chat.yaml \
  --provider ollama \
  --model qwen2.5:14b-instruct-q4_K_M
```

If you use a different Ollama endpoint:

```bash
engllm-chat interactive . --config chat.yaml \
  --provider ollama \
  --model qwen2.5:14b-instruct-q4_K_M \
  --ollama-base-url http://localhost:11434
```

## Hosted Provider Examples

OpenAI:

```bash
engllm-chat interactive . --config chat.yaml \
  --provider openai \
  --model gpt-5-mini
```

xAI:

```bash
engllm-chat interactive . --config chat.yaml \
  --provider xai \
  --model grok-4-fast-reasoning
```

Anthropic:

```bash
engllm-chat interactive . --config chat.yaml \
  --provider anthropic \
  --model claude-sonnet-4-5
```

Gemini:

```bash
engllm-chat interactive . --config chat.yaml \
  --provider gemini \
  --model gemini-2.5-flash
```

## Token And Context Behavior

The footer shows:

- session token estimate
- active-context token estimate
- answer confidence when available

When the active context grows too large, older turns can be removed from active
context while still remaining in visible transcript history. When that happens,
the app surfaces a warning in the transcript.

If the model needs too many tool rounds or tool calls to finish one turn, the
workflow pauses cleanly and asks for continuation instead of running forever.

## Troubleshooting

### `Cannot connect to Ollama`

Check that:

- Ollama is running
- the configured or overridden `ollama_base_url` is correct
- the selected model exists locally

### Hosted provider authentication fails

Check that:

- the matching API key environment variable is set
- the selected model name belongs to that provider
- the configured or overridden `api_base_url` is correct when using a proxy or
  compatibility gateway

### The model stops before answering

This usually means one of the configured safety limits was reached:

- `max_tool_round_trips`
- `max_tool_calls_per_round`
- `max_total_tool_calls_per_turn`
- `max_context_tokens`

Increase the relevant limit if you want the agent to explore more before it
must stop.

### The model cannot read a large file

This is expected when a file exceeds:

- `max_file_size_characters` for the hard readable-size ceiling
- `max_read_file_chars` for the full-read budget, when configured
- `max_tool_result_chars` for returned content size

In those cases, ask narrower questions or let the model read only a character
range from the file.

### Hidden files are not being considered

Set:

```yaml
source_filters:
  include_hidden: true
```

### The app starts but asks for a credential

This depends on provider config. The value entered in the startup modal is
session-only and is not written back to config or disk.

## Related Docs

- `docs/CHAT_SPEC.md`
- `docs/CHAT_PLAN.md`
- `docs/USAGE.md`
