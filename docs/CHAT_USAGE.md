# EngLLM Chat Usage Guide

This guide covers how to start and use the interactive directory chat feature:

```bash
engllm-chat <directory> --config <path>
```

The chat tool is local-first, directory-scoped, and read-only. The model can
inspect files only through the built-in safe toolset; it cannot execute shell
commands or modify your repository.

## What It Does

`engllm-chat` starts a Textual terminal app over one directory root.
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

## Which Provider Should I Use?

Use:

- `mock` when you want deterministic local testing with no network access
- `ollama` when you want a real local model and easy experimentation
- hosted providers when you want stronger remote models or need to compare
  behavior across provider ecosystems

For most development work:

- start with `mock` for deterministic tests
- use `ollama` for local real-provider workflow checks
- use a hosted provider when you specifically want to verify hosted behavior

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
  api_base_url: http://127.0.0.1:11434
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
.venv/bin/python -m engllm_chat . --config chat.yaml
```

Or, after installation:

```bash
engllm-chat . --config chat.yaml
```

If you want to ship the app as one file, build the `.pex` artifact and run it
with Python:

```bash
make package-pex
python dist/engllm-chat-0.2.0-py311-linux_x86_64.pex . --config chat.yaml
```

That artifact includes the project’s Python dependencies but still relies on an
installed Python runtime. The current build path is intended for the platform
you build on.

You can point the command at any directory:

```bash
engllm-chat src --config chat.yaml
engllm-chat /path/to/repo/subtree --config chat.yaml
```

## Common Runtime Overrides

CLI overrides take precedence over config values. The most useful overrides are:

```bash
engllm-chat . --config chat.yaml \
  --provider ollama \
  --model qwen2.5:14b-instruct-q4_K_M \
  --temperature 0.2 \
  --api-base-url http://127.0.0.1:11434
```

Hosted providers can override the OpenAI-compatible endpoint explicitly:

```bash
engllm-chat . --config chat.yaml \
  --provider xai \
  --model grok-4-fast-reasoning \
  --api-base-url https://api.x.ai/v1
```

You can also override session and tool limits at runtime:

```bash
engllm-chat . --config chat.yaml \
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

## When To Change Session Or Tool Limits

Change session limits when:

- the model keeps returning continuation results because it ran out of tool
  rounds or total tool budget
- you want more or fewer prior turns kept in active context

Change tool limits when:

- directory listings are too shallow or too truncated to be useful
- text searches return too few matches
- large-file handling is too conservative for the repo you are exploring

Keep in mind that raising limits can make turns slower and can increase the
amount of context the model has to reason over.

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
engllm-chat . --config chat.yaml --provider mock --model mock-chat
```

This is useful for local smoke testing, UI checks, and deterministic debugging.

## Ollama Example

Use a local Ollama model:

```bash
engllm-chat . --config chat.yaml \
  --provider ollama \
  --model qwen2.5:14b-instruct-q4_K_M
```

If you use a different Ollama endpoint:

```bash
engllm-chat . --config chat.yaml \
  --provider ollama \
  --model qwen2.5:14b-instruct-q4_K_M \
  --api-base-url http://localhost:11434
```

## Repeatable Chat Smoke Test

For a repeatable one-turn workflow check against a configured provider, run:

```bash
make smoke-chat
```

That command:

- uses the real shared provider adapter
- runs one chat turn through the real schema-first workflow
- fails if the model does not complete the turn
- fails if the model answers without making at least one tool call

For Ollama specifically, this remains available:

```bash
make smoke-ollama-chat \
  OLLAMA_MODEL=qwen2.5-coder:7b-instruct-q4_K_M \
  OLLAMA_BASE_URL=http://127.0.0.1:11434
```

To point the generic target at Gemini:

```bash
export GEMINI_API_KEY=...
make smoke-chat SMOKE_PROVIDER=gemini SMOKE_MODEL=gemini-2.5-flash
```

If you want the full structured summary, you can also run the module directly.

Ollama example:

```bash
.venv/bin/python -m engllm_chat.smoke_chat \
  --provider ollama \
  --directory . \
  --require-tool-call \
  --json
```

Gemini example:

```bash
export GEMINI_API_KEY=...
.venv/bin/python -m engllm_chat.smoke_chat \
  --provider gemini \
  --model gemini-2.5-flash \
  --directory . \
  --require-tool-call \
  --verbose-llm
```

### Packaging smoke test

Build the single-file `.pex` artifact and verify the packaged CLI still starts:

```bash
make smoke-pex
```

This checks:

- the wheel build path
- dependency collection into a local wheelhouse
- `.pex` artifact generation
- `python <artifact>.pex --help`
- `python <artifact>.pex probe-openai-api --help`

## Hosted Provider Examples

OpenAI:

```bash
engllm-chat . --config chat.yaml \
  --provider openai \
  --model gpt-5-mini
```

xAI:

```bash
engllm-chat . --config chat.yaml \
  --provider xai \
  --model grok-4-fast-reasoning
```

Anthropic:

```bash
engllm-chat . --config chat.yaml \
  --provider anthropic \
  --model claude-sonnet-4-5
```

Gemini:

```bash
engllm-chat . --config chat.yaml \
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

## Missing API keys

If a hosted provider fails before startup or during the first request:

- check that the expected environment variable is set
- check `api_key_env_var` in config if you overrode the default
- confirm the selected provider matches the key you exported

## Ollama not reachable

If Ollama requests fail:

- make sure the Ollama server is running
- confirm the base URL matches the local server
- make sure the selected model exists locally
- if you override the URL, make sure it points at the shared `/v1`-compatible path

## Model answers without grounding

If the model answers implementation questions without tool use:

- rerun a smoke test with `--require-tool-call`
- enable `--verbose-llm` to inspect the first returned action
- check whether the question is specific enough to strongly suggest code evidence

## Tool-budget continuation results

If a turn ends with `needs_continuation`:

- the workflow hit a configured safety boundary
- check the session tool-round and total tool-call limits
- decide whether the limits are too low or the model is taking too many low-value steps

## Large files are not fully readable

This is expected when a file exceeds:

- `max_file_size_characters` for the hard readable-size ceiling
- `max_read_file_chars` for the full-read budget, when configured
- `max_tool_result_chars` for returned content size

In those cases, ask narrower questions or let the model read only a character
range from the file.

## Hidden files are not being considered

Set:

```yaml
source_filters:
  include_hidden: true
```

## The app starts but asks for a credential

This depends on provider config. The value entered in the startup modal is
session-only and is not written back to config or disk.

## Related Reference Docs

- `docs/CHAT_CONFIG_REFERENCE.md`
- `docs/TESTING_AND_DEBUGGING.md`
- `docs/ARCHITECTURE.md`
- `docs/CHAT_SPEC.md`
