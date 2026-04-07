# Testing and Debugging

This guide explains how `engllm-chat` is tested and how to debug the most
common failures.

The project is designed so most behavior can be tested without a real network
call. That is one of the main reasons it works as a teaching project: the risky
parts are isolated, and the deterministic parts are easy to inspect directly.

## Testing Strategy

The project uses several layers of verification.

### Deterministic unit tests

These test the code that should behave the same way every time:

- config validation
- domain-model validation
- deterministic filesystem tools
- prompt builders
- workflow helpers
- registry wiring

These tests should not need live providers.

### Mock-provider workflow tests

These exercise the real workflow loop using a deterministic fake provider.

They are useful for checking:

- tool-request handling
- continuation and interruption behavior
- session-context trimming
- final-response validation

### Smoke tests against real providers

These are lightweight end-to-end checks for the real provider path.

Use them when:

- provider serialization/parsing changes
- prompt behavior changes materially
- you want confidence that a real model can still complete a turn

They are not a replacement for deterministic tests. They are a confidence check
on top of them.

### UI and integration checks

The Textual client also has tests, but some UX issues only show up in the full
app. That is why manual integrated checks are still worth doing after major
workflow or UI changes.

## Core Verification Commands

For normal development, run:

```bash
.venv/bin/python -m pytest
.venv/bin/python -m mypy src
.venv/bin/python -m ruff check src tests
```

### When each command helps

- `pytest`
  - catches behavioral regressions
- `mypy src`
  - catches broken type contracts between layers
- `ruff check src tests`
  - catches style and obvious code-quality issues

## Smoke-Test Commands

### Generic smoke test

Run a one-turn real-provider workflow check:

```bash
make smoke-chat
```

This exercises:

- the real provider adapter
- the schema-first workflow
- deterministic tool execution
- final-response validation

### Ollama-focused smoke test

Run the Ollama shortcut:

```bash
make smoke-ollama-chat
```

You can override model or base URL:

```bash
make smoke-ollama-chat \
  OLLAMA_MODEL=qwen2.5-coder:7b-instruct-q4_K_M \
  OLLAMA_BASE_URL=http://127.0.0.1:11434
```

### Packaging smoke test

Build the single-file `.pex` artifact and verify the packaged CLI still starts:

```bash
make smoke-pex
```

This validates:

- the wheel build path
- dependency collection into a local wheelhouse
- `.pex` artifact generation
- `python <artifact>.pex --help`
- `python <artifact>.pex probe-openai-api --help`

The `.pex` artifact bundles dependencies but still relies on an installed
Python runtime, and this build path is intentionally current-platform-first.
GitHub Actions also runs this packaging path so tagged releases can publish the
resulting `.pex` asset.

### Direct module usage with verbose logging

Use the smoke runner directly when you want detailed request/response logging:

```bash
.venv/bin/python -m engllm_chat.smoke_chat \
  --model qwen2.5-coder:7b-instruct-q4_K_M \
  --base-url http://127.0.0.1:11434/v1 \
  --directory . \
  --require-tool-call \
  --verbose-llm
```

For a hosted provider:

```bash
.venv/bin/python -m engllm_chat.smoke_chat \
  --model gemini-2.5-flash \
  --base-url https://generativelanguage.googleapis.com/v1beta/openai/ \
  --directory . \
  --require-tool-call \
  --verbose-llm
```

## How to Debug Common Problems

## Schema-invalid provider responses

Symptoms:

- retries happen repeatedly
- the provider eventually fails with a structured-response error
- the model appears to answer in prose instead of the required envelope

What to check:

- prompt wording in `prompts/chat/`
- schema-envelope construction in `llm/_openai_compatible/serialization.py`
- parsing fallback behavior in `llm/_openai_compatible/parsing.py`
- whether the provider actually supports the structured-output path you expect

Best debugging tool:

- rerun the smoke test with `--verbose-llm`

## Model answers without grounding

Symptoms:

- the model returns a final answer without using tools
- implementation questions are answered from prior knowledge instead of repo evidence

What to check:

- prompt guidance for tool use
- whether the question strongly suggests code inspection
- whether the tool catalog descriptions make the right tool obvious

What to do:

- run the smoke test with `--require-tool-call`
- inspect the first model action in verbose logs

## Tool-selection problems

Symptoms:

- the model uses `find_files` when it should use `search_text`
- the model reads files too early without checking metadata
- the model keeps exploring docs instead of code

What to check:

- tool descriptions in `tools/chat/registry.py`
- prompt guidance in `prompts/chat/`
- deterministic tool behavior if the tool output is too weak to guide the next step

## Continuation results

Symptoms:

- the turn ends with `needs_continuation`
- the model hit tool-round or tool-budget limits

What it means:

- the workflow behaved as designed
- the turn was stopped at a safety boundary rather than failing unexpectedly

What to check:

- session limits in config
- whether the model is taking too many low-value exploratory steps
- whether prompt guidance should encourage a more direct evidence-gathering path

## Provider connectivity issues

### Ollama not reachable

Check:

- Ollama is running
- the configured base URL matches the server
- the endpoint includes the expected `/v1` behavior when using the shared adapter path

### Hosted-provider auth failures

Check:

- the API key env var is set
- the provider/model name is correct
- the configured base URL matches the intended hosted endpoint

## What to Run After Different Kinds of Changes

### Prompt-only change

Run:

- `pytest`
- at least one smoke test against the provider you care about most

### Deterministic tool change

Run:

- `pytest`
- `mypy src`
- `ruff check src tests`

### Provider adapter change

Run:

- `pytest`
- `mypy src`
- `ruff check src tests`
- a smoke test with `--verbose-llm`

### UI-only change

Run:

- `pytest`
- targeted manual app exercise if the visible interaction changed

## Teaching Takeaway

The project’s testing strategy is part of the architecture, not an afterthought.

It teaches a useful pattern:

- keep most behavior deterministic
- test it directly
- keep the real-provider checks narrow and repeatable

That balance is one of the main reasons the project works as a reference
implementation rather than only as a demo.
