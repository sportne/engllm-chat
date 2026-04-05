# Chat Config Reference

This document explains the standalone chat config in plain language.

If you want a quick startup example, see `docs/CHAT_USAGE.md`.
If you want to understand what each setting means and when to change it, start
here.

## Config Shape

The chat config has five top-level sections:

- `llm`
- `source_filters`
- `session`
- `tool_limits`
- `ui`

## Minimal Config

```yaml
llm:
  provider: ollama
  model_name: qwen2.5:7b-instruct
  temperature: 0.1
  api_base_url: http://127.0.0.1:11434
  timeout_seconds: 60.0
  api_key_env_var: null
  prompt_for_api_key_if_missing: true
  verbose_llm_logging: false

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

## `llm`

This section controls which provider you use and how the provider client is
configured.

### `provider`

Allowed values:

- `ollama`
- `mock`
- `openai`
- `xai`
- `anthropic`
- `gemini`

Use:

- `mock` for deterministic local testing
- `ollama` for a local model
- hosted providers for real remote inference

### `model_name`

The provider model to use.

Change this when:

- you want a larger or smaller model
- you want a provider-specific model variant

### `temperature`

Controls how variable the model output is.

Safe default:

- `0.1`

Raise it when:

- you want slightly less rigid output

Lower it when:

- you want more repeatable behavior

### `api_base_url`

Optional explicit provider endpoint.

Use this when:

- you are pointing at Ollama locally
- you need to override a hosted provider default
- you are targeting a custom OpenAI-compatible endpoint

### `timeout_seconds`

How long the provider call may take before timing out.

Raise it when:

- a slower hosted model or local model needs more time

### `api_key_env_var`

Optional override for the API key environment variable name.

Most users should leave this as `null` and use the provider default.

### `prompt_for_api_key_if_missing`

Controls whether the UI is allowed to prompt for a missing API key.

Useful when:

- you want interactive credential entry in the UI

### `verbose_llm_logging`

Turns on detailed request/response logging for the provider adapter.

Useful when:

- debugging malformed structured responses
- checking which tool action the model requested

## `source_filters`

This section limits what the deterministic tool layer will include while
walking the selected root.

### `include`

A list of glob-style patterns to include.

Use this when:

- you want to focus the model on a known subtree or file family

### `exclude`

A list of glob-style patterns to hide from directory/search traversal.

Use this when:

- you want to keep generated files, large vendor trees, or noise out of scope

### `include_hidden`

Whether hidden files and directories should be visible to the tool layer.

Safe default:

- `false`

## `session`

This section controls chat-turn and context budgeting.

### `max_context_tokens`

Maximum estimated active-context tokens kept in memory for a turn.

Raise it when:

- you have a larger model context window
- you want more prior turns retained

Lower it when:

- you want more conservative context use

### `max_tool_round_trips`

Maximum number of model/tool loop iterations in one turn.

Raise it when:

- the model needs deeper investigations

### `max_tool_calls_per_round`

Maximum number of tool calls allowed in one assistant step.

This prevents one model action from exploding into too much work at once.

### `max_total_tool_calls_per_turn`

Maximum total tool calls across the whole turn.

This is the overall tool budget for one user question.

## `tool_limits`

This section controls deterministic filesystem safety limits.

### `max_entries_per_call`

Maximum entries returned by directory-listing and file-finding operations.

### `max_recursive_depth`

Maximum allowed recursive depth for recursive directory operations.

### `max_search_matches`

Maximum number of text-search matches returned by one call.

### `max_read_lines`

Maximum number of lines counted for line-based metadata reporting.

### `max_file_size_characters`

Maximum readable character size for content-based inspection.

This matters more than raw file bytes because the model ultimately consumes
text, not bytes.

### `max_read_file_chars`

Optional full-read limit for `read_file`.

If left `null`, the project derives a default limit from session context size.

### `max_tool_result_chars`

Maximum character size for one tool result payload.

## `ui`

This section controls a small number of Textual UI display toggles.

### `show_token_usage`

Show token estimates in the footer.

### `show_footer_help`

Show footer hints for the interactive UI.

## Practical Config Variants

## Local Ollama

```yaml
llm:
  provider: ollama
  model_name: qwen2.5:14b-instruct-q4_K_M
  api_base_url: http://127.0.0.1:11434
  temperature: 0.1
```

## Hosted Provider

```yaml
llm:
  provider: gemini
  model_name: gemini-2.5-flash
  temperature: 0.1
  api_base_url: null
  api_key_env_var: null
```

## Conservative Debugging Config

```yaml
llm:
  provider: mock
  model_name: mock-chat
  temperature: 0.0
  verbose_llm_logging: true

session:
  max_tool_round_trips: 4
  max_tool_calls_per_round: 2
  max_total_tool_calls_per_turn: 6

tool_limits:
  max_entries_per_call: 100
  max_search_matches: 20
  max_tool_result_chars: 12000
```

## When to Change Settings

Change provider settings when:

- you switch between local and hosted models
- you need a different endpoint or model

Change session settings when:

- the model keeps hitting continuation boundaries
- you want shorter or longer investigations

Change tool limits when:

- the repo is unusually large
- tool outputs are too small to be useful
- you want more conservative filesystem behavior

## Related Docs

- `docs/CHAT_USAGE.md`
- `docs/TESTING_AND_DEBUGGING.md`
- `docs/GLOSSARY.md`
