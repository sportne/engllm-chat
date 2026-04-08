# Directory Chat Tool Specification

This document is the detailed product and safety specification that guided the
feature work. It is still useful as a reference, but it is not the best place
to start if you are trying to understand the current codebase.

For current implementation guidance, start with:

- `docs/ARCHITECTURE.md`
- `docs/LLM_STRUCTURED_CALLS.md`
- `docs/DETERMINISTIC_TOOLS.md`
- `docs/CONTRIBUTING_CHAT_ARCHITECTURE.md`

## 1. Overview

`engllm-chat` is a new local-first command-line tool for interactive question
answering over the contents of a directory.

The user starts a chat session by specifying a directory root. During the
session, the LLM may use a fixed set of safe, read-only tools to explore the
directory and gather evidence before answering.

The first release is intentionally conservative:

- provider support: `ollama`, `mock`, `openai`, `xai`, `anthropic`, and `gemini`
- no persistent conversations across launches
- no destructive tools
- no arbitrary shell execution
- JSON-based model interaction throughout
- terminal rendering handled entirely by the application

This tool is distinct from the current `ask` workflows. It must not depend on
existing workflow code under other `src/engllm_chat/tools/*` namespaces.

---

## 2. Goals

The tool must:

- provide an interactive terminal chat experience over a chosen directory
- allow the model to inspect directory contents at runtime through explicit
  read-only tools
- keep deterministic behavior outside the LLM boundary
- keep responses reviewable with file/line citations where possible
- remain safe for non-expert users by limiting model powers to a known,
  enumerated toolset
- support low-friction local use through Ollama
- support simple hosted-provider setup through OpenAI-compatible endpoints

---

## 3. Non-Goals For Initial Release

The initial version does not need to support:

- persistent chat history across launches
- context compaction or summarization
- arbitrary shell commands
- file mutation, patching, or any destructive actions
- web UI or GUI
- reuse of existing `tools/ask`, `tools/sdd`, or other workflow modules

---

## 4. Public Interface

## 4.1 CLI

Initial command:

```bash
engllm-chat interactive <directory> --config <path>
```

This phase locks one initial public command only:

- `engllm-chat interactive <directory> --config <path>`

Required CLI inputs:

- positional: `<directory>`
- required option: `--config <path>`

Supported runtime overrides for v1:

- provider/model:
  - `--provider {ollama,mock,openai,xai,anthropic,gemini}`
  - `--model <name>`
  - `--temperature <float>`
  - `--api-base-url <url>`
- session limits:
  - `--max-context-tokens <int>`
  - `--max-tool-round-trips <int>`
  - `--max-tool-calls-per-round <int>`
  - `--max-total-tool-calls-per-turn <int>`
- tool-safety limits:
  - `--max-entries-per-call <int>`
  - `--max-recursive-depth <int>`
  - `--max-search-matches <int>`
  - `--max-read-lines <int>`
  - `--max-file-size-characters <int>`
  - `--max-tool-result-chars <int>`

Config precedence:

- the standalone chat config supplies defaults
- CLI flags override those defaults at runtime

Source-filter settings remain config-only in v1 and are not exposed as CLI
flags during C1-01.

Exact user-facing CLI help text:

- tool namespace help: `Interactive directory-scoped chat over repository files.`
- subcommand help: `Start an interactive read-only chat session for one directory.`

`<directory>` is required and defines the root that all tool access must be
confined to.

## 4.2 Config

The tool uses a standalone chat config file rather than extending the current
project config.

The standalone config does not choose the chat root. The root directory remains
the required `<directory>` CLI positional from `engllm-chat interactive`.

The config is locked to five top-level sections:

- `llm`: provider/model/runtime settings and provider credential metadata
- `source_filters`: include/exclude/hidden-file discovery rules applied within
  the CLI-selected root
- `session`: chat-loop and context-window behavior
- `tool_limits`: read-only tool safety bounds
- `ui`: minimal Textual UI behavior toggles only

Default provider: `ollama`

Config precedence:

- the standalone chat config supplies defaults
- CLI flags override config values for overlapping settings

Example v1 chat config:

```yaml
llm:
  provider: ollama
  model_name: qwen2.5-coder:7b-instruct-q4_K_M
  temperature: 0.1
  api_base_url: http://127.0.0.1:11434
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
  max_tool_result_chars: 24000

ui:
  show_token_usage: true
  show_footer_help: true
```

UI configuration is intentionally minimal in v1:

- no theme customization
- no layout customization
- no keybinding customization

### Credential Entry Policy

Credential resolution order:

1. provider-specific environment variable, if configured and present
2. masked startup prompt, if prompting is enabled and the environment variable
   is absent
3. empty string if the user submits one

Credential-entry rules:

- if the configured provider expects an API key and the configured environment
  variable is present, no startup prompt is shown
- if the environment variable is absent and
  `llm.prompt_for_api_key_if_missing` is true, the Textual app prompts at
  startup using visually masked input
- the prompt accepts an empty string because not all providers require a
  key/token
- any entered value is session-only and is not written to config, disk, or
  environment
- plaintext secrets are not normal persisted config values

Provider expectations:

- `ollama`: no key required, so no prompt by default
- `mock`: no key required, so no prompt by default
- `openai`: defaults to `OPENAI_API_KEY`
- `xai`: defaults to `XAI_API_KEY`
- `anthropic`: defaults to `ANTHROPIC_API_KEY`
- `gemini`: defaults to `GEMINI_API_KEY`

---

## 5. Architecture

The chat tool should follow the repository’s layered architecture:

- `domain/`: typed chat models and config schema
- `core/config/`: standalone chat config loading
- `core/chat/`: deterministic read-only tool implementations and any related
  safety helpers
- `prompts/chat/`: centralized prompt templates/builders
- `llm/`: provider-isolated chat/tool-calling adapters
- `tools/chat/`: orchestration and terminal rendering helpers
- `cli/`: thin command routing only

The tool must not bypass the provider abstraction, and must not embed prompt
text directly inside orchestration code.

---

## 6. LLM Interaction Model

The chat tool should introduce a provider-neutral interactive chat contract
alongside the existing structured one-shot generation contract.

The model interaction must support:

- ordered chat messages
- explicit tool definitions
- model-requested tool calls with structured arguments
- structured tool results returned to the model
- a final structured assistant response
- token usage estimates

The final assistant response schema should include at least:

- `answer`
- `citations`
- `confidence`
- `uncertainty`
- `missing_information`
- `follow_up_suggestions`

If the available evidence is insufficient, the response must explicitly reflect
that conservatively and use `TBD` where appropriate.

---

## 7. Read-Only Tool Catalog

The model must only receive explicit safe tools. Initial minimum toolset:

1. `list_directory`
2. `list_directory_recursive`
3. `find_files`
4. `search_text`
5. `read_file`

Rules:

- all paths must be normalized relative to the configured root
- any path escape outside the root must be rejected
- recursive traversal must not follow symlinks
- tools must not expose arbitrary command execution
- tool results must be structured JSON

`read_file` behavior:

- text files: return bounded text content
- supported non-text files: return Markdown converted via `markitdown`
- unsupported/binary files: return structured failure metadata

---

## 8. Terminal UX

The v1 chat UI should be implemented as a Textual-based terminal application.
It remains terminal-native, but it is no longer planned as a plain prompt-loop
or lightweight ad hoc terminal interface.

Required behavior:

- launch a Textual chat interface from `engllm-chat interactive`
- provide a scrollable transcript pane
- provide a multiline composer at the bottom of the screen
- provide separate transient status and persistent footer areas
- render final answers in the transcript with visible citations and uncertainty
- keep help and exit affordances minimal

Default layout:

- transcript scroll area
- composer container
- transient status row
- persistent footer row

App shell structure:

- the v1 Textual shell should be composed as a `ChatApp` with one main
  `ChatScreen`
- the main chat surface may open a blocking masked `CredentialModal` at startup
  when credential entry is required
- the main chat surface may open an `InterruptConfirmModal` when the user sends
  a new message while a turn is already active
- the selected root and startup guidance should appear as an initial
  system-style transcript entry rather than a dedicated header pane

Session affordances:

- the initial screen should indicate the selected root and provide a brief hint
  for `/help` and exit
- supported v1 commands remain `/help`, `quit`, `exit`
- v1 does not include `/clear`, a history browser, or a command palette

Composer behavior:

- `Enter` submits the current composed message
- `Shift+Enter` inserts a newline in the composer
- pasted multi-line text remains in the composer as one editable message draft
- blank or whitespace-only drafts do not send
- while a turn is active, the composer remains editable
- sending while a turn is active does not queue a second turn
- instead, sending while active opens an interrupt confirmation flow
- if interruption is confirmed, the active turn is cancelled and the new draft
  starts immediately
- if interruption is cancelled, the active turn continues and the current draft
  remains in the composer
- this key behavior is required even if it must be implemented explicitly on top
  of Textual widgets rather than relying on default widget behavior

Stop behavior:

- while a turn is active, the UI should expose a dedicated stop action in the
  composer or footer area
- activating stop interrupts the current turn without requiring a new message
- stop does not remove assistant content that has already streamed into the
  transcript

Status behavior:

- render one transient status area within the Textual interface while the agent
  is working
- status updates should not pollute the transcript history
- status vocabulary remains coarse and limited to phases such as `thinking`,
  `listing files`, `searching text`, `reading file`, and `drafting answer`
- the transient status row clears when the application returns to an idle state

Answer rendering:

- render assistant answers into the transcript as assistant message widgets
- render the answer body first
- render labeled sections only when present:
  - `Citations`
  - `Uncertainty`
  - `Missing Information`
  - `Follow-up Suggestions`
- citations are expanded by default and are not hidden behind an extra command
- interrupted assistant responses remain in the transcript where streaming
  stopped and are visibly marked as interrupted or incomplete
- interrupted assistant responses remain part of the conversation context for
  follow-up turns
- tool activity and coarse status changes do not become transcript entries in v1
- render errors as concise `Error: ...` notices in the interface, then return
  focus to the composer
- do not print stack traces in normal user-facing output
- `/help` renders as a help/system transcript entry
- `quit` and `exit` close the app

The user should see:

- total estimated tokens used in the session
- estimated tokens currently in active context

Token display:

- always render a compact persistent footer summary after each completed answer
- show `session tokens` and `active context tokens`
- confidence may appear alongside the token summary when available
- use the Textual footer for key-hint/discovery support in addition to `/help`
- footer/help guidance should include the stop affordance while a turn is active

Implementation boundary:

- Textual is the planned presentation/controller layer for v1
- orchestration, tool execution, provider interaction, and session state should
  remain outside the Textual widget tree
- UI-facing event/view-model types may be introduced under the chat tool
  namespace to connect the workflow layer to the TUI cleanly
- the transcript should be modeled as durable message widgets rather than a
  single append-only rich log

---

## 9. Session Behavior

Session history is in-memory only for v1.

When the active context approaches the configured token limit:

- drop the oldest raw visible turns from the active context
- surface a warning to the user
- do not perform compaction or summarization yet

Temporary debug artifacts are acceptable, but reusable persistent conversation
history is out of scope for the first release.

---

## 10. Safety Requirements

The tool must prioritize safe inspection over power.

Safety requirements:

- read-only operations only
- bounded file reads and bounded search results
- bounded recursion and bounded tool-call loops
- no shell passthrough
- no network dependency other than the configured local/provider endpoint
- explicit error reporting for provider, validation, and repository failures

---

## 11. Testing Requirements

The feature must be testable without live LLM access.

Tests should cover:

- chat config loading and validation
- deterministic tool behavior
- path confinement and symlink handling
- bounded reads/searches/results
- mock-driven multi-tool chat orchestration
- Ollama adapter request/response handling
- interactive CLI behavior

---

## 12. Roadmap Notes

Expected follow-on work after the first release:

- persistent conversations
- context compaction
- additional provider support
- richer trace/debug artifacts
- improved token accounting when provider metadata is available
