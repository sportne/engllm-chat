# Directory Chat Implementation Plan

## Summary

This plan adds a new local-first interactive tool, `engllm-chat`, for
directory-scoped conversational question answering. The implementation is split
into small increments that preserve architectural boundaries and keep all
deterministic logic outside the LLM boundary.

Provider scope for the first implementation is:

- `ollama`
- `mock`
- `openai`
- `xai`
- `anthropic`
- `gemini`

The plan assumes no code reuse from existing workflow modules under other
`src/engllm_chat/tools/*` namespaces.

---

## Phase 1: UX And Surface Design

Deliverables:

- final public CLI shape for `engllm-chat interactive <directory> --config <path>`
- standalone chat config shape
- Textual-based chat UI decisions for transcript, composer, status area, and
  token display

Key design decisions:

- adopt Textual as the primary and only planned v1 chat UI
- ship Textual as a core dependency rather than an optional extra
- keep the public command surface unchanged: `chat interactive`
- use a simple Textual layout: transcript pane, multiline composer, and
  footer/status area
- show session token estimate and active-context token estimate in the footer
- keep `/help` and `quit`/`exit` style affordances minimal
- expose only one initial public command in v1: `chat interactive`
- keep source-filter settings config-only in v1
- keep root selection CLI-only in v1; the standalone config does not define the
  chat root
- let the standalone chat config provide defaults and let CLI flags override
  those defaults at runtime

Locked CLI override set for C1-01:

- provider/model:
  - `--provider {ollama,mock,openai,xai,anthropic,gemini}`
  - `--model <name>`
  - `--temperature <float>`
  - `--api-base-url <url>`
- session limits:
  - `--max-context-tokens`
  - `--max-tool-round-trips`
  - `--max-tool-calls-per-round`
  - `--max-total-tool-calls-per-turn`
- tool-safety limits:
  - `--max-entries-per-call`
  - `--max-recursive-depth`
  - `--max-search-matches`
  - `--max-read-lines`
  - `--max-file-size-characters`
  - `--max-tool-result-chars`

Locked Textual UX for C1-02:

- `chat interactive` launches a Textual app, not a prompt-loop CLI
- default layout is transcript pane + multiline composer + footer/status area
- `Enter` sends the composed message
- `Shift+Enter` inserts a newline in the composer
- pasted multi-line text remains as one editable draft in the composer
- startup affordance shows the selected root and brief `/help` / exit guidance
- supported v1 commands remain `/help`, `quit`, `exit`
- status updates appear in a transient status/footer area and do not pollute the
  transcript
- answer rendering shows the answer first, then labeled `Citations`,
  `Uncertainty`, `Missing Information`, and `Follow-up Suggestions` sections
  only when present
- each completed answer updates a compact footer with session-token and
  active-context-token estimates, with confidence shown when available

Acceptance:

- UX rules are documented before implementation begins
- command line and config responsibilities are fully specified

---

## Phase 2: Typed Models And Config

Deliverables:

- chat config models in `domain/`
- standalone chat config loader in `core/config/`
- typed message, tool-call, tool-result, token-usage, and final-response models

Required interfaces:

- `ChatConfig`
- `ChatLLMConfig`
- `ChatSourceFilters`
- `ChatSessionConfig`
- `ChatToolLimits`
- `ChatUIConfig`
- provider-neutral chat request/response models

Notes:

- `ChatLLMConfig` includes provider credential metadata and startup-prompt
  policy, but does not store persisted plaintext secrets

Acceptance:

- all chat config and response shapes validate through typed models
- invalid limits fail with explicit validation errors

---

## Phase 3: Provider Abstraction

Deliverables:

- a new provider-neutral chat/tool-calling contract in `llm/base.py`
- Ollama chat adapter
- deterministic mock chat adapter
- factory wiring for chat-specific provider creation

Notes:

- keep the existing one-shot structured-generation interface unchanged
- isolate all provider-specific payload handling inside `llm/`
- treat displayed token values as estimates in v1

Acceptance:

- mock, Ollama, and hosted OpenAI-compatible adapters satisfy the shared chat contract
- provider failures produce explicit `LLMError` messages

---

## Phase 3.5: Textual App Structure

Deliverables:

- Textual app shell/layout plan
- widget model for transcript, composer, and footer/status area
- UI-facing event/view-model contract between chat orchestration and the TUI
- interrupt and startup-modal interaction model for active turns

Implementation notes:

- startup masked-key entry is part of the Textual UI flow for providers that
  declare an expected environment variable and have prompting enabled
- the Textual shell should be structured as:
  - `ChatApp`
  - main `ChatScreen`
  - optional `CredentialModal`
  - optional `InterruptConfirmModal`
  - scrollable message-widget transcript
  - composer controller/widget
  - separate transient status row and persistent footer row
- the UI layer should consume UI-facing state/events rather than raw provider
  or tool objects
- minimum UI-facing state should cover:
  - transcript entries, including interrupted assistant-message state
  - transient status state
  - footer/token summary state
  - credential-prompt state
  - active-turn and interrupt-confirmation state
  - composer draft state

Acceptance:

- the Textual app structure is defined before CLI/TUI implementation begins
- the workflow layer remains independent from Textual widgets

---

## Phase 4: Deterministic Read-Only Toolset

Deliverables:

- `core/chat/` tool implementations
- strict path confinement to the configured root
- deterministic JSON results for all tool calls

Minimum tools:

1. `list_directory`
2. `list_directory_recursive`
3. `find_files`
4. `search_text`
5. `read_file`

Important rules:

- never follow symlinks recursively
- never expose arbitrary command execution
- bound recursion, file sizes, line counts, and result counts
- convert supported non-text formats with `markitdown`

Acceptance:

- all tools are deterministic and read-only
- escape attempts outside the root fail explicitly

---

## Phase 5: Prompting And Orchestration

Deliverables:

- `prompts/chat/` templates/builders
- system prompt with tool catalog and short usage examples
- chat session orchestrator under `tools/chat/`

Runtime flow:

1. user submits a question
2. model receives current messages plus explicit tool definitions
3. model requests zero or more tool calls
4. application executes allowed tools and returns JSON tool results
5. model either requests more tools or emits a final structured answer
6. application renders the answer and token/status information

Guardrails:

- bound total tool calls per turn
- bound tool rounds per turn
- fail clearly if the model never reaches a final response

Acceptance:

- multi-step tool exploration works end to end with mock and Ollama adapters
- final responses are always validated structured data

---

## Phase 6: Session Management

Deliverables:

- in-memory session state for visible turns
- context-window limit handling
- oldest-turn truncation with user-visible warning

Explicit out of scope:

- context compaction
- summarization
- persisted session history

Acceptance:

- follow-up questions work inside one process
- overflow behavior is deterministic and visible to the user

---

## Phase 7: CLI Integration

Deliverables:

- `chat` tool namespace registration
- `interactive` subcommand
- config/runtime override wiring
- Textual app runner wiring

CLI flags should implement the locked C1-01 surface:

- provider/model/temperature
- Ollama base URL
- session limits
- tool safety limits

Precedence rule:

- chat config supplies defaults
- CLI flags override those defaults at runtime

Acceptance:

- `engllm-chat interactive <directory> --config <path>` runs through the new
  orchestration path and launches the Textual app
- CLI remains thin and delegates behavior into the chat tool namespace / Textual
  app layer
- startup credential entry is handled through the startup modal flow when needed
- busy sends trigger interrupt confirmation instead of queueing
- a dedicated bare-stop action is available while a turn is active
- interrupted streamed assistant output remains visible and contextual in the
  transcript

---

## Phase 8: Documentation

Deliverables:

- user-facing chat manual
- architecture documentation updates
- usage documentation updates
- roadmap/task documentation

Required documentation topics:

- quick start with Ollama
- config shape
- tool behavior and safety constraints
- troubleshooting for provider and path errors
- roadmap notes for persistence and compaction

Acceptance:

- a new user can configure and run the tool from docs alone

---

## Phase 9: Testing

Deliverables:

- config tests
- core tool tests
- provider tests
- orchestration tests
- Textual UI tests
- CLI entrypoint tests
- acceptance tests over fixture directories

Core scenarios:

- directory listing and recursive listing
- symlink non-following
- root-escape rejection
- grep-like searching
- Textual transcript + composer + footer layout
- `Enter` send / `Shift+Enter` newline behavior
- pasted multi-line draft handling
- status/footer updates that do not pollute transcript history
- bounded file reads
- `markitdown` fallback behavior
- tool-call loop completion and failure handling
- active-context truncation
- Ollama transport and payload errors

Acceptance:

- deterministic logic is covered without live LLM access
- mock-backed workflow tests are the default path for orchestration coverage

---

## Defaults And Assumptions

- provider scope for first implementation: `ollama` and `mock`
- token values displayed to users are estimates in v1
- no dependence on existing workflow code under other tool namespaces
- no persistent sessions in the first release
- no context compaction in the first release
