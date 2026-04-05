# EngLLM Chat Architecture

This document explains how `engllm-chat` is put together and how one chat turn
moves through the system.

It is written for readers who may be new to this style of project. The goal is
to make the codebase easier to navigate before you need to read individual
modules in detail.

## Big Picture

`engllm-chat` is a terminal chat tool for exploring one directory root. It is
designed around a few important rules:

- the model can only inspect the filesystem through a small set of explicit,
  read-only tools
- all non-LLM behavior should be deterministic and easy to test
- the provider layer should be isolated from the filesystem and UI layers
- prompts should stay centralized instead of being scattered across the codebase
- tool results are returned to the model as plain chat messages, not
  provider-native tool result payloads

The project is trying to do two jobs at once:

- be a useful interactive codebase chat tool
- be a clear example of how to build a safe structured LLM workflow

## High-Level System Map

The codebase is organized by responsibility.

### `domain/`

This layer defines the shared typed models and errors used across the project.

Examples:

- provider and chat configuration models
- provider-neutral chat messages
- tool-call and tool-result models
- final response, citations, and token-usage models
- shared domain errors

If you want to understand what shape of data flows through the system, start
here.

### `config/`

This layer loads YAML config into the typed domain models.

Its main job is to take user-facing configuration files and turn them into
validated `ChatConfig` objects that the rest of the system can trust.

### `core/chat/`

This layer contains the deterministic filesystem tools.

These tools do the real file and directory work:

- listing directories
- searching for file paths
- searching inside readable file contents
- inspecting file metadata
- reading bounded file content

This layer should not know anything about provider SDKs or Textual widgets. It
is intentionally read-only and provider-agnostic.

### `llm/`

This layer contains the provider abstraction and provider-specific adapters.

Its job is to:

- accept provider-neutral chat messages and tool definitions
- talk to a real provider or a mock provider
- parse the provider response into the project’s structured result models

The main production adapter is the OpenAI-compatible path, which is used both
for hosted providers and for Ollama’s `/v1` endpoint.

### `prompts/chat/`

This layer builds the system prompt and related prompt text for the chat
workflow.

The key design rule here is that workflow code should not hardcode prompt text.
If you want to change what the model is told, this is the first place to look.

### `tools/chat/`

This layer owns orchestration and the terminal chat experience.

It includes:

- tool registry wiring
- per-turn workflow logic
- in-memory session state
- continuation and interruption behavior
- the Textual app and controller pieces

This is the layer that coordinates the provider, prompts, domain models, and
deterministic tools into one working chat system.

### `cli/`

This layer is intentionally thin.

It parses command-line input and routes into the interactive app or the probe
command. It should not contain heavy business logic.

## One Chat Turn, End to End

Here is what happens when a user asks a question in the interactive chat UI.

### 1. The user submits a message

The Textual app in `tools/chat/` receives the draft from the composer and hands
it to the chat workflow.

### 2. The workflow builds the active conversation

The workflow:

- builds the system prompt from `prompts/chat/`
- takes the current in-memory session history
- trims old whole turns if needed to stay within the configured context limit
- appends the new user message

At this point, the workflow has a provider-neutral list of chat messages.

### 3. The workflow sends one structured request to the provider layer

The provider adapter receives:

- the ordered chat messages
- the expected final response model
- the tool definitions the model is allowed to request

The model is not allowed to return free-form output. It must return one
structured action:

- either a tool request
- or a final response

### 4. If the model requests a tool, the workflow validates and executes it

If the model chooses a tool:

- the tool name and arguments are validated against typed models
- the workflow dispatches through the chat tool registry
- the registry calls the deterministic implementation in `core/chat/`
- the tool result is wrapped in a typed `ChatToolResult`

That tool result is then added back into the conversation as a plain chat
message with `role="tool"`.

### 5. The workflow loops until it reaches a boundary

The workflow keeps going until one of these things happens:

- the model returns a final response
- the model needs more tool rounds than allowed
- the model needs more tool-call budget than allowed
- the user interrupts the turn

If the model hits one of the safety boundaries, the workflow returns
`needs_continuation` instead of crashing.

### 6. The UI renders the result

When the workflow finishes the turn, the UI updates:

- transcript entries
- status text
- footer token information
- continuation or interruption state

The UI simulates live assistant typing, but this is a workflow/UI behavior. It
does not rely on provider-side streaming.

## Why The Project Uses Structured Actions

Many LLM systems rely on provider-native tool calling. This project does not.

Instead, the provider is asked to return a schema-validated action envelope on
each step. That keeps the workflow more uniform across providers and makes the
non-provider logic easier to test.

At a high level, the model returns something conceptually like:

```json
{
  "action": {
    "kind": "tool_request",
    "tool_name": "search_text",
    "arguments": {
      "path": ".",
      "query": "OpenAICompatibleChatLLMClient"
    }
  }
}
```

or:

```json
{
  "action": {
    "kind": "final_response",
    "response": {
      "answer": "The shared OpenAI-compatible adapter lives in ...",
      "citations": []
    }
  }
}
```

If the model returns something that does not match the schema, the provider
adapter gives corrective feedback and retries up to the configured limit.

## Why The Deterministic Tool Layer Matters

The model is not trusted to inspect the filesystem directly.

Instead, the deterministic tools do the actual work and enforce the project’s
safety rules:

- paths must stay inside the configured root
- direct symlink targets are rejected
- reads are bounded
- non-text content is handled through controlled conversion rules
- tool outputs are typed and testable

This keeps the dangerous or ambiguous parts of the system out of the LLM.

## Where To Start Reading The Code

If you are new to the project, this is a good reading order:

1. `README.md`
2. `docs/CHAT_USAGE.md`
3. `docs/ARCHITECTURE.md`
4. `src/engllm_chat/domain/models.py`
5. `src/engllm_chat/tools/chat/workflow.py`
6. `src/engllm_chat/tools/chat/registry.py`
7. `src/engllm_chat/core/chat/listing.py`
8. `src/engllm_chat/llm/openai_compatible.py`

That path moves from the outside of the system inward.

## Current Teaching Focus

If you are reading this project as a reference implementation, the most useful
patterns to study are:

- how typed models define the system contract
- how the workflow stays provider-neutral
- how deterministic tools are separated from the LLM layer
- how prompts are centralized
- how continuation and interruption are treated as normal workflow states

This architecture guide is the starting map. For the next layer of detail, read:

- `docs/LLM_STRUCTURED_CALLS.md`
- `docs/DETERMINISTIC_TOOLS.md`
- `docs/CONTRIBUTING_CHAT_ARCHITECTURE.md`
- `docs/EXTENDING_CHAT_SYSTEM.md`
- `docs/TESTING_AND_DEBUGGING.md`
- `docs/CHAT_CONFIG_REFERENCE.md`
- `docs/GLOSSARY.md`
