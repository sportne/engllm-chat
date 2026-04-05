# Extending the Chat System

This guide explains how to add or change major pieces of `engllm-chat` without
breaking its design.

It is the practical companion to `docs/CONTRIBUTING_CHAT_ARCHITECTURE.md`.
That document explains where changes belong. This one explains how to carry out
those changes step by step.

## Before You Start

Keep these project rules in mind:

- deterministic tools stay read-only and provider-agnostic
- prompt text stays under `prompts/chat/`
- provider integration stays inside `llm/`
- tool results go back to the model as plain chat messages
- provider-native runtime tool calling should not be reintroduced

If a change pushes against those rules, treat that as a design decision that
needs to be made deliberately, not as an implementation detail.

## Add a New Provider

Use this path when you want to support a new provider or a new OpenAI-compatible
endpoint shape.

### Where to edit

- `src/engllm_chat/domain/models.py`
- `src/engllm_chat/llm/factory.py`
- `src/engllm_chat/llm/openai_compatible.py` and `src/engllm_chat/llm/_openai_compatible/`
- `src/engllm_chat/llm/mock.py` only if tests need new behavior
- `docs/CHAT_USAGE.md` and `docs/CHAT_CONFIG_REFERENCE.md`

### What to change

1. Add the provider name and any default API base URL or API key env var in the
   domain/config layer.
2. Make sure the factory can build the right client for the new provider.
3. If the provider is OpenAI-compatible, prefer using the shared adapter rather
   than creating a separate provider-specific runtime path.
4. If the provider has a real incompatibility, isolate that difference inside
   `llm/` only.
5. Update user-facing docs and examples.

### Interfaces involved

- `ChatLLMConfig`
- provider factory selection in `llm/factory.py`
- `ChatTurnRequest` / `ChatTurnResponse`
- schema-first action envelope parsing in `llm/_openai_compatible/`

### Tests to add or update

- config validation for defaults and env-var resolution
- factory tests for provider selection
- adapter tests for serialization/parsing behavior
- smoke-test docs if the provider should be exercised manually

### What not to change

- do not move provider-specific payload rules into workflow code
- do not let provider behavior change the deterministic tool contract
- do not bypass structured response validation

## Add a New Deterministic Tool

Use this path when you want the model to gain a new read-only capability.

### Where to edit

- `src/engllm_chat/tools/chat/models.py`
- `src/engllm_chat/core/chat/`
- `src/engllm_chat/tools/chat/registry.py`
- prompt docs only if the usage guidance needs updating

### What to change

1. Define the tool argument model in `tools/chat/models.py`.
2. Implement the deterministic behavior in `core/chat/`.
3. Return a typed result model from the deterministic layer.
4. Register the tool in `tools/chat/registry.py` with:
   - name
   - description
   - argument model
   - runner
5. Make sure the prompt-visible tool definitions still come from the registry.

### Interfaces involved

- `ChatToolCall`
- `ChatToolResult`
- `ChatToolDefinition`
- the typed argument model for the new tool
- the typed result model returned by the deterministic tool

### Tests to add or update

- deterministic tool behavior tests in `tests/test_chat_tools.py`
- registry tests for definition shape and dispatch
- workflow tests only if the new tool changes orchestration assumptions

### What not to change

- do not make the tool call providers or the network
- do not add write or shell execution behavior
- do not let tool definitions live only in prompt text

## Change Prompt Behavior

Use this path when the model needs better instructions, better tool-selection
guidance, or a clearer response contract.

### Where to edit

- `src/engllm_chat/prompts/chat/builders.py`
- `src/engllm_chat/prompts/chat/templates.py`

### What to change

1. Update the centralized prompt text or prompt-construction logic.
2. Keep prompt wording out of workflow and provider code.
3. If a prompt change depends on tool shape, confirm the tool registry still
   matches the prompt-visible definitions.
4. Validate the change with workflow tests or a smoke run when possible.

### Interfaces involved

- `build_chat_system_prompt(...)`
- provider-facing tool definitions derived from the registry

### Tests to add or update

- prompt builder tests
- workflow tests that assert the tool catalog or prompt behavior when relevant

### What not to change

- do not duplicate prompt instructions in multiple layers
- do not patch around a prompt issue by changing deterministic tool behavior

## Change Workflow Behavior

Use this path when you want to change how a turn is orchestrated.

### Where to edit

- `src/engllm_chat/tools/chat/workflow.py`
- `src/engllm_chat/tools/chat/_workflow/`
- `src/engllm_chat/tools/chat/registry.py` if dispatch behavior is affected

### What to change

1. Identify whether the change affects:
   - session-context preparation
   - token accounting
   - continuation behavior
   - interruption behavior
   - result finalization
   - interactive runner behavior
2. Keep the top-level workflow entrypoints stable unless there is a strong
   reason not to.
3. Preserve the one-action-at-a-time structured loop.
4. Treat continuation and interruption as normal workflow states, not as hidden
   exceptions.

### Interfaces involved

- `run_chat_turn(...)`
- `run_chat_session_turn(...)`
- `ChatSessionTurnRunner`
- `ChatWorkflowTurnResult`

### Tests to add or update

- `tests/test_chat_workflow.py`
- `tests/test_chat_workflow_internals.py`
- UI tests if the visible status/event behavior changes

### What not to change

- do not move provider serialization into the workflow
- do not let the UI become the source of truth for workflow state
- do not let workflow code hardcode prompt wording

## Add or Change UI Behavior

Use this path when the user experience changes but the underlying workflow does
not need to.

### Where to edit

- `src/engllm_chat/tools/chat/app.py`
- `src/engllm_chat/tools/chat/controller.py`
- `src/engllm_chat/tools/chat/presentation.py`
- `src/engllm_chat/tools/chat/screens.py`

### What to change

1. Keep UI work focused on presentation, event wiring, and interaction flow.
2. Reuse the workflow/session models instead of duplicating state logic in the
   widgets.
3. If the change needs a new workflow status or result shape, change the
   workflow deliberately first.

### Tests to add or update

- `tests/test_chat_ui.py`
- workflow tests if the UI change depends on workflow semantics

### What not to change

- do not hide business logic in widget methods
- do not duplicate deterministic tool rules inside the UI

## A Safe Checklist for Any Change

No matter what you are adding, a safe sequence is:

1. update typed models if the data contract changes
2. update the owning layer only
3. wire the change through the next boundary if needed
4. update prompts or docs if user-facing behavior changes
5. add or update tests where the behavior actually lives

That order keeps the codebase understandable and makes regressions easier to
find.

## Verification Commands

After architecture or behavior changes, run:

```bash
.venv/bin/python -m pytest
.venv/bin/python -m mypy src
.venv/bin/python -m ruff check src tests
```

If the change touches real-provider behavior, also consider:

```bash
make smoke-chat
make smoke-ollama-chat
```

For verbose provider debugging:

```bash
.venv/bin/python -m engllm_chat.smoke_chat --directory . --require-tool-call --verbose-llm
```
