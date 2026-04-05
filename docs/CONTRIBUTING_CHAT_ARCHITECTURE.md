# Contributing to the Chat Architecture

This guide explains where to make changes in `engllm-chat` when you want to
extend the system.

It is aimed at contributors who may understand the high-level docs but are not
yet comfortable navigating the codebase on their own.

## Start With the Right Question

Before changing code, ask what kind of change you are making.

Most chat-related work falls into one of these buckets:

- adding or changing a provider integration
- adding or changing a deterministic tool
- changing prompt behavior
- changing workflow/orchestration behavior
- changing the Textual UI
- changing configuration or shared model shapes

Each kind of change belongs in a different layer. Keeping those boundaries
clear is a big part of what makes this project a useful reference
implementation.

## If You Want to Add or Change a Provider

Start in:

- `src/engllm_chat/llm/`

Important places:

- `llm/base.py`
  - shared provider-neutral request/response contract
- `llm/factory.py`
  - chooses the right client from config
- `llm/openai_compatible.py`
  - public façade for the shared OpenAI-compatible adapter
- `llm/_openai_compatible/`
  - transport, serialization, parsing, and retry helpers
- `llm/mock.py`
  - deterministic fake provider used in tests

Use this layer when:

- you need to support a new provider
- you need to change how structured provider calls are serialized or parsed
- you need to change retry behavior for schema-invalid responses

Do **not** put provider-specific logic into:

- `core/chat/`
- `tools/chat/`
- `prompts/chat/`

Those layers should not need to know provider-specific request formats.

## If You Want to Add or Change a Deterministic Tool

Start in:

- `src/engllm_chat/core/chat/`
- `src/engllm_chat/tools/chat/registry.py`
- `src/engllm_chat/tools/chat/models.py`

The usual flow is:

1. define or update the tool argument model in `tools/chat/models.py`
2. implement the deterministic behavior in `core/chat/`
3. expose the tool through `tools/chat/registry.py`
4. make sure the prompt tool catalog still matches the runtime registry
5. add behavior tests

Use this layer when:

- you want the model to have a new read-only capability
- you need to tighten path rules or result shaping
- you need to improve deterministic file-handling behavior

Do **not** make the tool layer depend on:

- provider SDKs
- prompt-building code
- Textual UI code

The tool layer should stay deterministic, read-only, and easy to unit test.

## If You Want to Change Prompt Behavior

Start in:

- `src/engllm_chat/prompts/chat/`

Important places:

- `prompts/chat/builders.py`
- `prompts/chat/templates.py`

Use this layer when:

- you want the model to prefer better evidence
- you want to improve tool-selection guidance
- you want to clarify the final-response contract

Do **not** hardcode prompt text into:

- workflow code
- provider adapters
- UI code

If the model should be told something, that instruction should usually live in
the prompt layer.

## If You Want to Change Workflow Behavior

Start in:

- `src/engllm_chat/tools/chat/workflow.py`
- `src/engllm_chat/tools/chat/_workflow/`
- `src/engllm_chat/tools/chat/registry.py`

Use this layer when:

- you want to change continuation behavior
- you want to change interruption behavior
- you want to change session context trimming
- you want to change how a tool result is fed back into the conversation

This layer is responsible for orchestration. It coordinates the other layers,
but it should not absorb their responsibilities.

For example:

- prompt wording does not belong here
- provider-specific payload formatting does not belong here
- raw filesystem traversal logic does not belong here

## If You Want to Change the Textual UI

Start in:

- `src/engllm_chat/tools/chat/app.py`
- `src/engllm_chat/tools/chat/controller.py`
- `src/engllm_chat/tools/chat/presentation.py`
- `src/engllm_chat/tools/chat/screens.py`

Use this layer when:

- you want to change transcript presentation
- you want to change footer or status behavior
- you want to change composer interactions
- you want to change modal behavior

Try to keep the UI layer focused on presentation, event wiring, and user
experience. The workflow and deterministic tool behavior should remain outside
the Textual widgets.

## If You Want to Change Shared Model Shapes

Start in:

- `src/engllm_chat/domain/models.py`
- `src/engllm_chat/domain/_models/`

Use this layer when:

- you want to add a field to a shared chat model
- you want to change config validation
- you want to change the final response schema

Be careful here. These models are used across:

- config loading
- provider adapters
- workflow orchestration
- deterministic tools
- UI rendering
- tests

A small model change can ripple through many layers.

## Guardrails: What Not to Do

These rules are core to the project’s design.

### Do not reintroduce provider-native tool calling

This project intentionally uses a schema-first action envelope instead of
provider-native runtime tool calling.

If you are changing provider code, preserve that contract unless the project
direction changes intentionally.

### Do not add nondeterministic behavior to `core/chat/`

The deterministic tool layer should not:

- call provider SDKs
- call the network
- guess what the user meant
- depend on UI state

It should behave like normal application logic with clear, typed inputs and
outputs.

### Do not hardcode prompt text outside `prompts/chat/`

Prompt logic should stay centralized. This makes the system easier to reason
about and easier to update without hunting through unrelated files.

### Do not let the model bypass the tool layer

If the model needs filesystem evidence, it should get it through deterministic
tools. Avoid designs where the model is given broader execution power than the
rest of the architecture expects.

### Do not mix UI concerns into workflow or provider code

The UI can present workflow state, but it should not become the source of truth
for workflow decisions.

## A Safe Change Pattern

When adding a feature, a good default sequence is:

1. update the typed models if the data contract needs to change
2. update deterministic logic or provider logic in its own layer
3. wire the change through the workflow or registry if needed
4. update prompts if the model needs new instructions
5. update the UI only if presentation or interaction needs to change
6. add or update tests at the layer where the behavior lives

This sequence helps keep the design understandable.

## Good Places to Read Before Editing

If you are unsure where something belongs, read in this order:

1. `docs/ARCHITECTURE.md`
2. `docs/LLM_STRUCTURED_CALLS.md`
3. `docs/DETERMINISTIC_TOOLS.md`
4. `src/engllm_chat/domain/models.py`
5. the layer you think you need to change

That gives you both the design intent and the implementation shape.

## How to Keep the Project Teachable

This codebase is intended to be a reference implementation, not just a working
tool. When contributing, prefer choices that help a newer developer understand
the system:

- keep module responsibilities clear
- avoid clever shortcuts that hide the architecture
- add short explanatory comments when the “why” is not obvious
- preserve typed boundaries between layers
- prefer a small number of clear patterns used consistently

The project becomes more useful when a reader can follow the flow without
already being an expert.
