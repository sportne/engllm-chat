# `engllm-chat` as a Reference Implementation

This guide is for readers who want to learn how to build a system like
`engllm-chat`, not just use this repository.

The project is intentionally structured to demonstrate a conservative pattern
for connecting an LLM to local tools:

- keep the LLM interaction structured
- keep the local tool behavior deterministic
- keep the layers separated so each part stays understandable
- make the non-LLM parts easy to test without network access

If you are newer to software architecture, this document is meant to explain
the design in plain language and show a safe order for building a similar
system.

## The Core Idea

At a high level, this kind of application has three jobs:

1. accept a user request
2. let the model gather evidence through a limited toolset
3. return a grounded answer based on that evidence

The tricky part is making those three jobs work together without creating a
system that is hard to test, hard to reason about, or too powerful for its own
safety model.

`engllm-chat` handles that by separating responsibilities:

- typed models define the data contracts
- deterministic tools do the real file work
- the provider layer only handles LLM communication
- the workflow orchestrates turns and limits
- the UI presents state to the user

That separation is the main pattern to copy.

## A Good Build Order

If you were building a similar project from scratch, a safe order looks like
this.

### 1. Define the typed contracts first

Before worrying about prompts or models, decide what data the rest of the
application will trust.

In this project, that means models such as:

- `ChatMessage`
- `ChatToolCall`
- `ChatToolResult`
- `ChatFinalResponse`
- `ChatConfig`

Why start here:

- every other layer depends on these shapes
- validation errors become explicit instead of accidental
- tests can talk in terms of shared models instead of loose dictionaries

If you skip this step, the workflow usually becomes a pile of ad hoc JSON and
string parsing.

## 2. Build the deterministic tool layer next

The model should not be the thing that actually touches the filesystem.

Instead, build normal application functions that:

- take typed inputs
- enforce safety rules
- return typed outputs

In `engllm-chat`, this is the read-only tool layer in `core/chat/`.

Important design choices to copy:

- paths are confined to one root
- direct symlink targets are rejected
- readable content is bounded
- tool results have fixed shapes
- the tool layer does not know anything about the provider SDK or UI

This is what makes the project safe enough to hand to the model.

## 3. Add a registry between the model and the deterministic tools

Do not let prompt text become the hidden source of truth for what tools exist.

Instead, keep a registry that knows:

- the tool name the model sees
- the description the model sees
- the JSON schema for arguments
- the function that actually runs the tool

In this project, `tools/chat/registry.py` is that bridge.

Why this matters:

- the prompt and runtime use the same tool catalog
- adding a new tool becomes a repeatable process
- argument validation happens at one clear boundary

Without a registry, the tool surface tends to drift between prompt text,
provider formatting, and actual runtime behavior.

## 4. Use structured LLM calls instead of free-form output

This project treats the model as a component that must return one validated
action per step.

That action is either:

- a tool request
- or a final response

Why this is safer than free-form chat output:

- the workflow does not have to guess what the model meant
- retries can be explicit when the response shape is wrong
- the same contract works across different providers

You can think of this as a very small protocol between the workflow and the
model.

The project deliberately does **not** rely on provider-native tool calling as
the runtime contract. That keeps the orchestration loop more uniform.

## 5. Keep the workflow as the orchestrator, not the owner of everything

The workflow should coordinate the pieces, not absorb their responsibilities.

In a system like this, the workflow should be responsible for:

- building the conversation for the current turn
- asking the model for one action
- executing a deterministic tool when requested
- feeding the tool result back into the conversation
- enforcing continuation and interruption limits

It should **not** become the place where:

- prompt text lives
- provider-specific payload formatting lives
- filesystem logic lives
- UI rendering rules live

That separation is a big reason this project stays teachable.

## 6. Add the UI after the core loop is already understandable

The UI matters, but it should sit on top of the workflow rather than define the
business logic.

That is why `engllm-chat` treats the Textual app as a presentation layer over
the underlying chat session and workflow.

This keeps two things easier:

- you can test the workflow without the UI
- you can improve the UI without rewriting the core architecture

For a less-experienced developer, this is an important lesson:

- build the logic first
- build the presentation around it

## 7. Test the non-LLM behavior directly

A project like this only stays reliable if most of its behavior can be tested
without calling a real model.

That is why this repository emphasizes:

- unit tests for deterministic tools
- unit tests for config and model validation
- workflow tests using mock providers
- a separate smoke path for real-provider checks

This split is worth copying:

- deterministic tests for correctness
- smoke tests for real integration confidence

## The Architecture Rules This Project Is Trying to Teach

If you only take a few lessons from this repository, they should be these:

### Keep the LLM boundary narrow

The model decides which action to request. It should not decide how the tool
works internally.

### Keep the dangerous logic out of the model

Filesystem rules, path safety, size limits, and conversion behavior belong in
normal code.

### Keep prompts centralized

If the model needs an instruction, it should live in the prompt layer, not be
hidden in workflow branches.

### Keep provider code separate from local tool code

The provider layer should talk to remote or local model APIs. It should not own
filesystem behavior.

### Keep structured validation at the boundaries

Validate config input, validate tool calls, validate provider responses, and
validate tool result shapes. That reduces ambiguity everywhere else.

## A Practical Reading Order for Learners

If you want to study the project as an example, this is a good order:

1. `README.md`
2. `docs/ARCHITECTURE.md`
3. `docs/LLM_STRUCTURED_CALLS.md`
4. `docs/DETERMINISTIC_TOOLS.md`
5. `docs/CONTRIBUTING_CHAT_ARCHITECTURE.md`
6. `src/engllm_chat/domain/models.py`
7. `src/engllm_chat/tools/chat/registry.py`
8. `src/engllm_chat/tools/chat/workflow.py`
9. `src/engllm_chat/llm/openai_compatible.py`
10. `src/engllm_chat/core/chat/listing.py`

That order moves from concept to implementation.

## What This Project Does Not Try to Demonstrate

This repository is intentionally conservative. It is not trying to show:

- autonomous shell-executing agents
- write-capable coding agents
- provider-native tool calling as the main orchestration contract
- streaming-token cancellation in the provider layer
- a highly dynamic plugin system for tools

Those can be valid product choices, but they would teach a different set of
tradeoffs.

## The Main Takeaway

The most important design lesson in `engllm-chat` is not any single module.

It is the combination of these choices:

- typed contracts
- deterministic tools
- structured provider responses
- a clear orchestration loop
- a UI layered on top of that loop

That combination is what makes the project useful as both a tool and a
teaching example.

## Practical Next Reading

Once the ideas in this guide make sense, the most useful practical companions
are:

- `docs/EXTENDING_CHAT_SYSTEM.md`
- `docs/TESTING_AND_DEBUGGING.md`
- `docs/CHAT_CONFIG_REFERENCE.md`
- `docs/GLOSSARY.md`
