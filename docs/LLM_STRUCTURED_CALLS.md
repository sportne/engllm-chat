# Structured LLM Calls in `engllm-chat`

This document explains how `engllm-chat` talks to language models.

It is written for readers who want to understand the project’s core LLM
pattern, especially if they have heard about provider-native tool calling but
have not built a structured action loop before.

## The Short Version

On each step, the model must return exactly one structured action:

- either a tool request
- or a final response

The application validates that action, executes it if needed, and then feeds
the result back into the conversation.

The project does **not** rely on provider-native tool calling at runtime.

## Why This Project Uses a Schema-First Action Envelope

Many model providers offer their own tool-calling API formats. Those can be
useful, but they also create a few problems for a project like this:

- every provider has slightly different behavior
- provider-native tool payloads make the workflow more provider-specific
- tests become harder to keep uniform across local and hosted providers
- it becomes easier for workflow behavior to drift between providers

`engllm-chat` instead uses one provider-neutral contract:

1. send ordinary chat messages plus a structured response schema
2. require the model to emit one validated action
3. if the action is a tool request, execute a deterministic tool
4. append the tool result as a plain chat message
5. ask the model for the next action

That gives the project one orchestration loop that works the same way across:

- Ollama through its `/v1` OpenAI-compatible endpoint
- hosted OpenAI-compatible providers
- the mock provider used in tests

## The Two Allowed Actions

At a high level, the model is only allowed to return one of two action kinds.

### 1. Tool request

Conceptually:

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

This means:

- the model is not answering yet
- it wants the application to run one specific tool
- the tool arguments must match that tool’s typed schema

### 2. Final response

Conceptually:

```json
{
  "action": {
    "kind": "final_response",
    "response": {
      "answer": "The shared OpenAI-compatible adapter lives in `src/engllm_chat/llm/openai_compatible.py`.",
      "citations": [],
      "confidence": 0.84,
      "uncertainty": [],
      "missing_information": [],
      "follow_up_suggestions": []
    }
  }
}
```

This means:

- the model is done exploring for this turn
- the final answer must match the `ChatFinalResponse` schema

## What “Structured” Means Here

The provider adapter does not accept free-form final output as valid just
because it looks reasonable.

Instead, the response has to validate against a Pydantic model. If it does not,
the adapter treats that as a schema failure, gives corrective feedback, and
tries again up to the configured retry limit.

So “structured” here means:

- the model output has a required shape
- the output is validated before the rest of the app trusts it
- the workflow can branch on the result without guessing

## The End-to-End Turn Cycle

Here is the full cycle in plain language.

### Step 1. Build the request

The workflow prepares:

- the system prompt
- prior chat messages
- the latest user message
- the tool catalog the model is allowed to request
- the final response schema

### Step 2. Ask the provider for one action

The provider adapter sends the messages to the model and asks for one action
that matches the required schema.

### Step 3. Validate the action

If the provider response is malformed or does not match the schema:

- the adapter adds corrective feedback to the conversation
- the adapter retries

If the response does validate, the workflow looks at `action.kind`.

### Step 4. If the model requested a tool, run it

The workflow:

- validates the tool name and arguments
- dispatches through the chat tool registry
- executes the deterministic tool implementation
- wraps the result in a typed `ChatToolResult`

### Step 5. Feed the tool result back as a plain chat message

This project does **not** send provider-native tool result objects back to the
model.

Instead, it adds an ordinary chat message that contains the tool result data in
plain structured text. This keeps the workflow provider-neutral.

### Step 6. Ask for the next action

The workflow loops again.

The model can:

- request another tool
- or return a final response

The loop stops when:

- a valid final response is returned
- tool round or tool budget limits are reached
- the user interrupts the workflow

## A Concrete Example Using This Repo

Suppose the user asks:

> Which file implements the shared OpenAI-compatible chat provider?

The turn might look like this:

### User message

```text
Which file implements the shared OpenAI-compatible chat provider?
```

### Model action 1

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

### Tool result fed back into the conversation

Conceptually:

```json
{
  "call_id": "tool-call-0",
  "tool_name": "search_text",
  "status": "ok",
  "payload": {
    "matches": [
      {
        "path": "src/engllm_chat/llm/openai_compatible.py",
        "line_number": 34,
        "line_text": "class OpenAICompatibleChatLLMClient:"
      }
    ]
  }
}
```

### Model action 2

```json
{
  "action": {
    "kind": "final_response",
    "response": {
      "answer": "The shared OpenAI-compatible adapter is implemented in `src/engllm_chat/llm/openai_compatible.py`.",
      "citations": [
        {
          "source_path": "src/engllm_chat/llm/openai_compatible.py",
          "line_start": 34,
          "line_end": 34
        }
      ],
      "confidence": 0.92,
      "uncertainty": [],
      "missing_information": [],
      "follow_up_suggestions": []
    }
  }
}
```

That is the full structured loop:

- ask for one action
- validate it
- run a deterministic tool if needed
- feed the result back
- repeat until final response

## Where This Lives in the Code

If you want to trace this pattern in the implementation, these are the most
important places to read:

- `src/engllm_chat/tools/chat/workflow.py`
  - owns the top-level turn orchestration
- `src/engllm_chat/tools/chat/registry.py`
  - maps tool requests to deterministic tool execution
- `src/engllm_chat/llm/openai_compatible.py`
  - public provider adapter façade
- `src/engllm_chat/llm/_openai_compatible/serialization.py`
  - builds the structured action schema and serializes messages
- `src/engllm_chat/llm/_openai_compatible/parsing.py`
  - validates and extracts provider responses
- `src/engllm_chat/domain/models.py`
  - defines the shared message, tool, and final-response models

## Important Design Rules

When working in this codebase, keep these rules in mind:

- prompts belong under `prompts/chat/`
- deterministic tools belong under `core/chat/`
- provider-specific behavior belongs under `llm/`
- workflow orchestration belongs under `tools/chat/`
- tool results are plain chat messages, not provider-native tool result objects
- free-form unvalidated model output is not accepted

Those rules are what make the architecture teachable and testable.

## Common Misunderstandings

### “Why not just let the provider handle tools?”

Because this project wants one predictable orchestration pattern across
providers, including Ollama and the mock provider.

### “Is the model directly reading files?”

No. The model can only request deterministic read-only tools. The application
does the actual filesystem work.

### “Is this streaming?”

Not in the provider layer. The UI simulates incremental display, but the chat
workflow itself is built around non-streaming structured responses.

## Suggested Next Reading

After this document, the most useful next docs are:

- `docs/ARCHITECTURE.md`
- `docs/DETERMINISTIC_TOOLS.md`
- `docs/CONTRIBUTING_CHAT_ARCHITECTURE.md`
