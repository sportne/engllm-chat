# TASKS

This file tracks the main follow-up work for `engllm-chat` after the recent modularity refactors.

## Next up

- [ ] Further reduce [`workflow.py`](./src/engllm_chat/tools/chat/workflow.py)
  - Extract session context preparation and trimming helpers.
  - Extract token estimation/accounting helpers.
  - Extract turn finalization and continuation/interruption helpers.
  - Consider moving `ChatSessionTurnRunner` into its own module if `workflow.py` still feels dense after helper extraction.
  - Keep exported workflow entrypoints and runtime behavior unchanged.

- [ ] Split [`openai_compatible.py`](./src/engllm_chat/llm/openai_compatible.py) by concern
  - Separate request/message serialization.
  - Separate schema model construction and retry feedback helpers.
  - Separate response parsing and token-usage extraction.
  - Keep `OpenAICompatibleChatLLMClient` as the public entrypoint.
  - Preserve schema-first action handling, retry behavior, verbose logging, and Ollama normalization.

- [ ] Split [`domain/models.py`](./src/engllm_chat/domain/models.py) into grouped internal modules
  - Separate provider/config models.
  - Separate chat protocol/message models.
  - Separate response/citation/token usage models.
  - Keep `domain/models.py` as a compatibility re-export layer so imports do not churn.

- [ ] Split [`smoke_chat.py`](./src/engllm_chat/smoke_chat.py) into smaller helper modules
  - Isolate CLI parser construction.
  - Isolate config resolution.
  - Isolate smoke execution and expectation validation.
  - Isolate output formatting.
  - Keep the current CLI flags and `python -m engllm_chat.smoke_chat` behavior unchanged.

- [ ] Split [`probe_openai_api.py`](./src/engllm_chat/probe_openai_api.py) into internal modules
  - Separate SDK loading and exception classification.
  - Separate probe catalog/spec registration.
  - Separate individual probe implementations.
  - Separate CLI formatting/output.
  - Keep probe behavior stable while making the code easier to navigate.

- [ ] After the modularity passes, add a major documentation and teaching pass
  - Expand the project docs with substantial narrative Markdown aimed at less-experienced developers.
  - Explain the architecture, the schema-first action loop, the provider abstraction, and the deterministic tool layer in plain language.
  - Document how to design structured LLM calls, how tool schemas are defined, and how tool results flow back into the conversation.
  - Add significant explanatory code comments in places where the patterns are important but not obvious from the code alone.
  - Review the code and design for areas that are unnecessarily hard for newer developers to follow, and simplify or reshape them where that would materially improve clarity.
  - Treat the project not only as a usable tool, but also as a reference implementation for building this kind of system.

## Product and behavior follow-ups

- [ ] Tighten the probe utility output so it distinguishes required runtime APIs from optional or extra OpenAI-compatible capabilities.

- [ ] Continue improving prompt/tool-selection behavior so models prefer source-code evidence over docs-heavy evidence for implementation questions.

- [ ] Keep validating the smoke test flow against real providers
  - Re-run the generic smoke test against local Ollama after prompt or workflow changes.
  - Re-run it against hosted providers like Gemini when provider-layer behavior changes.
  - Preserve `--verbose-llm` support and keep the smoke path easy to run manually.

## Maintenance guardrails

- [ ] Preserve the current design constraints during future refactors
  - Do not reintroduce provider-native tool calling.
  - Do not reintroduce provider-side streaming in the chat path.
  - Keep deterministic filesystem behavior provider-agnostic and unit-testable.
  - Keep prompt text centralized under `prompts/chat/`.

- [ ] Keep verification green after each pass
  - Run `pytest`.
  - Run `.venv/bin/python -m mypy src`.
  - Run `.venv/bin/python -m ruff check src tests`.
  - Keep total coverage at or above 90%.

## Recently completed

- [x] Extracted the internal chat tool registry from workflow.
- [x] Split the Textual chat app into presentation, screens, and controller modules.
- [x] Split deterministic chat tool listing logic into a moderate internal `_listing` package while keeping `listing.py` as the public facade.
- [x] Added a repeatable chat smoke test with verbose LLM logging support.
- [x] Generalized the smoke test path so it can target hosted providers such as Gemini.
