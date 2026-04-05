# TASKS

This file tracks the main follow-up work for `engllm-chat` after the recent modularity refactors.

## Next up

- [x] Complete the major documentation and teaching pass for less-experienced developers
  - [x] Add architecture and onboarding docs that show how one chat turn moves through the system end to end.
  - [x] Add a reference-implementation guide that explains the bigger design lessons behind the project.
  - [x] Document the schema-first action loop and the deterministic tool layer in plain language.
  - [x] Add contributor-facing architecture and extension guides.
  - [x] Add testing/debugging guidance and a dedicated chat config reference.
  - [x] Add a glossary for project terminology used across the docs.
  - [x] Add teaching-oriented code comments in the main workflow/provider/tool boundaries.
  - [x] Polish README and cross-links so the docs form a clearer learning path.

## Product and behavior follow-ups

- [x] Tighten the probe utility output so it distinguishes required runtime APIs from optional or extra OpenAI-compatible capabilities.

- [ ] Preserve question-driven evidence selection
  - Prefer evidence sources based on the user's question and the repository contents.
  - Do not introduce a blanket preference for code over docs, config, data files, or other repository artifacts.

- [ ] Decide later whether smoke/probe should stay as packaged modules or become script-first utilities
  - `scripts/chat_smoke.py`, `scripts/ollama_chat_smoke.py`, and `scripts/openai_api_probe.py` already provide script entrypoints.
  - Keep the current packaged implementations under `src/engllm_chat/` for now because tests import them directly, the CLI imports `probe_openai_api`, and docs/Make targets already point at the packaged module path.
  - If we revisit this later, do it as a separate packaging and entrypoint cleanup rather than as part of the modularity roadmap.

- [x] Do integrated end-to-end testing of the Textual chat client and user experience
  - Exercise the actual Textual app with the full workflow, provider layer, prompts, and deterministic tools wired together.
  - Test real chat sessions against local and hosted providers where practical, not just isolated workflow helpers.
  - Review the overall UX for responsiveness, interruptions, status messaging, transcript readability, and error recovery.
  - Identify rough edges that only appear in the full app experience and fix them as a dedicated UX-quality pass.
  - Add or improve regression coverage where the integrated client behavior can be tested deterministically.

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
- [x] Split the chat workflow into internal helper modules while keeping `workflow.py` as the public facade.
- [x] Split the OpenAI-compatible adapter into internal helper modules while keeping `openai_compatible.py` as the public facade.
- [x] Split the domain model layer into grouped internal modules while keeping `domain/models.py` as the public facade.
- [x] Added a repeatable chat smoke test with verbose LLM logging support.
- [x] Generalized the smoke test path so it can target hosted providers such as Gemini.
- [x] Completed the contributor-reference and teaching-oriented documentation pass.
- [x] Tightened the probe utility output around `engllm-chat` runtime requirements.
- [x] Removed the software-specific code-bias prompt idea and replaced it with a question-driven evidence-selection principle.
- [x] Added stronger integrated Textual chat-client coverage for startup, busy-state, continuation, interruption, and recovery flows.
