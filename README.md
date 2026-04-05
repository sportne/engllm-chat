# engllm-chat

`engllm-chat` is a terminal chat tool for exploring one directory root with a
safe, read-only toolset. It provides a Textual chat UI, deterministic
filesystem tools, support for local and hosted providers, and a small
OpenAI-compatible API probe utility.

## Features

- Read-only directory chat rooted at one chosen path
- Deterministic file listing, search, and bounded read tools
- Session-aware workflow with continuation boundaries and context trimming
- Chat providers: `ollama`, `mock`, `openai`, `xai`, `anthropic`, `gemini`
- Textual interactive UI
- OpenAI-compatible endpoint probing command

## Installation

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .[dev]
```

## Quick start

Create a chat config file:

```yaml
llm:
  provider: ollama
  model_name: qwen2.5:7b-instruct
  api_base_url: http://127.0.0.1:11434
```

Launch chat:

```bash
engllm-chat interactive . --config chat.yaml
```

Probe an OpenAI-compatible endpoint:

```bash
engllm-chat probe-openai-api --base-url http://localhost:11434/v1 --api-key dummy
```

Run a repeatable chat workflow smoke test:

```bash
make smoke-chat
```

For Ollama specifically, the existing shortcut still works:

```bash
make smoke-ollama-chat
```

Hosted-provider configs can use the same chat workflow with provider-specific
defaults:

- `openai` -> `OPENAI_API_KEY` and `https://api.openai.com/v1`
- `xai` -> `XAI_API_KEY` and `https://api.x.ai/v1`
- `anthropic` -> `ANTHROPIC_API_KEY` and `https://api.anthropic.com/v1/`
- `gemini` -> `GEMINI_API_KEY` and
  `https://generativelanguage.googleapis.com/v1beta/openai/`

You can also override a hosted endpoint explicitly with `--api-base-url`.

## Docs

### Start here

- [docs/CHAT_USAGE.md](docs/CHAT_USAGE.md) for setup, runtime overrides, and
  smoke-test usage
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for a high-level system map and
  one-turn walkthrough
- [docs/CHAT_CONFIG_REFERENCE.md](docs/CHAT_CONFIG_REFERENCE.md) for every
  config setting and practical defaults

### Build a system like this

- [docs/REFERENCE_IMPLEMENTATION.md](docs/REFERENCE_IMPLEMENTATION.md) for the
  bigger design lessons behind the project
- [docs/LLM_STRUCTURED_CALLS.md](docs/LLM_STRUCTURED_CALLS.md) for the
  schema-first provider interaction pattern
- [docs/DETERMINISTIC_TOOLS.md](docs/DETERMINISTIC_TOOLS.md) for the read-only
  filesystem tool layer and its safety rules
- [docs/GLOSSARY.md](docs/GLOSSARY.md) for project terminology

### Contribute or extend it

- [docs/CONTRIBUTING_CHAT_ARCHITECTURE.md](docs/CONTRIBUTING_CHAT_ARCHITECTURE.md)
  for where changes belong
- [docs/EXTENDING_CHAT_SYSTEM.md](docs/EXTENDING_CHAT_SYSTEM.md) for step-by-step
  extension guidance
- [docs/CHAT_SPEC.md](docs/CHAT_SPEC.md) for the detailed feature and safety
  specification

### Run and debug it

- [docs/TESTING_AND_DEBUGGING.md](docs/TESTING_AND_DEBUGGING.md) for tests,
  smoke runs, and debugging guidance
- [docs/CHAT_PLAN.md](docs/CHAT_PLAN.md) for historical implementation context
