# engllm-chat

`engllm-chat` is a standalone extraction of the EngLLM interactive repository
chat feature. It provides a Textual chat UI over one directory root plus a
small OpenAI-compatible API probe utility.

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

Additional usage details live in [docs/CHAT_USAGE.md](docs/CHAT_USAGE.md).
