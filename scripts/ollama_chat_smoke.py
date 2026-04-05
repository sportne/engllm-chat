#!/usr/bin/env python3
"""Thin wrapper for the backward-compatible Ollama smoke entry point."""

from engllm_chat.smoke_ollama_chat import main


if __name__ == "__main__":
    raise SystemExit(main())
