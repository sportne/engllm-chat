#!/usr/bin/env python3
"""Build or smoke-check a single-file .pex artifact for engllm-chat."""

from __future__ import annotations

from engllm_chat._pex_build import main


if __name__ == "__main__":
    raise SystemExit(main())
