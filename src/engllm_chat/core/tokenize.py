"""Small deterministic tokenizer for chat token estimation."""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase lexical units."""

    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text)]
