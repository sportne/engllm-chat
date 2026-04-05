"""Backward-compatible wrapper for the generic smoke test module."""

from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    from engllm_chat.smoke_chat import main as smoke_chat_main

    return smoke_chat_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
