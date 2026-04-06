#!/usr/bin/env python3
"""Build and smoke-check the engllm-chat .pex artifact."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from engllm_chat._pex_build import build_pex_artifact


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and smoke-check the engllm-chat .pex artifact."
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Repository root containing pyproject.toml (default: current directory).",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use for build and smoke commands.",
    )
    return parser


def _run_checked(command: list[str], *, cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    project_root = Path(args.project_root).resolve()
    artifact_path = build_pex_artifact(
        project_root=project_root,
        python_executable=args.python,
    )

    _run_checked([args.python, str(artifact_path), "--help"], cwd=project_root)
    _run_checked(
        [args.python, str(artifact_path), "probe-openai-api", "--help"],
        cwd=project_root,
    )
    print(artifact_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
