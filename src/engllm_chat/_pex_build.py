"""Helpers for building a single-file .pex distribution artifact."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import sysconfig
import tomllib
from pathlib import Path

PROJECT_DISTRIBUTION_NAME = "engllm-chat"
DEFAULT_PYTHON_SHEBANG = "/usr/bin/env python3"


def read_project_version(pyproject_path: Path) -> str:
    """Read the project version from pyproject.toml."""
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    return str(pyproject["project"]["version"])


def normalize_platform_tag(platform_tag: str | None = None) -> str:
    """Build a filename-safe tag from the current platform."""
    raw_platform = platform_tag or sysconfig.get_platform()
    return re.sub(r"[^A-Za-z0-9]+", "_", raw_platform).strip("_")


def build_artifact_name(
    version: str,
    *,
    python_version: tuple[int, int] | None = None,
    platform_tag: str | None = None,
) -> str:
    """Name the artifact for the current Python runtime and platform."""
    major, minor = python_version or sys.version_info[:2]
    resolved_platform = normalize_platform_tag(platform_tag)
    return (
        f"{PROJECT_DISTRIBUTION_NAME}-{version}-py{major}{minor}-"
        f"{resolved_platform}.pex"
    )


def build_wheel_command(python_executable: str, dist_dir: Path) -> list[str]:
    return [
        python_executable,
        "-m",
        "build",
        "--wheel",
        "--no-isolation",
        "--outdir",
        str(dist_dir),
    ]


def build_wheelhouse_command(
    python_executable: str,
    wheel_path: Path,
    wheelhouse_dir: Path,
) -> list[str]:
    return [
        python_executable,
        "-m",
        "pip",
        "wheel",
        "--wheel-dir",
        str(wheelhouse_dir),
        str(wheel_path),
    ]


def build_pex_command(
    python_executable: str,
    *,
    wheel_paths: list[Path],
    output_path: Path,
    python_shebang: str = DEFAULT_PYTHON_SHEBANG,
) -> list[str]:
    if not wheel_paths:
        raise ValueError("wheel_paths must include at least one wheel")
    return [
        python_executable,
        "-m",
        "pex",
        *[str(wheel_path) for wheel_path in sorted(wheel_paths)],
        "-c",
        "engllm-chat",
        "--venv",
        "--python-shebang",
        python_shebang,
        "-o",
        str(output_path),
    ]


def find_project_wheel(dist_dir: Path, version: str) -> Path:
    normalized_name = PROJECT_DISTRIBUTION_NAME.replace("-", "_")
    matches = sorted(dist_dir.glob(f"{normalized_name}-{version}-*.whl"))
    if not matches:
        raise FileNotFoundError(
            f"Could not find a built wheel for {PROJECT_DISTRIBUTION_NAME} {version} "
            f"in {dist_dir}"
        )
    return matches[-1]


def _run_checked(command: list[str], *, cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def smoke_test_pex_artifact(
    artifact_path: Path,
    *,
    project_root: Path,
    python_executable: str,
) -> None:
    _run_checked([python_executable, str(artifact_path), "--help"], cwd=project_root)
    _run_checked(
        [python_executable, str(artifact_path), "probe-openai-api", "--help"],
        cwd=project_root,
    )


def build_pex_artifact(
    *,
    project_root: Path,
    python_executable: str = sys.executable,
    dist_dir: Path | None = None,
    build_dir: Path | None = None,
) -> Path:
    project_root = project_root.resolve()
    resolved_dist_dir = (dist_dir or project_root / "dist").resolve()
    resolved_build_dir = (build_dir or project_root / "build").resolve()
    wheelhouse_dir = resolved_build_dir / "wheelhouse"
    pyproject_path = project_root / "pyproject.toml"
    version = read_project_version(pyproject_path)
    artifact_path = resolved_dist_dir / build_artifact_name(version)

    resolved_dist_dir.mkdir(parents=True, exist_ok=True)
    if wheelhouse_dir.exists():
        shutil.rmtree(wheelhouse_dir)
    wheelhouse_dir.mkdir(parents=True, exist_ok=True)

    _run_checked(
        build_wheel_command(python_executable, resolved_dist_dir), cwd=project_root
    )
    wheel_path = find_project_wheel(resolved_dist_dir, version)
    _run_checked(
        build_wheelhouse_command(python_executable, wheel_path, wheelhouse_dir),
        cwd=project_root,
    )
    built_project_wheel = wheelhouse_dir / wheel_path.name
    if not built_project_wheel.exists():
        shutil.copy2(wheel_path, built_project_wheel)
    wheel_paths = sorted(wheelhouse_dir.glob("*.whl"))
    _run_checked(
        build_pex_command(
            python_executable,
            wheel_paths=wheel_paths,
            output_path=artifact_path,
        ),
        cwd=project_root,
    )
    return artifact_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a single-file .pex artifact for engllm-chat."
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Repository root containing pyproject.toml (default: current directory).",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use for build, wheelhouse, and pex steps.",
    )
    parser.add_argument(
        "--dist-dir",
        default=None,
        help="Override the output dist directory (default: <project-root>/dist).",
    )
    parser.add_argument(
        "--build-dir",
        default=None,
        help="Override the working build directory (default: <project-root>/build).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="After building, verify the packaged CLI help paths still work.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    artifact_path = build_pex_artifact(
        project_root=Path(args.project_root),
        python_executable=args.python,
        dist_dir=Path(args.dist_dir) if args.dist_dir else None,
        build_dir=Path(args.build_dir) if args.build_dir else None,
    )
    if args.smoke:
        smoke_test_pex_artifact(
            artifact_path,
            project_root=Path(args.project_root).resolve(),
            python_executable=args.python,
        )
    print(artifact_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
