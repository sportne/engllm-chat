from __future__ import annotations

from pathlib import Path

from engllm_chat._pex_build import (
    DEFAULT_PYTHON_SHEBANG,
    build_artifact_name,
    build_pex_command,
    build_wheel_command,
    build_wheelhouse_command,
    normalize_platform_tag,
    read_project_version,
)


def test_read_project_version_matches_repo_metadata() -> None:
    project_root = Path(__file__).resolve().parents[1]
    version = read_project_version(project_root / "pyproject.toml")
    assert version == "0.1.0"


def test_build_artifact_name_includes_version_python_and_platform() -> None:
    artifact_name = build_artifact_name(
        "0.1.0",
        python_version=(3, 11),
        platform_tag="linux-x86_64",
    )

    assert artifact_name == "engllm-chat-0.1.0-py311-linux_x86_64.pex"


def test_build_commands_include_expected_packaging_flags(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    wheelhouse_dir = tmp_path / "wheelhouse"
    wheel_path = dist_dir / "engllm_chat-0.1.0-py3-none-any.whl"
    output_path = dist_dir / "engllm-chat-0.1.0-py311-linux_x86_64.pex"

    wheel_command = build_wheel_command(".venv/bin/python", dist_dir)
    wheelhouse_command = build_wheelhouse_command(
        ".venv/bin/python",
        wheel_path,
        wheelhouse_dir,
    )
    pex_command = build_pex_command(
        ".venv/bin/python",
        wheel_paths=[
            wheelhouse_dir / "dependency.whl",
            wheelhouse_dir / "engllm_chat-0.1.0-py3-none-any.whl",
        ],
        output_path=output_path,
    )

    assert wheel_command == [
        ".venv/bin/python",
        "-m",
        "build",
        "--wheel",
        "--no-isolation",
        "--outdir",
        str(dist_dir),
    ]
    assert wheelhouse_command == [
        ".venv/bin/python",
        "-m",
        "pip",
        "wheel",
        "--wheel-dir",
        str(wheelhouse_dir),
        str(wheel_path),
    ]
    assert pex_command == [
        ".venv/bin/python",
        "-m",
        "pex",
        str(wheelhouse_dir / "dependency.whl"),
        str(wheelhouse_dir / "engllm_chat-0.1.0-py3-none-any.whl"),
        "-c",
        "engllm-chat",
        "--venv",
        "--python-shebang",
        DEFAULT_PYTHON_SHEBANG,
        "-o",
        str(output_path),
    ]


def test_normalize_platform_tag_replaces_filename_unfriendly_characters() -> None:
    assert normalize_platform_tag("macosx-14.0-arm64") == "macosx_14_0_arm64"
