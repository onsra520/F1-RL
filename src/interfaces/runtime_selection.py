"""Select an existing Python runtime for local scripts without authorizing execution.

Runtime selection is independent of training and scientific admission. It only
relaunches the caller's command when its interpreter lacks declared dependencies.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def runtime_ready(executable: str | Path) -> bool:
    try:
        check = subprocess.run(
            [str(executable), "-B", "-c",
             "import torch, gymnasium, stable_baselines3, yaml, cryptography"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15, check=False,
        )
        return check.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def interpreter_candidates(project_root: Path) -> list[Path]:
    root = Path(project_root)
    if os.name != "nt":
        return []
    directories = [
        root / ".venv" / "Scripts" / "python.exe",
        root.parent / ".venv" / "Scripts" / "python.exe",
    ]
    siblings = [root.parent, root.parent / ".worktrees" / "F1-RL"]
    for base in siblings:
        if base.is_dir():
            directories.extend(sorted(base.glob("*/.venv/Scripts/python.exe")))
    return directories


def ensure_runtime(
    project_root: Path,
    launcher_path: Path,
    command: list[str] | None = None,
    *,
    runtime_check=None,
    candidate_provider=None,
    subprocess_call=None,
) -> int | None:
    """Return None for a ready interpreter or the fallback process exit code.

    The injection points preserve compatibility with the historical demo
    launcher's existing test hooks; they are not permission/admission controls.
    """
    root = Path(project_root)
    ready = runtime_ready if runtime_check is None else runtime_check
    candidates = (
        (lambda: interpreter_candidates(root))
        if candidate_provider is None else candidate_provider
    )
    invoke = subprocess.call if subprocess_call is None else subprocess_call
    if ready(sys.executable):
        return None
    for interpreter in candidates():
        if str(interpreter).lower() != sys.executable.lower() and ready(interpreter):
            print("[ENV] Using installed F1-RL Python:", interpreter, flush=True)
            return invoke(
                [str(interpreter), "-B", *(command or [str(launcher_path)])],
                cwd=root,
            )
    raise ImportError("No Python interpreter with F1-RL RL/science dependencies was found")
