"""Process-local capability used only by the separate LOCAL_DEMO runner."""
from __future__ import annotations

from hashlib import sha256
import inspect
import os
from pathlib import Path
import subprocess
import time
from weakref import WeakKeyDictionary


_CAPABILITIES = WeakKeyDictionary()


def _source_identity(root: Path) -> tuple[str, str]:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                            capture_output=True, text=True).stdout.strip()
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=root, check=True,
                            capture_output=True, text=True).stdout.strip()
    tree = subprocess.run(["git", "rev-parse", "HEAD^{tree}"], cwd=root, check=True,
                          capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"],
                           cwd=root, check=True, capture_output=True, text=True).stdout
    if dirty or branch in ("", "main") or len(commit) != 40 or len(tree) != 40:
        raise ValueError("LOCAL_DEMO requires a clean committed source tree")
    return commit, sha256(b"f1rl-local-demo-git-tree-v1\x00" + bytes.fromhex(tree)).hexdigest()


class LocalDemoStopped(RuntimeError):
    pass


class _LocalDemoSession:
    def __init__(self, root: Path, environment, *, commit: str, tree_sha256: str,
                 deadline: float, max_timesteps: int) -> None:
        if os.name != "nt":
            raise ValueError("LOCAL_DEMO execution is supported on local Windows only")
        if not isinstance(root, Path) or not root.is_dir():
            raise ValueError("a repository root and vector TRAIN environment are required")
        if (type(getattr(environment, "num_envs", None)) is not int
                or not callable(getattr(environment, "get_attr", None))
                or not callable(getattr(environment, "env_method", None))
                or environment.num_envs != 1 or type(max_timesteps) is not int or max_timesteps <= 0
                or not time.monotonic() < deadline):
            raise ValueError("LOCAL_DEMO resource budget is invalid")
        flags = environment.get_attr("requires_scientific_admission")
        if len(flags) != 1 or flags != [True]:
            raise ValueError("LOCAL_DEMO requires its explicit research-simulation environment")
        self.root = root.resolve(strict=True)
        self.environment = environment
        self.commit = commit
        self.tree_sha256 = tree_sha256
        self.deadline = deadline
        self.max_timesteps = max_timesteps
        self._verified_source = False

    def verify_source(self) -> None:
        try:
            commit, tree_sha256 = _source_identity(self.root)
        except ValueError as error:
            raise ValueError("LOCAL_DEMO source identity changed during preflight") from error
        if commit != self.commit or tree_sha256 != self.tree_sha256:
            raise ValueError("LOCAL_DEMO source identity changed during preflight")
        self._verified_source = True

    def check(self, timesteps: int | None = None) -> None:
        if time.monotonic() >= self.deadline:
            raise LocalDemoStopped("LOCAL_DEMO wall-time limit reached")
        if timesteps is not None and timesteps > self.max_timesteps:
            raise LocalDemoStopped("LOCAL_DEMO timestep limit exceeded")


def _issue(root: Path, environment, *, commit: str, tree_sha256: str,
           deadline: float, max_timesteps: int) -> _LocalDemoSession:
    caller = inspect.currentframe().f_back
    if caller is None or caller.f_globals.get("__name__") != "src.local_demo.runner":
        raise PermissionError("LOCAL_DEMO admission can only be issued by its runner")
    session = _LocalDemoSession(root, environment, commit=commit, tree_sha256=tree_sha256,
                                deadline=deadline, max_timesteps=max_timesteps)
    session.verify_source()
    _CAPABILITIES[environment] = session
    return session


def admitted(environment) -> bool:
    """Recognize only a live vector env explicitly issued by LOCAL_DEMO."""
    try:
        session = _CAPABILITIES.get(environment)
    except TypeError:
        # Objects outside the VecEnv admission contract may not support weakrefs.
        return False
    if session is None or session.environment is not environment:
        return False
    session.check()
    if not session._verified_source:
        return False
    return True


def _revoke(environment) -> None:
    _CAPABILITIES.pop(environment, None)
