"""Runtime selection remains a non-authorizing adapter and preserves demo callers."""
from pathlib import Path
import importlib.util
import subprocess
import sys

import pytest

from src.interfaces import runtime_selection as selection

ROOT = Path(__file__).resolve().parents[2]


def test_ready_interpreter_returns_without_subprocess(monkeypatch):
    called = []
    monkeypatch.setattr(selection, "runtime_ready", lambda executable: executable == sys.executable)
    monkeypatch.setattr(selection, "interpreter_candidates", lambda root: called.append(root) or [])
    assert selection.ensure_runtime(ROOT, ROOT / "scripts/launch_exploratory_demo.py") is None
    assert called == []


def test_fallback_preserves_command_arguments_and_working_directory(monkeypatch):
    calls = []
    monkeypatch.setattr(selection, "runtime_ready", lambda executable: str(executable) == "rl-python")
    monkeypatch.setattr(selection, "interpreter_candidates", lambda root: ["rl-python"])
    monkeypatch.setattr(
        selection.subprocess, "call",
        lambda command, **kwargs: calls.append((command, kwargs)) or 23,
    )
    command = ["run_inference.py", "--model", "weights.zip"]
    assert selection.ensure_runtime(ROOT, ROOT / "scripts/launch_exploratory_demo.py", command) == 23
    assert calls == [(["rl-python", "-B", *command], {"cwd": ROOT})]


def test_missing_interpreter_fails_without_launch(monkeypatch):
    monkeypatch.setattr(selection, "runtime_ready", lambda executable: False)
    monkeypatch.setattr(selection, "interpreter_candidates", lambda root: [])
    monkeypatch.setattr(
        selection.subprocess, "call",
        lambda *args, **kwargs: pytest.fail("fallback must not launch"),
    )
    with pytest.raises(ImportError, match="No Python interpreter"):
        selection.ensure_runtime(ROOT, ROOT / "scripts/run_inference.py")


def test_launcher_keeps_historical_runtime_hook_contract(monkeypatch):
    launcher_path = ROOT / "scripts/launch_exploratory_demo.py"
    spec = importlib.util.spec_from_file_location("launcher_runtime_regression", launcher_path)
    launcher = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(launcher)
    monkeypatch.setattr(launcher, "_runtime_ready", lambda executable: str(executable) == "rl-python")
    monkeypatch.setattr(launcher, "_interpreter_candidates", lambda: ["rl-python"])
    calls = []
    monkeypatch.setattr(
        launcher.subprocess, "call",
        lambda command, **kwargs: calls.append((command, kwargs)) or 0,
    )
    command = ["run_inference.py", "--model", "trained-final.zip"]
    assert launcher.ensure_runtime(command) == 0
    assert calls == [(["rl-python", "-B", *command], {"cwd": ROOT})]


def test_inference_rejects_invalid_arguments_before_loading_model():
    path = ROOT / "scripts/run_inference.py"
    for invalid in ("0", "33"):
        result = subprocess.run(
            [sys.executable, str(path), "--episodes", invalid],
            cwd=ROOT, capture_output=True, text=True, timeout=20,
        )
        assert result.returncode == 2
        assert "episodes must be between 1 and 32" in result.stderr


def test_inference_no_longer_imports_a_sibling_executable():
    source = (ROOT / "scripts/run_inference.py").read_text(encoding="utf-8")
    assert "from launch_exploratory_demo import" not in source
    grouped = (ROOT / "scripts/inference/run_policy.py").read_text(encoding="utf-8")
    assert "from src.interfaces.runtime_selection import ensure_runtime" in grouped
    assert "from scripts.inference import run_policy as _implementation" in source
