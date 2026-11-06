import importlib.util
from pathlib import Path


def test_runtime_fallback_relaunches_inference_without_training(monkeypatch):
    path = Path(__file__).resolve().parents[2]/'f1rl/cli/training/exploratory_demo_launcher_v4.py'
    spec = importlib.util.spec_from_file_location('demo_launcher', path)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    monkeypatch.setattr(launcher, '_runtime_ready', lambda executable: str(executable) == 'rl-python')
    monkeypatch.setattr(launcher, '_interpreter_candidates', lambda: ['rl-python'])
    calls = []
    monkeypatch.setattr(launcher.subprocess, 'call', lambda command, **kwargs: calls.append(command) or 0)
    command = ['run_inference.py', '--model', 'trained-final.zip']
    assert launcher.ensure_runtime(command) == 0
    assert calls == [['rl-python', '-B', *command]]
