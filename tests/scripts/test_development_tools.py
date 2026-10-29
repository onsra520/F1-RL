"""Non-training regression for modularized development tools."""
from pathlib import Path
import importlib.util
import subprocess
import sys
import json

import numpy as np
import pytest

from src.evaluation.driving_comparison import SEEDS, metrics, improved
from src.evaluation.driving_diagnostics import diagnostic_action

ROOT = Path(__file__).resolve().parents[2]


def _sample(progress, *, success=False, valid=True, reason="step_limit"):
    return {
        "success": success, "valid_sample": valid, "error": None,
        "progress": progress, "collisions": 0,
        "completed_checkpoint_ids": ["first"], "outcome": {"reason": reason},
    }


def test_metric_aggregation_preserves_tuning_contract():
    rows = [_sample(0.2), _sample(0.6, success=True, reason="finish")]
    value = metrics(rows)
    assert value["episodes"] == 2
    assert value["successes"] == 1
    assert value["all_valid"] is True
    assert value["mean_progress"] == pytest.approx(0.4)
    assert value["median_progress"] == pytest.approx(0.4)
    assert value["worst_progress"] == pytest.approx(0.2)
    assert value["checkpoints_passed"] == 2
    assert value["terminal_reasons"] == {"step_limit": 1, "finish": 1}


def test_comparison_acceptance_remains_prespecified_and_fail_closed():
    original = metrics([_sample(0.2), _sample(0.3)])
    candidate = metrics([_sample(0.3), _sample(0.5)])
    assert improved(original, candidate) is True
    assert improved(original, metrics([_sample(0.29), _sample(0.29)])) is False
    invalid = metrics([_sample(0.3, valid=False), _sample(0.5)])
    assert improved(original, invalid) is False
    assert SEEDS == (10000, 10001, 11000, 11001, 11002, 11003, 11004, 11005)


def test_scripted_diagnostic_action_is_stable():
    observation = np.zeros(13, dtype=np.float32)
    observation[12] = 1.0
    action = diagnostic_action(observation, 0.0)
    np.testing.assert_array_equal(action, np.array([0.0, 0.6, 0.0], dtype=np.float32))
    assert action.dtype == np.float32


@pytest.mark.parametrize("legacy,group,alias", [
    ("compare_driving.py", "evaluation", "compare-driving"),
    ("diagnose_driving.py", "diagnostics", "diagnose-driving"),
])
def test_legacy_and_versionless_development_help_are_identical(legacy, group, alias):
    old = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / legacy), "--help"],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    modern = subprocess.run(
        [sys.executable, "-m", "scripts.workflows", group, alias, "--help"],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    assert old.returncode == modern.returncode == 0, (old.stderr, modern.stderr)
    assert old.stdout == modern.stdout
    assert "--output" in old.stdout


def test_cuda_probe_has_stable_keys_and_legacy_wrapper(monkeypatch):
    from src.interfaces import hardware_probe
    class MockVersion:
        cuda = None
    class MockTorch:
        version = MockVersion()
        class cuda:
            @staticmethod
            def is_available():
                return False
    assert hardware_probe.capture_hardware(MockTorch) == {
        "cuda_runtime": None, "cuda_available": False, "gpu_devices": [],
    }
    path = ROOT / "scripts" / "wp7_capture_gpu.py"
    spec = importlib.util.spec_from_file_location("legacy_gpu_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.main)


def test_gpu_probe_subprocess_output_matches_versionless_alias():
    old = subprocess.run(
        [sys.executable, str(ROOT / "scripts/wp7_capture_gpu.py")],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    new = subprocess.run(
        [sys.executable, "-m", "scripts.workflows", "diagnostics", "probe-gpu"],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    assert old.returncode == new.returncode == 0, (old.stderr, new.stderr)
    assert json.loads(old.stdout) == json.loads(new.stdout)
