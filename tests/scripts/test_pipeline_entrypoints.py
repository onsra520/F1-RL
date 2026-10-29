"""Active simulation-to-inference entrypoints exclude retired demo launchers."""
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.workflows.catalog import WORKFLOWS

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "module,group,action,expected",
    [
        ("scripts.training.campaign", "training", "campaign", ("--output",)),
        (
            "scripts.inference.run_policy",
            "inference",
            "run-policy",
            ("--episodes", "--model"),
        ),
    ],
)
def test_pipeline_cli_help_parity(module, group, action, expected):
    grouped = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    routed = subprocess.run(
        [sys.executable, "-m", "scripts.workflows", group, action, "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert grouped.returncode == routed.returncode == 0, (grouped.stderr, routed.stderr)
    assert grouped.stdout == routed.stdout
    assert all(option in grouped.stdout for option in expected)


def test_pipeline_domain_modules_import_without_launching_training():
    from src.core.simulation import Simulation
    from src.evaluation.evaluator import evaluate_episode
    from src.rl.gym_environment import F1RLEnv
    from src.training.runner import TrainingRunner

    assert all(callable(stage) for stage in (Simulation, F1RLEnv, TrainingRunner, evaluate_episode))


def test_active_registry_has_no_retired_demo_handoff_or_launcher():
    workflows = {(entry.group, entry.name) for entry in WORKFLOWS}
    assert ("training", "campaign") in workflows
    assert ("training", "local-simulation") in workflows
    assert ("evaluation", "coordinate") in workflows
    assert ("inference", "run-policy") in workflows
    assert ("inference", "package-handoff") not in workflows
    assert ("training", "launch-exploration") not in workflows
    assert not (ROOT / "START-DEMO.cmd").exists()
    assert not (ROOT / "START-DEMO-TRAIN.cmd").exists()
    assert not (ROOT / "scripts/inference/package_handoff.py").exists()
    assert not (ROOT / "scripts/training/launch_demo.py").exists()
