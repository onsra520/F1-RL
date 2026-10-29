"""Grouped demo and packaging CLIs retain their argument and output contracts."""
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("module,group,action,expected", [
    ("scripts.inference.package_handoff", "inference", "package-handoff", ("--run", "--destination")),
    ("scripts.inference.run_policy", "inference", "run-policy", ("--episodes", "--model")),
    ("scripts.training.local_simulation", "training", "local-simulation", ("--candidate", "--tuning-seed-index")),
])
def test_cli_help_parity(module, group, action, expected):
    grouped = subprocess.run([sys.executable, "-m", module, "--help"],
                             cwd=ROOT, capture_output=True, text=True, timeout=30)
    routed = subprocess.run([sys.executable, "-m", "scripts.workflows", group, action, "--help"],
                            cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert grouped.returncode == routed.returncode == 0, (grouped.stderr, routed.stderr)
    assert grouped.stdout == routed.stdout
    assert all(option in grouped.stdout for option in expected)


def test_inference_rejects_invalid_episode_count():
    result = subprocess.run([sys.executable, "-m", "scripts.inference.run_policy",
                             "--episodes", "0"], cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert "episodes must be between 1 and 32" in result.stderr


def test_grouped_entrypoints_import_shared_src_implementations():
    from scripts.inference import package_handoff, run_policy
    from scripts.training import local_simulation
    assert package_handoff.ROOT == local_simulation.ROOT == run_policy.ROOT == ROOT
    assert callable(package_handoff.save)
    assert callable(package_handoff.main)
    assert callable(local_simulation.main)
    assert callable(run_policy.main)


def test_handoff_rejects_missing_inputs_without_outputs(tmp_path):
    from src.interfaces.demo_handoff import main
    dest = tmp_path / "package"
    with pytest.raises(FileNotFoundError):
        main(ROOT, ["--run", str(tmp_path / "missing"), "--destination", str(dest)],
             torch_factory=lambda: pytest.fail("invalid input must not allocate Torch"))
    assert not dest.exists()


def test_demo_handoff_copies_grouped_inference_runtime(tmp_path):
    import shutil
    from src.interfaces.demo_handoff import demo_runtime_scripts

    portable = tmp_path / "demo"
    portable.mkdir()
    shutil.copytree(ROOT / "src", portable / "src", ignore=shutil.ignore_patterns("__pycache__"))
    members = demo_runtime_scripts(ROOT)
    assert len(members) == 6 and all(path.is_file() for path in members)
    for path in members:
        dest = portable / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    command = [sys.executable, "-m", "scripts.inference.run_policy", "--help"]
    check = subprocess.run(command, cwd=portable, capture_output=True, text=True, timeout=30)
    assert check.returncode == 0, check.stderr
    assert "--model" in check.stdout
