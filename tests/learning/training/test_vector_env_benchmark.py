from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "benchmark_vector_envs.py"


def _run_dummy_benchmark() -> dict[str, object]:
    completed = subprocess.run(
        [
            sys.executable, str(SCRIPT), "--backend", "dummy", "--n-envs", "1",
            "--transitions", "8",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    return json.loads(completed.stdout)


def test_vector_environment_benchmark_is_bounded_and_never_admits_training() -> None:
    result = _run_dummy_benchmark()

    assert result["backend"] == "dummy"
    assert result["n_envs"] == 1
    assert result["transitions"] == 8
    assert result["training_permitted"] is False
    assert result["heldout_permitted"] is False
    assert result["learning_called"] is False
    assert result["checkpoint_created"] is False
    assert isinstance(result["checksum_sha256"], str)
    assert len(result["checksum_sha256"]) == 64


def test_vector_environment_benchmark_checksum_is_deterministic() -> None:
    first = _run_dummy_benchmark()
    second = _run_dummy_benchmark()

    assert first["checksum_sha256"] == second["checksum_sha256"]
