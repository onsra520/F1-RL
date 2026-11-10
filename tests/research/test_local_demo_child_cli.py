from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).parents[2]
CLI = ROOT / "scripts" / "local_demo.py"
INJECTED_RUNNER = r"""
import json
import os
from pathlib import Path
import runpy
import sys
import f1rl.local_demo.runner as runner

def fake_run_local_demo(root, *, candidate_id, tuning_seed_index, output_directory,
                        dry_run=False, progress_observer=None, profile="tuning"):
    if os.environ.get("F1RL_TEST_CHILD_FAIL") == "1":
        raise RuntimeError("simulated child failure")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=False)
    (output / "child-result.json").write_text(json.dumps({
        "candidate_id": candidate_id,
        "tuning_seed_index": tuning_seed_index,
        "profile": profile,
    }), encoding="utf-8")
    return {"state": {"status": "SMOKE_INTEGRATION_PASS" if profile == "smoke"
                       else "learning_completed"}}

runner.run_local_demo = fake_run_local_demo
script = sys.argv.pop(1)
sys.argv[0] = script
entry = runpy.run_path(script)
raise SystemExit(entry["main"]())
"""


def _run_cli(*args: str, fail: bool = False) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.pop("F1RL_TEST_CHILD_FAIL", None)
    if fail:
        env["F1RL_TEST_CHILD_FAIL"] = "1"
    return subprocess.run(
        [sys.executable, "-c", INJECTED_RUNNER, str(CLI), *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_child_cli_runs_tuning_worker_in_isolated_output(tmp_path):
    first = tmp_path / "A-seed0"
    second = tmp_path / "B-seed1"

    first_result = _run_cli(
        "train", "--candidate", "A", "--tuning-seed-index", "0", "--output", str(first)
    )
    second_result = _run_cli(
        "train", "--candidate", "B", "--tuning-seed-index", "1", "--output", str(second)
    )

    assert first_result.returncode == second_result.returncode == 0
    assert json.loads((first / "child-result.json").read_text(encoding="utf-8")) == {
        "candidate_id": "A", "tuning_seed_index": 0, "profile": "tuning",
    }
    assert json.loads((second / "child-result.json").read_text(encoding="utf-8")) == {
        "candidate_id": "B", "tuning_seed_index": 1, "profile": "tuning",
    }

    duplicate = _run_cli(
        "train", "--candidate", "A", "--tuning-seed-index", "0", "--output", str(first)
    )
    assert duplicate.returncode != 0
    assert (first / "child-result.json").exists()


def test_child_cli_rejects_invalid_arguments_before_runner(tmp_path):
    output = tmp_path / "must-not-exist"

    result = _run_cli(
        "train", "--candidate", "Z", "--tuning-seed-index", "0", "--output", str(output)
    )

    assert result.returncode == 2
    assert not output.exists()


def test_child_cli_forwards_benchmark_profile(tmp_path):
    output = tmp_path / "benchmark-run"
    result = _run_cli("train", "--candidate", "A", "--tuning-seed-index", "0",
                      "--output", str(output), "--profile", "benchmark")
    assert result.returncode == 0
    assert json.loads((output / "child-result.json").read_text(encoding="utf-8"))["profile"] == "benchmark"


def test_child_cli_accepts_smoke_integration_pass(tmp_path):
    output = tmp_path / "smoke-run"
    result = _run_cli("train", "--candidate", "C", "--tuning-seed-index", "0",
                      "--output", str(output), "--profile", "smoke")
    assert result.returncode == 0
    assert json.loads((output / "child-result.json").read_text(encoding="utf-8"))["profile"] == "smoke"


def test_child_cli_propagates_runner_failure(tmp_path):
    output = tmp_path / "failed-run"

    result = _run_cli(
        "train", "--candidate", "C", "--tuning-seed-index", "0", "--output", str(output),
        fail=True,
    )

    assert result.returncode == 1
    assert "simulated child failure" in result.stderr
    assert not output.exists()
