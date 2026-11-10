import json
from hashlib import sha256
import importlib.util
from pathlib import Path

import pytest

from f1rl.local_demo.real_ppo_benchmark import (
    _execution_intervals,
    _require_prior_passes,
    _resolve_artifact_root,
    _sample_skew_seconds,
    _sequential_intervals_are_disjoint,
    _summarize_process_cpu,
    _validate_timing,
    _verify_run_manifest,
)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_sequential_benchmark_rejects_missing_or_forged_smoke_evidence(tmp_path):
    with pytest.raises(RuntimeError, match="requires a passing real PPO smoke"):
        _require_prior_passes(tmp_path, tmp_path, "sequential", "source-a", "config-a", {})

    attempt = tmp_path / "smoke-attempt"
    _write(attempt / "attempt.json", {"mode": "smoke", "source_commit": "source-a"})
    _write(attempt / "result.json", {
        "status": "SMOKE_INTEGRATION_PASS", "source_commit": "source-a",
        "attempt_manifest_sha256": sha256((attempt / "attempt.json").read_bytes()).hexdigest(),
    })
    with pytest.raises(RuntimeError, match="requires a passing real PPO smoke"):
        _require_prior_passes(tmp_path, tmp_path, "sequential", "source-a", "config-a", {})

    newer = tmp_path / "smoke-newer"
    _write(newer / "attempt.json", {"mode": "smoke", "source_commit": "source-a"})
    _write(newer / "result.json", {
        "status": "SMOKE_INTEGRATION_FAILED", "source_commit": "source-a",
        "attempt_manifest_sha256": sha256((newer / "attempt.json").read_bytes()).hexdigest(),
    })
    with pytest.raises(RuntimeError, match="requires a passing real PPO smoke"):
        _require_prior_passes(tmp_path, tmp_path, "sequential", "source-a", "config-a", {})


def test_parallel_benchmark_requires_matching_sequential_pass(tmp_path):
    with pytest.raises(RuntimeError, match="requires a passing sequential result"):
        _require_prior_passes(tmp_path, tmp_path, "parallel", "source-a", "config-a", {})

    attempt = tmp_path / "sequential-attempt"
    _write(attempt / "attempt.json", {"mode": "sequential", "source_commit": "source-a"})
    report = attempt / "result.json"
    _write(report, {"status": "ENGINEERING_PASS", "source_commit": "source-a",
                    "benchmark_profile_sha256": "config-a",
                    "attempt_manifest_sha256": sha256((attempt / "attempt.json").read_bytes()).hexdigest()})
    with pytest.raises(RuntimeError, match="requires a passing sequential result"):
        _require_prior_passes(tmp_path, tmp_path, "parallel", "source-a", "config-a", {})


def test_overlap_uses_child_exit_times_and_excludes_cleanup(tmp_path):
    jobs = [{"model_id": "A"}, {"model_id": "B"}]
    outcomes = {
        "A": {"process_started_at": "2026-11-10T00:00:00+00:00",
              "process_exit_observed_at": "2026-11-10T00:00:04+00:00", "cleanup_seconds": 5},
        "B": {"process_started_at": "2026-11-10T00:00:08+00:00",
              "process_exit_observed_at": "2026-11-10T00:00:10+00:00", "cleanup_seconds": 0},
    }
    intervals, overlap = _execution_intervals(jobs, outcomes)
    assert overlap == 0
    assert intervals["A"]["exit_observed_at"] == outcomes["A"]["process_exit_observed_at"]

    outcomes["B"]["process_started_at"] = "2026-11-10T00:00:03+00:00"
    _, overlap = _execution_intervals(jobs, outcomes)
    assert overlap == 1
    assert not _sequential_intervals_are_disjoint(outcomes)
    outcomes["B"]["process_started_at"] = "2026-11-10T00:00:10+00:00"
    outcomes["B"]["process_exit_observed_at"] = "2026-11-10T00:00:12+00:00"
    assert _sequential_intervals_are_disjoint(outcomes)


def test_process_cpu_summary_sums_jobs_and_reports_average_host_utilization():
    summary = _summarize_process_cpu({
        "A": {"resource_usage": {"final_process_tree_cpu_seconds": 6.0}},
        "B": {"resource_usage": {"final_process_tree_cpu_seconds": 4.0}},
    }, elapsed_seconds=10.0, logical_processors=2)
    assert summary == {"total_process_tree_cpu_seconds": 10.0,
                       "average_process_tree_cpu_percent_of_host": 50.0}


def test_artifact_root_resolves_before_repository_containment_check(tmp_path):
    repository = tmp_path / "repo"
    external = tmp_path / "external"
    repository.mkdir()
    external.mkdir()
    assert _resolve_artifact_root(repository, external).parent == tmp_path
    with pytest.raises(ValueError, match="outside the repository"):
        _resolve_artifact_root(repository, repository / "artifacts")
    junction = tmp_path / "repo-link"
    try:
        junction.symlink_to(repository, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks are unavailable: {error}")
    with pytest.raises(ValueError, match="outside the repository"):
        _resolve_artifact_root(repository, junction / "artifacts")


def test_concurrent_process_sample_rejects_process_exiting_between_samples():
    assert _sample_skew_seconds([100.0, 100.05]) <= 0.1
    assert _sample_skew_seconds([100.0, 101.0]) > 0.1
    assert _sample_skew_seconds([100.0]) is None


@pytest.mark.parametrize("timing", [
    {},
    {"collect_rollouts_seconds": 1, "optimizer_updates_seconds": 0},
    {"collect_rollouts_seconds": float("nan"), "optimizer_updates_seconds": 1},
])
def test_training_timing_rejects_missing_zero_or_nonfinite_values(timing):
    with pytest.raises(ValueError, match="must be finite and positive"):
        _validate_timing(timing)


@pytest.mark.parametrize(("mode", "status"), [
    ("smoke", "SMOKE_INTEGRATION_PASS"),
    ("sequential", "ENGINEERING_PASS"),
    ("parallel", "ENGINEERING_PASS"),
])
def test_benchmark_cli_returns_success_for_each_stage_pass(tmp_path, mode, status):
    path = Path(__file__).parents[2] / "scripts/wp3_real_ppo_benchmark.py"
    spec = importlib.util.spec_from_file_location("wp3_real_ppo_benchmark_cli", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    cli.run_real_ppo_benchmark = lambda _root, _output, _mode: {"status": status}
    assert cli.main([mode, "--output-root", str(tmp_path)]) == 0


def test_run_manifest_seed_is_checked_against_independent_expected_bundle(tmp_path):
    root = Path(__file__).parents[2]
    output = tmp_path / "run"
    seed_bundle = {"ppo": 0, "sampling": 0, "training_reset": 1000}
    manifest = {"source_commit": "source", "profile": "benchmark",
                "benchmark_profile_sha256": "profile", "candidate": {"candidate_id": "A"},
                "training_seed_bundle": seed_bundle, "split": "TRAIN",
                "heldout_access": False, "scientific_evidence": False,
                "scientific_training_authorized": False}
    manifest["sha256"] = sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                                 allow_nan=False).encode()).hexdigest()
    _write(output / "run-manifest.json", manifest)
    assert _verify_run_manifest(output, source_commit="source", profile="benchmark",
                                profile_hash="profile", candidate="A", seed_bundle=seed_bundle)
    manifest["training_seed_bundle"] = {**seed_bundle, "ppo": 9}
    manifest.pop("sha256")
    manifest["sha256"] = sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                                 allow_nan=False).encode()).hexdigest()
    _write(output / "run-manifest.json", manifest)
    with pytest.raises(ValueError, match="benchmark identity"):
        _verify_run_manifest(output, source_commit="source", profile="benchmark",
                             profile_hash="profile", candidate="A", seed_bundle=seed_bundle)
