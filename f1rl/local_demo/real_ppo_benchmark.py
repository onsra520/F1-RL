from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4

from .admission import _source_identity
from .parallel_campaign import _run_bounded_stage, _run_parallel_child
from .process_resources import sample_process_tree, windows_cpu_load_percent, windows_host_resources
from .runner import _load_benchmark_profile, _load_smoke_profile


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"benchmark evidence is not an object: {path.name}")
    return value


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True,
                          text=True, encoding="utf-8").stdout.strip()


def _source_tree_digest(root: Path) -> str:
    # Match LOCAL_DEMO admission: SHA-256 of domain-separated Git tree object ID.
    return _source_identity(root)[1]


def _resolve_artifact_root(root: Path, path: str | Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("benchmark artifacts must use an absolute path outside the repository")
    resolved = path.resolve()
    if resolved.is_relative_to(root):
        raise ValueError("benchmark artifacts must use an absolute path outside the repository")
    return resolved


def _execution_intervals(jobs: list[dict], outcomes: dict) -> tuple[dict, float]:
    intervals = {}
    for job in jobs:
        outcome = outcomes[job["model_id"]]
        start = datetime.fromisoformat(outcome["process_started_at"])
        end = datetime.fromisoformat(outcome["process_exit_observed_at"])
        if end < start:
            raise ValueError("child process exit time precedes its start time")
        intervals[job["model_id"]] = {"started_at": start.isoformat(),
                                      "exit_observed_at": end.isoformat()}
    overlap = 0.0
    if len(intervals) == 2:
        rows = list(intervals.values())
        overlap = max(0.0, (min(datetime.fromisoformat(row["exit_observed_at"]) for row in rows)
                            - max(datetime.fromisoformat(row["started_at"]) for row in rows)).total_seconds())
    return intervals, overlap


def _sequential_intervals_are_disjoint(outcomes: dict) -> bool:
    jobs = [{"model_id": model_id} for model_id in outcomes]
    if len(jobs) != 2:
        return False
    _, overlap = _execution_intervals(jobs, outcomes)
    return overlap == 0


def _summarize_process_cpu(outcomes: dict, elapsed_seconds: float,
                            logical_processors: int) -> dict:
    cpu_seconds = [row.get("resource_usage", {}).get("final_process_tree_cpu_seconds")
                   for row in outcomes.values()]
    measured = [value for value in cpu_seconds if isinstance(value, (int, float))]
    total = sum(measured) if len(measured) == len(cpu_seconds) else None
    return {"total_process_tree_cpu_seconds": total,
            "average_process_tree_cpu_percent_of_host": (
                100.0 * total / (elapsed_seconds * logical_processors)
                if total is not None and elapsed_seconds > 0 and logical_processors > 0 else None)}


def _sample_skew_seconds(sample_times: list[float]) -> float | None:
    if len(sample_times) != 2:
        return None
    return max(sample_times) - min(sample_times)


def _validate_job(job: dict, result: dict) -> dict:
    output = Path(job["output"])
    state = _read_json(output / "state.json")
    manifest = _read_json(output / "run-manifest.json")
    integrity = _read_json(output / "training-integrity.json")
    telemetry_manifest = _read_json(output / "training-telemetry-manifest.json")
    telemetry_path = output / telemetry_manifest["path"]
    telemetry_hash = sha256(telemetry_path.read_bytes()).hexdigest()
    checkpoint = output / "model-final.zip"
    digest = sha256(checkpoint.read_bytes()).hexdigest()
    if (state.get("status") != "learning_completed"
            or state.get("learning_timesteps") != 4096
            or state.get("optimizer_steps", 0) <= 0
            or manifest.get("profile") != "benchmark"
            or manifest.get("source_commit") != job["source_commit"]
            or manifest.get("benchmark_profile_sha256") != job["benchmark_profile_sha256"]
            or manifest.get("candidate", {}).get("candidate_id") != job["candidate_id"]
            or manifest.get("training_seed_bundle", {}).get("ppo") != job["ppo_seed"]
            or not integrity.get("weights_changed")
            or integrity.get("optimizer_steps") != state.get("optimizer_steps")
            or integrity.get("finite_gradient_steps") != state.get("optimizer_steps")
            or integrity.get("nonzero_gradient_steps") != state.get("optimizer_steps")
            or integrity.get("ppo_update_rows", 0) <= 0
            or integrity.get("metrics_finite") is not True
            or telemetry_manifest.get("complete") is not True
            or telemetry_manifest.get("sha256") != telemetry_hash
            or not checkpoint.is_file()
            or state.get("final_model", {}).get("sha256") != digest):
        raise ValueError("real PPO benchmark job evidence failed identity, learning, or checkpoint validation")
    timing = _read_json(output / "ppo-runtime-timings.json")
    _validate_timing(timing)
    return {"status": "ENGINEERING_PASS", "timesteps": 4096,
            "optimizer_steps": state["optimizer_steps"], "checkpoint_sha256": digest,
            "rollout_seconds": timing["collect_rollouts_seconds"],
            "optimizer_seconds": timing["optimizer_updates_seconds"]}


def _validate_timing(timing: dict) -> None:
    for key in ("collect_rollouts_seconds", "optimizer_updates_seconds"):
        value = timing.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"PPO benchmark timing must be finite and positive: {key}")


def _expected_seed_bundle(root: Path) -> dict:
    config = _read_json(root / "configs/research/simulation-only-ppo-v1.json")
    ppo, sampling, reset = config["seeds"]["tuning"][0]
    return {"ppo": ppo, "sampling": sampling, "training_reset": reset}


def _verify_run_manifest(output: Path, *, source_commit: str, profile: str,
                         profile_hash: str, candidate: str, seed_bundle: dict) -> dict:
    manifest = _read_json(output / "run-manifest.json")
    unsigned = {key: value for key, value in manifest.items() if key != "sha256"}
    expected_hash = sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()
    if (manifest.get("sha256") != expected_hash
            or manifest.get("source_commit") != source_commit
            or manifest.get("profile") != profile
            or manifest.get(f"{profile}_profile_sha256") != profile_hash
            or manifest.get("candidate", {}).get("candidate_id") != candidate
            or manifest.get("training_seed_bundle") != seed_bundle
            or manifest.get("split") != "TRAIN"
            or manifest.get("heldout_access") is not False
            or manifest.get("scientific_evidence") is not False
            or manifest.get("scientific_training_authorized") is not False):
        raise ValueError("run manifest does not match the admitted benchmark identity")
    return manifest


def _verify_smoke_attempt(root: Path, attempt: Path, source_commit: str, seed_bundle: dict) -> bool:
    try:
        header = _read_json(attempt / "attempt.json")
        result = _read_json(attempt / "result.json")
        profile, profile_hash = _load_smoke_profile(root / "configs/research/local-demo-ppo-smoke-v1.json")
        resource_profile, _ = _load_benchmark_profile(root / "configs/research/local-demo-ppo-benchmark-v1.json")
        output = attempt / "C-seed0"
        if (header.get("mode") != "smoke" or header.get("attempt_id") != attempt.name
                or header.get("source_commit") != source_commit
                or header.get("smoke_profile_sha256") != profile_hash
                or header.get("max_processes_per_worker") != resource_profile["max_processes_per_worker"]
                or result.get("status") != "SMOKE_INTEGRATION_PASS"
                or result.get("source_commit") != source_commit
                or result.get("smoke_profile_sha256") != profile_hash
                or result.get("attempt_manifest_sha256") != sha256((attempt / "attempt.json").read_bytes()).hexdigest()
                or result.get("learning_timesteps") != profile["timesteps"]
                or result.get("optimizer_steps", 0) <= 0
                or result.get("cleanup_verified") is not True):
            return False
        manifest = _verify_run_manifest(output, source_commit=source_commit, profile="smoke",
                                        profile_hash=profile_hash, candidate="C", seed_bundle=seed_bundle)
        state = _read_json(output / "state.json")
        validation = _read_json(output / "SMOKE-VALIDATION.json")
        checkpoint_record = _read_json(output / "SMOKE-CHECKPOINT-VERIFICATION.json")
        weights = _read_json(output / "SMOKE-WEIGHT-COMPARISON.json")
        telemetry = _read_json(output / "training-telemetry-manifest.json")
        telemetry_path = output / telemetry["path"]
        checkpoint = output / checkpoint_record["archive_path"]
        return (state.get("status") == "SMOKE_INTEGRATION_PASS"
                and validation.get("status") == "SMOKE_INTEGRATION_PASS"
                and validation.get("source_commit") == source_commit
                and result.get("child_result", {}).get("returncode") == 0
                and result.get("child_result", {}).get("cleanup_seconds") is not None
                and checkpoint_record.get("verified") is True
                and weights.get("weights_changed") is True
                and weights.get("action_policy_weights_changed") is True
                and weights.get("reload_matches_final") is True
                and weights.get("nonzero_gradient_steps", 0) == result.get("optimizer_steps")
                and weights.get("optimizer_steps") == result.get("optimizer_steps")
                and checkpoint_record.get("archive_sha256") == sha256(checkpoint.read_bytes()).hexdigest()
                and checkpoint_record.get("archive_sha256") == result.get("checkpoint_sha256")
                and telemetry.get("complete") is True
                and telemetry.get("step_rows") == profile["timesteps"]
                and telemetry.get("sha256") == sha256(telemetry_path.read_bytes()).hexdigest()
                and state.get("learning_timesteps") == profile["timesteps"]
                and state.get("optimizer_steps") == result.get("optimizer_steps"))
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def _verify_sequential_attempt(root: Path, attempt: Path, source_commit: str,
                               profile_hash: str, seed_bundle: dict) -> bool:
    try:
        header = _read_json(attempt / "attempt.json")
        result = _read_json(attempt / "result.json")
        profile, _ = _load_benchmark_profile(root / "configs/research/local-demo-ppo-benchmark-v1.json")
        if (header.get("mode") != "sequential" or header.get("attempt_id") != attempt.name
                or header.get("concurrency") != 1
                or header.get("max_processes_per_worker") != profile["max_processes_per_worker"]
                or header.get("total_process_budget_including_supervisor")
                    != profile["max_processes_per_worker"] + 1
                or header.get("source_commit") != source_commit
                or header.get("benchmark_profile_sha256") != profile_hash
                or result.get("status") != "ENGINEERING_PASS"
                or result.get("source_commit") != source_commit
                or result.get("benchmark_profile_sha256") != profile_hash
                or result.get("attempt_manifest_sha256") != sha256((attempt / "attempt.json").read_bytes()).hexdigest()
                or result.get("process_cleanup_verified") is not True
                or result.get("minimum_available_memory_bytes", 0) < profile["minimum_available_memory_bytes"]
                or header.get("attempt_ids") != result.get("attempt_ids")
                or set(result.get("completed_jobs", {})) != {"A-seed0", "B-seed0"}):
            return False
        if not _sequential_intervals_are_disjoint(result["completed_jobs"]):
            return False
        for candidate in profile["candidates"]:
            model_id = f"{candidate}-seed0"
            output = attempt / model_id
            validation = result["completed_jobs"][model_id]
            run_manifest = _verify_run_manifest(output, source_commit=source_commit,
                                                 profile="benchmark", profile_hash=profile_hash,
                                                 candidate=candidate, seed_bundle=seed_bundle)
            state = _read_json(output / "state.json")
            integrity = _read_json(output / "training-integrity.json")
            telemetry = _read_json(output / "training-telemetry-manifest.json")
            telemetry_path = output / telemetry["path"]
            timing = _read_json(output / "ppo-runtime-timings.json")
            _validate_timing(timing)
            checkpoint = output / "model-final.zip"
            process = _read_json(attempt / f"{model_id}.process.json")
            if (state.get("status") != "learning_completed"
                    or state.get("learning_timesteps") != profile["timesteps"]
                    or state.get("optimizer_steps", 0) <= 0
                    or not integrity.get("weights_changed")
                    or integrity.get("finite_gradient_steps") != state["optimizer_steps"]
                    or integrity.get("nonzero_gradient_steps") != state["optimizer_steps"]
                    or integrity.get("metrics_finite") is not True
                    or integrity.get("ppo_update_rows", 0) <= 0
                    or telemetry.get("complete") is not True
                    or telemetry.get("step_rows") != profile["timesteps"]
                    or telemetry.get("sha256") != sha256(telemetry_path.read_bytes()).hexdigest()
                    or state.get("final_model", {}).get("sha256") != sha256(checkpoint.read_bytes()).hexdigest()
                    or validation.get("checkpoint_sha256") != state["final_model"]["sha256"]
                    or validation.get("optimizer_steps") != state["optimizer_steps"]
                    or validation.get("timesteps") != profile["timesteps"]
                    or validation.get("rollout_seconds") != timing["collect_rollouts_seconds"]
                    or validation.get("optimizer_seconds") != timing["optimizer_updates_seconds"]
                    or validation.get("status") != "ENGINEERING_PASS"
                    or result["completed_jobs"][model_id].get("returncode") != 0
                    or result["completed_jobs"][model_id].get("cleanup_seconds") is None
                    or result["completed_jobs"][model_id].get("resource_usage", {}).get(
                        "peak_process_tree_rss_bytes", profile["max_process_tree_rss_bytes"] + 1)
                    > profile["max_process_tree_rss_bytes"]
                    or result["completed_jobs"][model_id].get("resource_usage", {}).get(
                        "maximum_process_count", profile["max_processes_per_worker"] + 1)
                    > profile["max_processes_per_worker"]
                    or process.get("model_id") != model_id
                    or process.get("source_commit") != source_commit
                    or process.get("parallel_manifest_sha256") != profile_hash
                    or process.get("attempt_id") != result.get("attempt_ids", {}).get(model_id)
                    or process.get("pid") != result["completed_jobs"][model_id].get(
                        "process_identity", {}).get("pid")
                    or process.get("creation_time_100ns") != result["completed_jobs"][model_id].get(
                        "process_identity", {}).get("creation_time_100ns")):
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def _run_smoke(root: Path, artifact_root: Path) -> dict:
    profile, profile_hash = _load_smoke_profile(root / "configs/research/local-demo-ppo-smoke-v1.json")
    benchmark_profile, _ = _load_benchmark_profile(root / "configs/research/local-demo-ppo-benchmark-v1.json")
    source_commit = _git(root, "rev-parse", "HEAD")
    if _git(root, "status", "--porcelain"):
        raise RuntimeError("real PPO smoke source admission requires a clean worktree")
    host = windows_host_resources()
    if host["available_memory_bytes"] < 8 * 1024**3:
        raise RuntimeError("available host memory is below the 8 GiB smoke admission budget")
    attempt = artifact_root / f"smoke-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    attempt.mkdir(parents=False, exist_ok=False)
    output = attempt / "C-seed0"
    model = {"model_id": "C-seed0", "candidate_id": "C", "seed_index": 0,
             "attempt_id": uuid4().hex, "source_commit": source_commit,
             "profile": "smoke", "parallel_manifest_sha256": profile_hash}
    resources = {"per_worker_process_tree_rss_limit_bytes": 4 * 1024**3,
                 "max_processes_per_worker": benchmark_profile["max_processes_per_worker"],
                 "memory_poll_interval_seconds": 1,
                 "shutdown_grace_seconds": 10, "cleanup_timeout_seconds": 30}
    header = {"schema_version": "f1rl_real_ppo_benchmark_attempt_v1", "mode": "smoke",
              "attempt_id": attempt.name, "source_commit": source_commit,
              "source_tree_sha256": _source_tree_digest(root),
              "smoke_profile_sha256": profile_hash, "candidate_id": "C",
              "tuning_seed_index": 0, "timesteps": 256, "concurrency": 1,
              "max_processes_per_worker": benchmark_profile["max_processes_per_worker"],
              "invocation": [sys.executable, "scripts/local_demo.py", "train", "--candidate", "C",
                             "--tuning-seed-index", "0", "--output", os.fspath(output),
                             "--profile", "smoke"],
              "python_version": sys.version, "torch_version": __import__("torch").__version__,
              "started_at": datetime.now(timezone.utc).isoformat(), "host_initial": host,
              "heldout_access": False, "scientific_evaluation": False}
    _write_json(attempt / "attempt.json", header)
    started = time.monotonic()
    outcome = None
    error = None
    try:
        from threading import Event
        outcome = _run_parallel_child(root, output, attempt / "C-seed0.log", model,
                                      {"max_wall_seconds_per_model": 1680},
                                      abort_event=Event(), resources=resources,
                                      process_record_path=attempt / "C-seed0.process.json")
        validation = _read_json(output / "SMOKE-VALIDATION.json")
        state = _read_json(output / "state.json")
        checkpoint = _read_json(output / "SMOKE-CHECKPOINT-VERIFICATION.json")
        if (outcome.get("returncode") != 0 or validation.get("status") != "SMOKE_INTEGRATION_PASS"
                or state.get("status") != "SMOKE_INTEGRATION_PASS"
                or state.get("learning_timesteps") != profile["timesteps"]
                or not checkpoint.get("verified") or not checkpoint.get("archive_sha256")):
            raise RuntimeError("real PPO smoke failed learning, telemetry, checkpoint, or cleanup validation")
        summary = {"status": "SMOKE_INTEGRATION_PASS", "source_commit": source_commit,
                   "smoke_profile_sha256": profile_hash, "learning_timesteps": 256,
                   "attempt_manifest_sha256": sha256((attempt / "attempt.json").read_bytes()).hexdigest(),
                   "optimizer_steps": state["optimizer_steps"],
                   "checkpoint_sha256": checkpoint["archive_sha256"],
                   "wall_seconds": time.monotonic() - started, "child_result": outcome,
                   "ended_at": datetime.now(timezone.utc).isoformat(),
                   "cleanup_verified": outcome.get("cleanup_seconds") is not None}
        _write_json(attempt / "result.json", summary)
        return summary
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
        _write_json(attempt / "result.json", {"status": "SMOKE_INTEGRATION_FAILED",
                                                "source_commit": source_commit,
                                                "smoke_profile_sha256": profile_hash,
                                                "wall_seconds": time.monotonic() - started,
                                                "child_result": outcome, "error": error})
        raise RuntimeError(f"real PPO smoke failed; stop here; evidence preserved at {attempt}: {error}") from exception


def _require_prior_passes(root: Path, artifact_root: Path, mode: str, source_commit: str,
                          profile_hash: str, seed_bundle: dict) -> None:
    if mode == "sequential":
        attempts = sorted(artifact_root.glob("smoke-*"))
        if not attempts or not _verify_smoke_attempt(root, attempts[-1], source_commit, seed_bundle):
            raise RuntimeError("sequential benchmark requires a passing real PPO smoke for this source")
        return
    attempts = sorted(artifact_root.glob("sequential-*"))
    if (not attempts or not _verify_sequential_attempt(root, attempts[-1], source_commit,
                                                       profile_hash, seed_bundle)):
        raise RuntimeError("parallel benchmark requires a passing sequential result for this source and profile")


def run_real_ppo_benchmark(root: str | Path, artifact_root: str | Path, mode: str) -> dict:
    root = Path(root).resolve(strict=True)
    if os.name != "nt" or mode not in {"smoke", "sequential", "parallel"}:
        raise ValueError("real PPO benchmark requires Windows and smoke, sequential, or parallel mode")
    artifact_root = _resolve_artifact_root(root, artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    source_commit = _git(root, "rev-parse", "HEAD")
    if _git(root, "status", "--porcelain"):
        raise RuntimeError("real PPO benchmark source admission requires a clean worktree")
    if mode == "smoke":
        return _run_smoke(root, artifact_root)
    profile, profile_hash = _load_benchmark_profile(root / "configs/research/local-demo-ppo-benchmark-v1.json")
    seed_bundle = _expected_seed_bundle(root)
    _require_prior_passes(root, artifact_root, mode, source_commit, profile_hash, seed_bundle)
    logical_processors = os.cpu_count() or 1
    concurrency = 1 if mode == "sequential" else 2
    if profile["torch_threads"] * concurrency > logical_processors:
        raise RuntimeError("benchmark Torch thread budget would oversubscribe logical processors")
    host = windows_host_resources()
    if host["available_memory_bytes"] < profile["minimum_available_memory_bytes"] * concurrency:
        raise RuntimeError("available host memory is below the benchmark admission budget")
    artifact_root.mkdir(parents=True, exist_ok=True)
    attempt = artifact_root / f"{mode}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    attempt.mkdir(parents=False, exist_ok=False)
    resources = {
        "per_worker_process_tree_rss_limit_bytes": profile["max_process_tree_rss_bytes"],
        "max_processes_per_worker": profile["max_processes_per_worker"],
        "memory_poll_interval_seconds": 1,
        "shutdown_grace_seconds": 10, "cleanup_timeout_seconds": 30,
    }
    jobs = []
    for candidate in profile["candidates"]:
        model_id = f"{candidate}-seed0"
        output = attempt / model_id
        jobs.append({"model_id": model_id, "candidate_id": candidate, "seed_index": 0,
                     "ppo_seed": seed_bundle["ppo"], "attempt_id": uuid4().hex,
                     "source_commit": source_commit, "profile": "benchmark",
                     "benchmark_profile_sha256": profile_hash,
                     "parallel_manifest_sha256": profile_hash,
                     "output": str(output), "log_path": str(attempt / f"{model_id}.log"),
                     "process_record_path": str(attempt / f"{model_id}.process.json")})
    # Seed identity is pinned by the existing registered tuning roster and recorded by the child.
    header = {"schema_version": "f1rl_real_ppo_benchmark_attempt_v1", "mode": mode,
              "attempt_id": attempt.name, "source_commit": source_commit,
              "source_tree_sha256": _source_tree_digest(root),
              "benchmark_profile_sha256": profile_hash, "candidate_ids": profile["candidates"],
              "timesteps_per_job": profile["timesteps"], "seed_index": 0,
              "concurrency": concurrency, "torch_threads": profile["torch_threads"],
              "torch_interop_threads": profile["torch_interop_threads"],
              "logical_processors": logical_processors,
              "max_processes_per_worker": profile["max_processes_per_worker"],
              "total_process_budget_including_supervisor": (
                  concurrency * profile["max_processes_per_worker"] + 1),
              "invocation": [sys.executable, "scripts/wp3_real_ppo_benchmark.py",
                             mode, "--output-root", os.fspath(artifact_root)],
              "child_invocations": [[sys.executable, "scripts/local_demo.py", "train",
                                     "--candidate", job["candidate_id"],
                                     "--tuning-seed-index", "0", "--output", job["output"],
                                     "--profile", "benchmark"] for job in jobs],
              "attempt_ids": {job["model_id"]: job["attempt_id"] for job in jobs},
              "python_version": sys.version,
              "torch_version": __import__("torch").__version__,
              "started_at": datetime.now(timezone.utc).isoformat(),
              "host_initial": host, "heldout_access": False,
              "scientific_evaluation": False}
    _write_json(attempt / "attempt.json", header)
    stats = {"aggregate_peak_rss_bytes": 0, "aggregate_rss_measurement": "sampled WorkingSetSize, 1s interval",
             "peak_simultaneous_process_tree_cpu_seconds": 0.0,
             "host_cpu_samples_percent": [], "minimum_available_memory_bytes": host["available_memory_bytes"],
             "observed_concurrent_process_sample": None}
    validations = {}

    def on_tick(active_jobs: tuple[dict, ...], _delta: float) -> str | None:
        host_sample = windows_host_resources()
        stats["host_cpu_samples_percent"].append(windows_cpu_load_percent(0.1))
        stats["minimum_available_memory_bytes"] = min(stats["minimum_available_memory_bytes"],
                                                       host_sample["available_memory_bytes"])
        if host_sample["available_memory_bytes"] < profile["minimum_available_memory_bytes"]:
            return "available host memory dropped below 8 GiB"
        rss = cpu = processes = 0
        sampled_pids = []
        sample_times = []
        for active_job in active_jobs:
            identity_path = Path(active_job["process_record_path"])
            if not identity_path.is_file():
                continue
            identity = _read_json(identity_path)
            if (identity.get("model_id") != active_job["model_id"]
                    or identity.get("attempt_id") != active_job["attempt_id"]
                    or identity.get("source_commit") != source_commit
                    or identity.get("parallel_manifest_sha256") != profile_hash):
                return "child process identity differs from the dispatched benchmark job"
            sample = sample_process_tree(identity["pid"])
            if sample is not None:
                root_row = next((row for row in sample["processes"] if row["pid"] == identity["pid"]), None)
                if root_row is None or root_row["creation_time_100ns"] != identity["creation_time_100ns"]:
                    continue
                rss += sample["rss_bytes"]
                cpu += sample["cpu_seconds"]
                processes += sample["process_count"]
                sampled_pids.append(identity["pid"])
                sample_times.append(sample["sampled_at_monotonic"])
        sample_skew = _sample_skew_seconds(sample_times)
        if (concurrency == 2 and sample_skew is not None and sample_skew <= 0.1
                and stats["observed_concurrent_process_sample"] is None):
            stats["observed_concurrent_process_sample"] = {
                "sampled_at": datetime.now(timezone.utc).isoformat(),
                "pids": sampled_pids, "sample_skew_seconds": sample_skew}
        stats["aggregate_peak_rss_bytes"] = max(stats["aggregate_peak_rss_bytes"], rss)
        stats["peak_simultaneous_process_tree_cpu_seconds"] = max(
            stats["peak_simultaneous_process_tree_cpu_seconds"], cpu)
        if (processes > concurrency * profile["max_processes_per_worker"]
                or rss > concurrency * profile["max_process_tree_rss_bytes"]):
            return "benchmark process or aggregate RSS budget exceeded"
        return None

    def run_child(job: dict, *, abort_event):
        return _run_parallel_child(root, Path(job["output"]), Path(job["log_path"]), job,
                                   {"max_wall_seconds_per_model": 780},
                                   abort_event=abort_event, resources=resources,
                                   process_record_path=Path(job["process_record_path"]))

    started = time.monotonic()
    def validate_job(job: dict, child_result: dict) -> dict:
        validation = _validate_job(job, child_result)
        validations[job["model_id"]] = validation
        return validation

    result = _run_bounded_stage(jobs, concurrency, run_child, validate_job,
                                on_tick=on_tick, tick_interval_seconds=1)
    elapsed = time.monotonic() - started
    if result["status"] != "ENGINEERING_PASS":
        _write_json(attempt / "result.json", {"status": result["status"],
                                                "failures": result["failures"],
                                                "outcomes": result["outcomes"],
                                                "elapsed_seconds": elapsed, **stats})
        raise RuntimeError(f"real PPO {mode} benchmark failed; evidence preserved at {attempt}")
    transitions = 8192
    intervals, overlap_estimate_seconds = _execution_intervals(jobs, result["outcomes"])
    if mode == "parallel" and stats["observed_concurrent_process_sample"] is None:
        _write_json(attempt / "result.json", {"status": "ENGINEERING_FAILED",
                                                "mode": mode, "source_commit": source_commit,
                                                "benchmark_profile_sha256": profile_hash,
                                                "execution_intervals": intervals,
                                                "interval_overlap_estimate_seconds": overlap_estimate_seconds,
                                                "observed_concurrent_process_sample": None,
                                                "elapsed_seconds": elapsed, **stats})
        raise RuntimeError(f"parallel benchmark lacked a simultaneous live-process sample; evidence preserved at {attempt}")
    summary = {"status": "ENGINEERING_PASS", "mode": mode,
               "source_commit": source_commit,
               "benchmark_profile_sha256": profile_hash,
               "attempt_manifest_sha256": sha256((attempt / "attempt.json").read_bytes()).hexdigest(),
               "attempt_ids": {job["model_id"]: job["attempt_id"] for job in jobs},
               "total_wall_seconds": elapsed,
               "aggregate_transitions_per_second": transitions / elapsed,
               "completed_jobs": {model_id: {**validations[model_id], **result["outcomes"][model_id]}
                                  for model_id in result["outcomes"]}, **stats,
               **_summarize_process_cpu(result["outcomes"], elapsed, logical_processors),
               "execution_intervals": intervals,
               "interval_overlap_estimate_seconds": overlap_estimate_seconds,
               "ended_at": datetime.now(timezone.utc).isoformat(),
               "process_cleanup_verified": all(
                   row.get("cleanup_seconds") is not None for row in result["outcomes"].values())}
    _write_json(attempt / "result.json", summary)
    return summary
