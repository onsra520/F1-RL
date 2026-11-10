from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

from ..learning.pilot_preflight import TRAIN_IDS
from .campaign import (
    CAMPAIGN_CONFIG, _CampaignLock, _atomic_json, _canonical, _inside,
    _iter_jsonl, _load_campaign_config, _run_child, _sha256, _strict_json,
    _verify_evidence_manifest, _write_evidence_manifest, _write_immutable_json,
    _write_new_json, _write_training_deliverables,
    build_campaign_manifest, validate_training_run,
)

PARALLEL_CAMPAIGN_CONFIG = Path("configs/research/wp3-ppo-tuning-campaign-parallel-v1.json")
PARALLEL_CAMPAIGN_BASE_SHA256 = "8a67db80464dc3b93072c0edc431f7a71b1b271c392b47d0915af0f2cac106e4"


def _expected_parallel_campaign_config() -> dict:
    return {
        "schema_version": "f1rl_wp3_ppo_tuning_campaign_parallel_v1",
        "campaign_id": "wp3-abcd-12-models-parallel-v1",
        "base_campaign_config": CAMPAIGN_CONFIG.as_posix(),
        "base_campaign_config_sha256": PARALLEL_CAMPAIGN_BASE_SHA256,
        "allowed_concurrency": [1, 2, 4],
        "default_concurrency": 1,
        "scheduler": {
            "gate_model_id": "C-seed0",
            "queue_policy": "dynamic_fifo_after_engineering_pass",
            "failure_policy": "stop_dispatch_cancel_active_preserve",
            "explicit_opt_in": True,
        },
        "resources": {
            "device": "cpu", "n_envs": 1,
            "per_worker_torch_threads": 8,
            "per_worker_torch_interop_threads": 1,
            "max_total_intraop_threads": 32,
            "max_total_interop_threads": 4,
            "per_worker_process_tree_rss_limit_bytes": 4294967296,
            "max_total_process_tree_rss_bytes": 17179869184,
            "minimum_available_memory_bytes": 8589934592,
            "max_training_processes": 4,
            "max_processes_per_worker": 1,
            "supervisor_processes": 1,
            "max_total_processes": 5,
            "memory_poll_interval_seconds": 1,
            "host_cpu_load_limit_percent": 20.0,
            "shutdown_grace_seconds": 10,
            "cleanup_timeout_seconds": 30,
            "minimum_free_disk_safety_bytes": 1073741824,
        },
        "recovery": {
            "automatic_retries_per_model": 0,
            "maximum_manual_retries_per_model": 1,
            "resume_policy": "reuse_only_fully_validated_attempts",
            "incomplete_attempt_policy": "preserve_and_require_explicit_retry",
        },
        "benchmark": {
            "schema_version": "f1rl_wp3_parallel_scheduler_benchmark_v1",
            "comparison_worker_counts": [1, 2, 4],
            "workload_kind": "mock_subprocess_only",
            "campaign_completion": False,
            "metrics": [
                "campaign_wall_seconds", "completed_jobs_per_hour",
                "training_transitions_per_second", "ppo_rollout_seconds",
                "ppo_optimizer_seconds", "aggregate_cpu_percent",
                "aggregate_peak_rss_bytes", "per_job_duration_seconds",
                "worker_utilization", "scheduler_idle_seconds",
                "process_cleanup_seconds", "failure_count", "retry_count",
            ],
        },
        "scientific_training_authorized": False,
        "scientific_evidence": False,
        "heldout_access": False,
        "wp7_and_gate2": "BLOCKED",
    }


def validate_parallel_contract(config: dict, base_config_sha256: str) -> None:
    if base_config_sha256 != PARALLEL_CAMPAIGN_BASE_SHA256:
        raise ValueError("parallel campaign requires the pinned historical v3 config")
    if config != _expected_parallel_campaign_config():
        raise ValueError("parallel campaign contract differs from its frozen v1 contract")


def load_parallel_contract(root: Path) -> tuple[dict, str]:
    _base_config, _base_raw, base_config_sha256 = _load_campaign_config(root)
    config, raw = _strict_json(root / PARALLEL_CAMPAIGN_CONFIG)
    validate_parallel_contract(config, base_config_sha256)
    return config, sha256(raw).hexdigest()


def validate_parallel_resources(config: dict, workers: int, *,
                                logical_processors: int,
                                available_memory_bytes: int) -> dict:
    validate_parallel_contract(config, config.get("base_campaign_config_sha256"))
    if type(workers) is not int or workers not in config["allowed_concurrency"]:
        raise ValueError("worker count must be one of the versioned concurrency choices")
    if type(logical_processors) is not int or logical_processors < 1:
        raise ValueError("logical processor count must be a positive integer")
    if type(available_memory_bytes) is not int or available_memory_bytes < 0:
        raise ValueError("available memory must be a nonnegative integer")

    resources = config["resources"]
    intraop_threads = workers * resources["per_worker_torch_threads"]
    interop_threads = workers * resources["per_worker_torch_interop_threads"]
    process_tree_rss_limit = workers * resources["per_worker_process_tree_rss_limit_bytes"]
    training_processes = workers * resources["max_processes_per_worker"]
    total_processes = training_processes + resources["supervisor_processes"]
    if intraop_threads > logical_processors:
        raise ValueError("worker Torch thread budget exceeds host logical processor capacity")
    if (intraop_threads > resources["max_total_intraop_threads"]
            or interop_threads > resources["max_total_interop_threads"]):
        raise ValueError("worker Torch thread budget exceeds the parallel contract")
    if total_processes > resources["max_total_processes"]:
        raise ValueError("worker process budget exceeds the parallel contract")
    if process_tree_rss_limit > resources["max_total_process_tree_rss_bytes"]:
        raise ValueError("worker process-tree RSS budget exceeds the parallel contract")
    required_memory = resources["minimum_available_memory_bytes"] + process_tree_rss_limit
    if available_memory_bytes < required_memory:
        raise ValueError("available memory is below the parallel worker admission budget")
    return {
        "training_processes": workers,
        "training_process_tree_processes": training_processes,
        "total_processes": total_processes,
        "torch_intraop_threads": intraop_threads,
        "torch_interop_threads": interop_threads,
        "process_tree_rss_limit_bytes": process_tree_rss_limit,
        "required_available_memory_bytes": required_memory,
    }


def _run_bounded_stage(jobs: list[dict], workers: int, run_child,
                       validate_job, *, on_dispatch=None, on_event=None, on_tick=None,
                       tick_interval_seconds: float = 0.25) -> dict:
    if type(workers) is not int or workers not in (1, 2, 4):
        raise ValueError("worker count must be one of 1, 2, or 4")
    if type(tick_interval_seconds) not in (int, float) or tick_interval_seconds <= 0:
        raise ValueError("scheduler tick interval must be positive")
    pending = list(jobs)
    model_ids = [job.get("model_id") for job in pending]
    if any(not isinstance(model_id, str) or not model_id for model_id in model_ids):
        raise ValueError("parallel schedule contains an invalid model identity")
    if len(set(model_ids)) != len(model_ids):
        raise ValueError("parallel schedule contains duplicate model identities")

    cancellation = threading.Event()
    outcomes = {}
    validations = {}
    failures = {}
    cancelled = []
    active = {}
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="wp3-training-child")
    failure_seen = False

    def dispatch() -> None:
        while pending and len(active) < workers and not failure_seen:
            job = pending.pop(0)
            if on_dispatch is not None:
                job = on_dispatch(job)
            if on_event is not None:
                on_event("job_dispatched", job, None)
            future = pool.submit(run_child, job, abort_event=cancellation)
            active[future] = job

    try:
        dispatch()
        last_tick = time.monotonic()
        while active:
            completed, _ = wait(active, timeout=tick_interval_seconds,
                                 return_when=FIRST_COMPLETED)
            if on_tick is not None:
                now = time.monotonic()
                violation = on_tick(tuple(active.values()), max(0.0, now - last_tick))
                last_tick = now
                if violation and not failure_seen:
                    failures["resource_budget"] = str(violation)
                    failure_seen = True
                    cancellation.set()
                    if on_event is not None:
                        on_event("resource_budget_exceeded", {}, str(violation))
                        for active_job in active.values():
                            on_event("cancellation_requested", active_job, None)
            if not completed:
                continue
            failed_now = False
            for future in completed:
                job = active.pop(future)
                model_id = job["model_id"]
                try:
                    result = future.result()
                except KeyboardInterrupt:
                    cancellation.set()
                    raise
                except Exception as error:
                    failures[model_id] = f"{type(error).__name__}: {error}"
                    if on_event is not None:
                        on_event("job_failed", job, failures[model_id])
                    failed_now = True
                    continue
                if not isinstance(result, dict):
                    failures[model_id] = "ValueError: child result must be an object"
                    if on_event is not None:
                        on_event("job_failed", job, failures[model_id])
                    failed_now = True
                    continue
                if result.get("cancelled") is True:
                    cancelled.append(model_id)
                    if on_event is not None:
                        on_event("job_cancelled", job, result)
                    continue
                if result.get("returncode") != 0:
                    failures[model_id] = f"RuntimeError: child returned {result.get('returncode')}"
                    if on_event is not None:
                        on_event("job_failed", job, failures[model_id])
                    failed_now = True
                    continue
                try:
                    validation = validate_job(job, result)
                    if not isinstance(validation, dict) or validation.get("status") != "ENGINEERING_PASS":
                        raise ValueError("job validation did not return ENGINEERING_PASS")
                except Exception as error:
                    failures[model_id] = f"{type(error).__name__}: {error}"
                    if on_event is not None:
                        on_event("job_failed", job, failures[model_id])
                    failed_now = True
                    continue
                outcomes[model_id] = result
                validations[model_id] = validation
                if on_event is not None:
                    on_event("job_engineering_pass", job, {"validation": validation, "result": result})
            if failed_now:
                failure_seen = True
                cancellation.set()
                if on_event is not None:
                    for job in active.values():
                        on_event("cancellation_requested", job, None)
            dispatch()
    except KeyboardInterrupt:
        cancellation.set()
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    except BaseException:
        cancellation.set()
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)

    return {
        "status": "ENGINEERING_FAILED" if failures else "ENGINEERING_PASS",
        "outcomes": outcomes,
        "validations": validations,
        "failures": failures,
        "failed_model_id": next(iter(failures), None),
        "cancelled_model_ids": cancelled,
        "not_started_model_ids": [job["model_id"] for job in pending],
    }


def _c0_gate_passes(validation: dict) -> bool:
    if not isinstance(validation, dict):
        return False
    lifecycle = validation.get("lifecycle")
    if (validation.get("status") != "ENGINEERING_PASS"
            or validation.get("model_id") != "C-seed0"
            or validation.get("learning_timesteps") != 65536
            or not isinstance(lifecycle, dict)
            or lifecycle.get("automatic_resets_verified", 0) < 1):
        return False
    track_counts = lifecycle.get("track_timestep_counts")
    return (isinstance(track_counts, dict)
            and set(track_counts) == set(TRAIN_IDS)
            and all(type(track_counts[track]) is int and track_counts[track] > 0
                    for track in TRAIN_IDS))


def run_parallel_schedule(schedule: list[dict], workers: int, run_child,
                          validate_job, *, on_dispatch=None, on_event=None,
                          on_tick=None, on_gate_result=None,
                          tick_interval_seconds: float = 0.25) -> dict:
    if type(workers) is not int or workers not in (1, 2, 4):
        raise ValueError("worker count must be one of 1, 2, or 4")
    model_ids = [row.get("model_id") for row in schedule]
    if (not schedule or model_ids[0] != "C-seed0"
            or any(not isinstance(model_id, str) for model_id in model_ids)
            or len(set(model_ids)) != len(model_ids)):
        raise ValueError("parallel schedule must start with C-seed0 and contain unique model identities")

    gate_stage = _run_bounded_stage([schedule[0]], 1, run_child, validate_job,
                                    on_dispatch=on_dispatch, on_event=on_event,
                                    on_tick=on_tick,
                                    tick_interval_seconds=tick_interval_seconds)
    if on_gate_result is not None:
        on_gate_result(gate_stage)
    gate_validation = gate_stage["validations"].get("C-seed0")
    if (gate_stage["status"] != "ENGINEERING_PASS"
            or not _c0_gate_passes(gate_validation)):
        gate_failed = dict(gate_stage)
        if gate_validation is not None and not _c0_gate_passes(gate_validation):
            gate_failed["failures"] = {"C-seed0": "C0 full-length engineering gate failed"}
        gate_failed.update(
            status="C0_ENGINEERING_FAILED",
            c0_validation=gate_validation,
            completed_model_ids=[],
            pending_model_ids=model_ids[1:],
        )
        return gate_failed

    remaining_stage = _run_bounded_stage(schedule[1:], workers, run_child, validate_job,
                                         on_dispatch=on_dispatch, on_event=on_event,
                                         on_tick=on_tick,
                                         tick_interval_seconds=tick_interval_seconds)
    completed = ["C-seed0", *remaining_stage["validations"]]
    return {
        **remaining_stage,
        "status": remaining_stage["status"],
        "c0_validation": gate_validation,
        "completed_model_ids": completed,
        "pending_model_ids": remaining_stage["not_started_model_ids"],
    }

def _run_parallel_child(root: Path, output: Path, log_path: Path, model: dict,
                         training: dict, *, abort_event, resources: dict,
                         process_record_path: Path | None) -> dict:
    from .process_resources import WindowsKillJob, sample_process_tree

    command = [sys.executable, "scripts/local_demo.py", "train",
               "--candidate", model["candidate_id"],
               "--tuning-seed-index", str(model["seed_index"]),
               "--output", str(output), "--profile", model.get("profile", "tuning")]
    started = time.monotonic()
    deadline = started + training["max_wall_seconds_per_model"] + 120
    poll_seconds = resources.get("memory_poll_interval_seconds", 1)
    max_tree_rss = resources["per_worker_process_tree_rss_limit_bytes"]
    max_processes = resources["max_processes_per_worker"]
    peak_rss_bytes = 0
    max_process_count = 0
    final_sample = None
    root_identity = None
    startup_seconds = None
    process_started_at = None
    cleanup_started = None
    with log_path.open("x", encoding="utf-8", newline="\n") as log:
        proc = subprocess.Popen(command, cwd=root, env=os.environ.copy(), stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT,
                                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        process_started_at = datetime.now(timezone.utc).isoformat()
        job = None
        try:
            job = WindowsKillJob()
            job.assign(proc.pid)
            while True:
                sample = sample_process_tree(proc.pid)
                if sample is not None:
                    final_sample = sample
                    peak_rss_bytes = max(peak_rss_bytes, sample["rss_bytes"])
                    max_process_count = max(max_process_count, sample["process_count"])
                    root_row = next((row for row in sample["processes"] if row["pid"] == proc.pid), None)
                    if root_identity is None and root_row is not None:
                        root_identity = {"pid": proc.pid, "creation_time_100ns": root_row["creation_time_100ns"]}
                        startup_seconds = time.monotonic() - started
                        if process_record_path is not None:
                            if process_record_path.exists():
                                raise FileExistsError("attempt process identity record already exists")
                            _atomic_json(process_record_path, {
                                "schema_version": "f1rl_wp3_child_process_identity_v1",
                                "model_id": model["model_id"], "attempt_id": model["attempt_id"],
                                "source_commit": model["source_commit"],
                                "parallel_manifest_sha256": model["parallel_manifest_sha256"],
                                **root_identity, "started_at": datetime.now(timezone.utc).isoformat(),
                            })
                    if sample["rss_bytes"] > max_tree_rss:
                        raise RuntimeError("training process tree exceeded its RSS budget")
                    if sample["process_count"] > max_processes:
                        raise RuntimeError("training process tree exceeded its process-count budget")

                returncode = proc.poll()
                if returncode is not None:
                    process_exit_observed_at = datetime.now(timezone.utc).isoformat()
                    break
                if abort_event.is_set():
                    cleanup_started = time.monotonic()
                    try:
                        proc.send_signal(getattr(__import__("signal"), "CTRL_BREAK_EVENT"))
                    except (OSError, ProcessLookupError):
                        pass
                    grace_deadline = time.monotonic() + resources["shutdown_grace_seconds"]
                    while proc.poll() is None and time.monotonic() < grace_deadline:
                        time.sleep(min(0.1, max(0.0, grace_deadline - time.monotonic())))
                    if proc.poll() is None:
                        job.terminate()
                        try:
                            proc.wait(timeout=resources["cleanup_timeout_seconds"])
                        except subprocess.TimeoutExpired as error:
                            raise TimeoutError("cancelled training process tree did not terminate") from error
                    process_exit_observed_at = datetime.now(timezone.utc).isoformat()
                    if not job.wait_empty(resources["cleanup_timeout_seconds"]):
                        raise TimeoutError("cancelled training process tree cleanup did not finish")
                    cleanup_seconds = time.monotonic() - cleanup_started
                    return {"returncode": proc.returncode, "cancelled": True,
                            "elapsed_seconds": time.monotonic() - started,
                            "process_started_at": process_started_at,
                            "process_exit_observed_at": process_exit_observed_at,
                            "startup_seconds": startup_seconds,
                            "cleanup_seconds": cleanup_seconds,
                            "resource_usage": _parallel_resource_usage(final_sample, peak_rss_bytes,
                                                                        max_process_count, started)}
                if time.monotonic() >= deadline:
                    cleanup_started = time.monotonic()
                    job.terminate()
                    try:
                        proc.wait(timeout=resources["cleanup_timeout_seconds"])
                    except subprocess.TimeoutExpired as error:
                        raise TimeoutError("timed-out training process tree did not terminate") from error
                    if not job.wait_empty(resources["cleanup_timeout_seconds"]):
                        raise TimeoutError("timed-out training process tree cleanup did not finish")
                    raise TimeoutError("campaign parent wall-time guard expired; child evidence preserved")
                time.sleep(poll_seconds)
            cleanup_started = time.monotonic()
            job.terminate()
            if not job.wait_empty(resources["cleanup_timeout_seconds"]):
                raise TimeoutError("training child descendants did not exit during cleanup")
            cleanup_seconds = time.monotonic() - cleanup_started
            return {
                "returncode": returncode, "cancelled": False,
                "elapsed_seconds": time.monotonic() - started,
                "process_started_at": process_started_at,
                "process_exit_observed_at": process_exit_observed_at,
                "startup_seconds": startup_seconds,
                "cleanup_seconds": cleanup_seconds,
                "process_identity": root_identity,
                "resource_usage": _parallel_resource_usage(final_sample, peak_rss_bytes,
                                                            max_process_count, started),
            }
        except BaseException:
            if proc.poll() is None:
                try:
                    if os.name == "nt":
                        proc.send_signal(getattr(__import__("signal"), "CTRL_BREAK_EVENT"))
                    proc.wait(timeout=resources.get("shutdown_grace_seconds", 10))
                except (OSError, ProcessLookupError, subprocess.TimeoutExpired):
                    if job is not None:
                        job.terminate()
                    else:
                        proc.kill()
                    try:
                        proc.wait(timeout=resources.get("cleanup_timeout_seconds", 30))
                    except subprocess.TimeoutExpired as error:
                        raise TimeoutError("failed training process tree did not terminate") from error
            if job is not None:
                job.terminate()
                if not job.wait_empty(resources.get("cleanup_timeout_seconds", 30)):
                    raise TimeoutError("failed training process tree cleanup did not finish")
            raise
        finally:
            if job is not None:
                job.close()


def _parallel_resource_usage(sample: dict | None, peak_rss_bytes: int,
                             max_process_count: int, started: float) -> dict:
    elapsed = max(0.0, time.monotonic() - started)
    cpu_seconds = sample.get("cpu_seconds") if sample else None
    logical_processors = os.cpu_count() or 1
    return {
        "peak_process_tree_rss_bytes": peak_rss_bytes or None,
        "final_process_tree_cpu_seconds": cpu_seconds,
        "aggregate_cpu_percent_of_host": (100.0 * cpu_seconds / (elapsed * logical_processors)
                                           if cpu_seconds is not None and elapsed > 0 else None),
        "maximum_process_count": max_process_count or None,
        "sampling_interval_seconds": 1,
        "measurement_source": "Windows Toolhelp32Snapshot/GetProcessMemoryInfo/GetProcessTimes",
    }

def _parallel_document(value: dict) -> dict:
    document = dict(value)
    document["sha256"] = sha256(_canonical(document)).hexdigest()
    return document


def _verify_parallel_document(value: dict, label: str) -> None:
    if value.get("sha256") != sha256(_canonical({key: item for key, item in value.items()
                                                    if key != "sha256"})).hexdigest():
        raise ValueError(f"parallel {label} self-checksum mismatch")


def _parallel_job_identity(job: dict, base_manifest: dict,
                           parallel_config_sha256: str) -> str:
    identity = {
        "model_id": job["model_id"], "candidate_id": job["candidate_id"],
        "seed_index": job["seed_index"], "seed_bundle": job["seed_bundle"],
        "candidate_factors": job["candidate_factors"],
        "source_commit": base_manifest["source_commit"],
        "source_tree_sha256": base_manifest["source_tree_sha256"],
        "campaign_config_sha256": base_manifest["campaign_config_sha256"],
        "parallel_config_sha256": parallel_config_sha256,
    }
    return sha256(_canonical(identity)).hexdigest()


def _parallel_intent(base_manifest: dict, schedule: list[dict],
                     parallel_config_sha256: str, workers: int,
                     maximum_manual_retries_per_model: int) -> dict:
    jobs = [{
        "model_id": row["model_id"],
        "job_identity_sha256": _parallel_job_identity(row, base_manifest, parallel_config_sha256),
    } for row in schedule]
    if len(jobs) != 12 or len({row["model_id"] for row in jobs}) != 12:
        raise ValueError("parallel intent requires the exact 12-job frozen schedule")
    return _parallel_document({
        "schema_version": "f1rl_wp3_parallel_campaign_intent_v1",
        "campaign_id": "wp3-abcd-12-models-parallel-v1",
        "base_campaign_manifest_sha256": base_manifest["sha256"],
        "source_commit": base_manifest["source_commit"],
        "source_tree_sha256": base_manifest["source_tree_sha256"],
        "base_campaign_config_sha256": base_manifest["campaign_config_sha256"],
        "parallel_config_sha256": parallel_config_sha256,
        "worker_count": workers,
        "maximum_manual_retries_per_model": maximum_manual_retries_per_model,
        "schedule": jobs,
        "scientific_training_authorized": False, "scientific_evidence": False,
        "heldout_access": False, "wp7_status": "BLOCKED", "gate2_status": "BLOCKED",
    })


def _append_parallel_event(path: Path, event: dict) -> None:
    payload = json.dumps({"schema_version": "f1rl_wp3_parallel_campaign_event_v1", **event},
                         sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _parallel_artifact_inventory(run_dir: Path) -> list[dict]:
    inventory = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_symlink():
            raise ValueError("parallel run evidence cannot contain symbolic links")
        if path.is_file():
            inventory.append({"path": path.relative_to(run_dir).as_posix(),
                              "bytes": path.stat().st_size, "sha256": _sha256(path)})
    return inventory


def _parallel_attempt_path(output: Path, model_id: str, attempt_id: str) -> Path:
    relative = Path("runs") / model_id / attempt_id
    return output / relative


def _parallel_validate_attempt(root: Path, output: Path, job: dict,
                               base_manifest: dict, config: dict) -> dict:
    run_dir = _parallel_attempt_path(output, job["model_id"], job["attempt_id"])
    validation = validate_training_run(root, run_dir, job, base_manifest, config)
    validation["sha256"] = sha256(_canonical(validation)).hexdigest()
    validation_path = run_dir / "engineering-validation.json"
    _write_immutable_json(validation_path, validation)
    stored, _ = _strict_json(validation_path)
    _verify_parallel_document(stored, "engineering validation")
    if stored != validation:
        raise ValueError("stored engineering validation differs from current artifact validation")
    return validation


def _parallel_preflight(root: Path, output: Path, parallel_config: dict,
                        campaign_config: dict, workers: int) -> dict:
    from .process_resources import windows_cpu_load_percent, windows_host_resources

    host = windows_host_resources()
    budget = validate_parallel_resources(
        parallel_config, workers,
        logical_processors=host["logical_processors"],
        available_memory_bytes=host["available_memory_bytes"],
    )
    cpu_load = windows_cpu_load_percent()
    if cpu_load > parallel_config["resources"]["host_cpu_load_limit_percent"]:
        raise RuntimeError(f"host CPU load {cpu_load:.1f}% exceeds the parallel admission limit")
    storage = campaign_config["resources"]
    required_disk = (storage["telemetry_per_model_max_bytes"] * 12
                     + parallel_config["resources"]["minimum_free_disk_safety_bytes"])
    output.parent.mkdir(parents=True, exist_ok=True)
    free_disk = shutil.disk_usage(output.parent).free
    if free_disk < required_disk:
        raise OSError(f"insufficient free disk at parallel campaign output: need {required_disk} bytes")
    return {
        "logical_processors": host["logical_processors"],
        "available_memory_bytes": host["available_memory_bytes"],
        "total_memory_bytes": host["total_memory_bytes"],
        "host_cpu_load_percent": cpu_load,
        "available_free_disk_bytes": free_disk,
        "required_free_disk_bytes": required_disk,
        "resource_budget": budget,
        "device": "cpu", "n_envs": 1,
        "torch_intraop_threads_per_job": parallel_config["resources"]["per_worker_torch_threads"],
        "torch_interop_threads_per_job": parallel_config["resources"]["per_worker_torch_interop_threads"],
        "scientific_evidence": False, "heldout_access": False,
    }


def _parallel_activation_paths(output: Path, attempt_id: str) -> tuple[Path, Path, Path]:
    return (
        output / "gates" / f"C0-ENGINEERING-GATE-{attempt_id}.json",
        output / "manifests" / f"WP3-PARALLEL-CAMPAIGN-MANIFEST-{attempt_id}.json",
        output / "decisions" / f"WP3-PARALLEL-SCHEDULING-DECISION-{attempt_id}.json",
    )


def _parallel_gate_payload(output: Path, state: dict, intent: dict,
                           base_manifest: dict, stage: dict) -> dict:
    entry = state["models"]["C-seed0"]
    attempt = entry.get("attempts", [])[-1] if entry.get("attempts") else None
    if attempt is None:
        raise ValueError("C0 gate has no isolated attempt record")
    run_dir = output / attempt["run_relative_path"]
    validation = stage.get("validations", {}).get("C-seed0")
    status = "ENGINEERING_PASS" if stage.get("status") == "ENGINEERING_PASS" and _c0_gate_passes(validation) else "ENGINEERING_FAIL"
    inventory = _parallel_artifact_inventory(run_dir) if run_dir.exists() else []
    inventory_sha = sha256(_canonical(inventory)).hexdigest()
    return _parallel_document({
        "schema_version": "f1rl_wp3_c0_parallel_engineering_gate_v1",
        "status": status, "model_id": "C-seed0",
        "attempt_id": attempt["attempt_id"],
        "job_identity_sha256": attempt["job_identity_sha256"],
        "intent_sha256": intent["sha256"],
        "base_campaign_manifest_sha256": base_manifest["sha256"],
        "source_commit": base_manifest["source_commit"],
        "source_tree_sha256": base_manifest["source_tree_sha256"],
        "run_relative_path": attempt["run_relative_path"],
        "validation_sha256": validation["sha256"] if validation else None,
        "artifact_inventory": inventory,
        "artifact_inventory_sha256": inventory_sha,
        "failure": (stage.get("failures", {}).get("C-seed0")
                    or ("C0 full-length lifecycle gate failed"
                        if validation is not None and not _c0_gate_passes(validation) else None)),
        "learning_timesteps": validation.get("learning_timesteps") if validation else None,
        "lifecycle": validation.get("lifecycle") if validation else None,
        "scientific_evidence": False, "heldout_access": False,
    })


def _parallel_write_activation(output: Path, state: dict, intent: dict,
                               base_manifest: dict, workers: int,
                               gate: dict) -> None:
    attempt_id = gate["attempt_id"]
    gate_path, manifest_path, decision_path = _parallel_activation_paths(output, attempt_id)
    _write_immutable_json(gate_path, gate)
    campaign_manifest = _parallel_document({
        "schema_version": "f1rl_wp3_parallel_campaign_manifest_v1",
        "intent_sha256": intent["sha256"],
        "base_campaign_manifest_sha256": base_manifest["sha256"],
        "source_commit": base_manifest["source_commit"],
        "source_tree_sha256": base_manifest["source_tree_sha256"],
        "parallel_config_sha256": intent["parallel_config_sha256"],
        "worker_count": workers, "c0_gate_path": gate_path.relative_to(output).as_posix(),
        "c0_gate_sha256": gate["sha256"], "c0_gate_status": gate["status"],
        "scientific_training_authorized": False, "scientific_evidence": False,
        "heldout_access": False, "wp7_status": "BLOCKED", "gate2_status": "BLOCKED",
    })
    _write_immutable_json(manifest_path, campaign_manifest)
    pending = [row["model_id"] for row in intent["schedule"] if row["model_id"] != "C-seed0"]
    decision = _parallel_document({
        "schema_version": "f1rl_wp3_parallel_scheduling_decision_v1",
        "intent_sha256": intent["sha256"],
        "campaign_manifest_path": manifest_path.relative_to(output).as_posix(),
        "campaign_manifest_sha256": campaign_manifest["sha256"],
        "c0_gate_sha256": gate["sha256"],
        "decision": "OPEN_11_JOB_QUEUE" if gate["status"] == "ENGINEERING_PASS" else "NO_JOBS_AFTER_C0_FAILURE",
        "queued_model_ids": pending if gate["status"] == "ENGINEERING_PASS" else [],
        "scheduled_at": datetime.now(timezone.utc).isoformat(),
        "scientific_evidence": False, "heldout_access": False,
    })
    _write_immutable_json(decision_path, decision)
    state.update({
        "active_c0_gate_path": gate_path.relative_to(output).as_posix(),
        "active_c0_gate_sha256": gate["sha256"],
        "active_campaign_manifest_path": manifest_path.relative_to(output).as_posix(),
        "active_campaign_manifest_sha256": campaign_manifest["sha256"],
        "active_scheduling_decision_path": decision_path.relative_to(output).as_posix(),
        "active_scheduling_decision_sha256": decision["sha256"],
    })
    if gate["status"] == "ENGINEERING_PASS":
        state.update(status="RUNNING", phase="STAGE_2_TRAINING")
    else:
        c0_entry = state["models"]["C-seed0"]
        c0_entry.update(status="GATE_FAILED", last_error=gate.get("failure"))
        state["completed_model_ids"] = [model_id for model_id in state["completed_model_ids"]
                                         if model_id != "C-seed0"]
        state["completed_training_timesteps"] = (
            len(state["completed_model_ids"]) * base_manifest["training"]["timesteps_per_model"])
        state.update(status="C0_ENGINEERING_FAILED", phase="STOPPED_BEFORE_QUEUE")
    _atomic_json(output / "parallel-campaign-state.json", state)


def _write_parallel_review_status(output: Path, state: dict, evaluation: dict) -> None:
    status = _parallel_document({
        "schema_version": "f1rl_wp3_parallel_review_status_v1",
        "engineering_status": "ENGINEERING_VALIDATION_COMPLETE",
        "independent_review_status": "PENDING_INDEPENDENT_REVIEW",
        "evaluation_status": evaluation["status"],
        "completed_model_ids": state["completed_model_ids"],
        "scientific_evidence": False, "heldout_access": False,
        "wp7_status": "BLOCKED", "gate2_status": "BLOCKED",
    })
    _write_new_json(output / "WP3-PARALLEL-REVIEW-STATUS.json", status)


def _verify_parallel_activation(root: Path, output: Path, state: dict, intent: dict,
                                base_manifest: dict, campaign_config: dict,
                                schedule: list[dict]) -> tuple[dict, dict]:
    required = ("active_c0_gate_path", "active_campaign_manifest_path",
                "active_scheduling_decision_path")
    if any(not state.get(key) for key in required):
        raise RuntimeError("parallel scheduling has no immutable C0 gate activation")
    gate, _ = _strict_json(_inside(output, state["active_c0_gate_path"]))
    manifest, _ = _strict_json(_inside(output, state["active_campaign_manifest_path"]))
    decision, _ = _strict_json(_inside(output, state["active_scheduling_decision_path"]))
    for value, label in ((gate, "C0 gate"), (manifest, "campaign manifest"),
                         (decision, "scheduling decision")):
        _verify_parallel_document(value, label)
    if (gate["sha256"] != state.get("active_c0_gate_sha256")
            or manifest["sha256"] != state.get("active_campaign_manifest_sha256")
            or decision["sha256"] != state.get("active_scheduling_decision_sha256")
            or gate.get("intent_sha256") != intent["sha256"]
            or gate.get("base_campaign_manifest_sha256") != base_manifest["sha256"]
            or gate.get("source_commit") != base_manifest["source_commit"]
            or gate.get("source_tree_sha256") != base_manifest["source_tree_sha256"]
            or manifest.get("intent_sha256") != intent["sha256"]
            or manifest.get("c0_gate_sha256") != gate["sha256"]
            or manifest.get("base_campaign_manifest_sha256") != base_manifest["sha256"]
            or manifest.get("source_commit") != base_manifest["source_commit"]
            or manifest.get("source_tree_sha256") != base_manifest["source_tree_sha256"]
            or manifest.get("worker_count") != intent["worker_count"]
            or manifest.get("parallel_config_sha256") != intent["parallel_config_sha256"]
            or decision.get("campaign_manifest_sha256") != manifest["sha256"]
            or decision.get("intent_sha256") != intent["sha256"]
            or decision.get("c0_gate_sha256") != gate["sha256"]):
        raise ValueError("parallel C0 gate, manifest, decision, source, and state hashes do not bind")
    c0_identity = next(row["job_identity_sha256"] for row in intent["schedule"]
                       if row["model_id"] == "C-seed0")
    c0_attempt = _parallel_find_attempt(state, "C-seed0", gate["attempt_id"])
    if (gate.get("model_id") != "C-seed0"
            or gate.get("job_identity_sha256") != c0_identity
            or c0_attempt.get("job_identity_sha256") != c0_identity
            or c0_attempt.get("run_relative_path") != gate.get("run_relative_path")):
        raise ValueError("C0 gate is not bound to the frozen job and isolated attempt")
    expected_queued = ([row["model_id"] for row in intent["schedule"] if row["model_id"] != "C-seed0"]
                       if gate.get("status") == "ENGINEERING_PASS" else [])
    expected_decision = "OPEN_11_JOB_QUEUE" if expected_queued else "NO_JOBS_AFTER_C0_FAILURE"
    if (decision.get("queued_model_ids") != expected_queued
            or decision.get("decision") != expected_decision):
        raise ValueError("parallel scheduling decision violates the C0 queue boundary")
    if gate.get("status") in {"ENGINEERING_PASS", "ENGINEERING_FAIL"} and gate.get("validation_sha256"):
        gate_job = next(row for row in schedule
                        if row["model_id"] == "C-seed0")
        run_dir = output / gate["run_relative_path"]
        validation = validate_training_run(root, run_dir, gate_job, base_manifest, campaign_config)
        validation["sha256"] = sha256(_canonical(validation)).hexdigest()
        stored, _ = _strict_json(run_dir / "engineering-validation.json")
        if (stored != validation or stored.get("sha256") != gate.get("validation_sha256")
                or (gate.get("status") == "ENGINEERING_PASS" and not _c0_gate_passes(validation))):
            raise ValueError("C0 artifacts fail full validation on resume")
    elif gate.get("status") not in {"ENGINEERING_PASS", "ENGINEERING_FAIL"}:
        raise ValueError("C0 gate has an unknown status")
    run_dir = output / gate["run_relative_path"]
    inventory = _parallel_artifact_inventory(run_dir) if run_dir.exists() else []
    if (inventory != gate.get("artifact_inventory")
            or sha256(_canonical(inventory)).hexdigest() != gate.get("artifact_inventory_sha256")):
        raise ValueError("C0 artifact inventory changed after its immutable engineering gate")
    return gate, decision


def _new_parallel_output(root: Path, output: Path, intent: dict, schedule: list[dict],
                         preflight: dict, base_manifest: dict,
                         campaign_config: dict) -> dict:
    if output.exists():
        raise FileExistsError("parallel campaign output already exists; use resume or explicit retry")
    output.mkdir(parents=True, exist_ok=False)
    for name in ("runs", "logs", "attempt-meta", "gates", "manifests", "decisions", "evaluation"):
        (output / name).mkdir()
    models = {
        row["model_id"]: {
            "status": "PENDING", "job_identity_sha256": row["job_identity_sha256"],
            "attempts": [],
        }
        for row in intent["schedule"]
    }
    state = {
        "schema_version": "f1rl_wp3_parallel_campaign_state_v1",
        "intent_sha256": intent["sha256"],
        "worker_count": intent["worker_count"],
        "status": "PREPARED", "phase": "C0_ENGINEERING_GATE",
        "models": models, "completed_model_ids": [], "completed_training_timesteps": 0,
        "host_preflight": preflight,
        "scientific_evidence": False, "heldout_access": False,
        "wp7_status": "BLOCKED", "gate2_status": "BLOCKED",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_new_json(output / "WP3-PARALLEL-INTENT.json", intent)
    _write_new_json(output / "HOST-PREFLIGHT.json", preflight)
    _write_new_json(output / "parallel-campaign-state.json", state)
    (output / "parallel-campaign-events.jsonl").touch(exist_ok=False)
    from .evaluation import build_evaluation_roster
    roster = build_evaluation_roster(root, base_manifest, campaign_config)
    _write_new_json(output / "evaluation" / "evaluation-roster.json", roster)
    return state


def _load_parallel_output(root: Path, output: Path, expected_intent: dict,
                          base_manifest: dict, schedule: list[dict]) -> tuple[dict, dict]:
    if not output.is_dir() or output.is_symlink():
        raise FileNotFoundError("parallel campaign output does not exist or is not a regular directory")
    if any(path.is_symlink() for path in output.rglob("*")):
        raise ValueError("parallel campaign evidence cannot contain symbolic links")
    intent, _ = _strict_json(output / "WP3-PARALLEL-INTENT.json")
    _verify_parallel_document(intent, "intent")
    if intent != expected_intent:
        raise ValueError("parallel resume source, v3 contract, worker count, or 12-job identity changed")
    state, _ = _strict_json(output / "parallel-campaign-state.json")
    if (state.get("schema_version") != "f1rl_wp3_parallel_campaign_state_v1"
            or state.get("intent_sha256") != intent["sha256"]
            or state.get("worker_count") != intent["worker_count"]
            or set(state.get("models", {})) != {row["model_id"] for row in schedule}):
        raise ValueError("parallel campaign state does not match its immutable intent")
    host_preflight, _ = _strict_json(output / "HOST-PREFLIGHT.json")
    if host_preflight != state.get("host_preflight"):
        raise ValueError("initial host preflight evidence differs from campaign state")
    events_path = output / "parallel-campaign-events.jsonl"
    if not events_path.is_file():
        raise ValueError("parallel campaign lifecycle event log is missing")
    for event in _iter_jsonl(events_path, 16384):
        if event.get("schema_version") != "f1rl_wp3_parallel_campaign_event_v1":
            raise ValueError("parallel campaign lifecycle event schema mismatch")
    job_hashes = {row["model_id"]: row["job_identity_sha256"] for row in intent["schedule"]}
    for model_id, entry in state["models"].items():
        if entry.get("job_identity_sha256") != job_hashes[model_id] or not isinstance(entry.get("attempts"), list):
            raise ValueError(f"parallel model identity or attempt inventory changed: {model_id}")
        seen_attempts = set()
        for attempt in entry["attempts"]:
            attempt_id = attempt.get("attempt_id")
            if (not isinstance(attempt_id, str) or not attempt_id.startswith("attempt-")
                    or attempt_id in seen_attempts or attempt.get("job_identity_sha256") != job_hashes[model_id]):
                raise ValueError(f"invalid or duplicate parallel attempt identity: {model_id}")
            seen_attempts.add(attempt_id)
            run_relative = Path(attempt.get("run_relative_path", ""))
            expected_run = Path("runs") / model_id / attempt_id
            if run_relative.is_absolute() or ".." in run_relative.parts or run_relative != expected_run:
                raise ValueError("parallel attempt output path escapes the campaign root")
            if attempt.get("log_relative_path") != f"logs/{model_id}-{attempt_id}.log":
                raise ValueError("parallel attempt log path does not match its isolated identity")
            record_path = output / "attempt-meta" / model_id / f"{attempt_id}.json"
            if not record_path.is_file():
                raise ValueError("parallel attempt metadata is missing; refusing implicit recovery")
            record, _ = _strict_json(record_path)
            if (record.get("model_id") != model_id or record.get("attempt_id") != attempt_id
                    or record.get("job_identity_sha256") != job_hashes[model_id]
                    or record.get("intent_sha256") != intent["sha256"]
                    or record.get("run_relative_path") != run_relative.as_posix()
                    or record.get("log_relative_path") != attempt["log_relative_path"]):
                raise ValueError("parallel attempt metadata differs from its bound model identity")
    if not state.get("active_c0_gate_path"):
        artifacts = list((output / "gates").glob("*.json")) + list((output / "manifests").glob("*.json")) + list((output / "decisions").glob("*.json"))
        if artifacts:
            raise ValueError("parallel activation artifacts exist without a state pointer; refusing to repair")
    return intent, state


def _parallel_find_attempt(state: dict, model_id: str, attempt_id: str) -> dict:
    attempts = state["models"][model_id]["attempts"]
    for attempt in attempts:
        if attempt["attempt_id"] == attempt_id:
            return attempt
    raise ValueError(f"parallel attempt is missing from state: {model_id}/{attempt_id}")


def _parallel_revalidate_state(root: Path, output: Path, state: dict,
                               base_manifest: dict, config: dict,
                               schedule: list[dict]) -> None:
    by_id = {row["model_id"]: row for row in schedule}
    completed = []
    for model_id, entry in state["models"].items():
        for attempt in entry["attempts"]:
            run_dir = _parallel_attempt_path(output, model_id, attempt["attempt_id"])
            validation_path = run_dir / "engineering-validation.json"
            if not validation_path.is_file():
                if attempt.get("status") == "ENGINEERING_PASS":
                    raise ValueError(f"passed parallel attempt lacks immutable validation evidence: {model_id}")
                continue
            job = dict(by_id[model_id], attempt_id=attempt["attempt_id"])
            validation = validate_training_run(root, run_dir, job, base_manifest, config)
            validation["sha256"] = sha256(_canonical(validation)).hexdigest()
            stored, _ = _strict_json(validation_path)
            _verify_parallel_document(stored, "engineering validation")
            if stored != validation:
                raise ValueError(f"parallel attempt validation changed after completion: {model_id}")
            attempt.update(status="ENGINEERING_PASS", validation_sha256=validation["sha256"],
                           validation=validation)
            attempt.setdefault("run_relative_path", f"runs/{model_id}/{attempt['attempt_id']}")
            entry.update(status="ENGINEERING_PASS", latest_attempt_id=attempt["attempt_id"],
                         validation=validation, output_path=attempt["run_relative_path"])
        if entry.get("status") == "ENGINEERING_PASS":
            if not entry.get("validation") or not entry.get("latest_attempt_id"):
                raise ValueError(f"parallel completed state lacks validated attempt details: {model_id}")
            completed.append(model_id)
    state["completed_model_ids"] = [row["model_id"] for row in schedule if row["model_id"] in completed]
    state["completed_training_timesteps"] = len(completed) * config["training"]["timesteps_per_model"]


def _parallel_assert_no_live_children(output: Path, state: dict) -> None:
    from .process_resources import sample_process_tree

    for model_id, entry in state["models"].items():
        for attempt in entry["attempts"]:
            identity_path = output / "attempt-meta" / model_id / attempt["attempt_id"] / "process-identity.json"
            if not identity_path.is_file():
                continue
            identity, _ = _strict_json(identity_path)
            if (identity.get("model_id") != model_id
                    or identity.get("attempt_id") != attempt["attempt_id"]):
                raise ValueError("live child identity record differs from its attempt")
            sample = sample_process_tree(identity.get("pid", -1))
            if sample is None:
                continue
            root = next((row for row in sample["processes"] if row["pid"] == identity.get("pid")), None)
            if root and root.get("creation_time_100ns") == identity.get("creation_time_100ns"):
                raise RuntimeError(f"parallel child {model_id}/{attempt['attempt_id']} is still alive; refusing duplicate launch")


def _parallel_mark_interrupted(output: Path, state: dict, event_path: Path) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    for model_id, entry in state["models"].items():
        if entry.get("status") not in {"RUNNING", "CANCELLATION_REQUESTED"}:
            continue
        entry["status"] = "INTERRUPTED"
        attempt_id = entry.get("latest_attempt_id")
        if attempt_id:
            attempt = _parallel_find_attempt(state, model_id, attempt_id)
            attempt.update(status="INTERRUPTED", interrupted_at=timestamp)
            if attempt.get("attempt_kind") == "RETRY":
                state["recovery"] = {
                    "status": "RETRY_INTERRUPTED", "model_id": model_id,
                    "attempt_id": attempt_id,
                }
            _append_parallel_event(event_path, {
                "event": "retry_interrupted" if attempt.get("attempt_kind") == "RETRY" else "job_interrupted",
                "model_id": model_id,
                "attempt_id": attempt_id, "timestamp": timestamp,
            })
    state.update(status="INTERRUPTED", phase="INTERRUPTED_PRESERVED", current_model_id=None)
    _atomic_json(output / "parallel-campaign-state.json", state)


def _parallel_mark_failed(output: Path, state: dict, event_path: Path,
                          error: BaseException) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    message = f"{type(error).__name__}: {error}"
    for model_id, entry in state["models"].items():
        if entry.get("status") not in {"RUNNING", "CANCELLATION_REQUESTED"}:
            continue
        entry["status"] = "FAILED"
        attempt_id = entry.get("latest_attempt_id")
        if attempt_id:
            attempt = _parallel_find_attempt(state, model_id, attempt_id)
            attempt.update(
                status="FAILED", error=message, completed_at=timestamp)
            if attempt.get("attempt_kind") == "RETRY":
                state["recovery"] = {
                    "status": "RETRY_FAILED", "model_id": model_id,
                    "attempt_id": attempt_id, "error": message,
                }
            _append_parallel_event(event_path, {
                "event": "retry_failed" if attempt.get("attempt_kind") == "RETRY" else "job_failed",
                "model_id": model_id,
                "attempt_id": attempt_id, "detail": message, "timestamp": timestamp,
            })
    state.update(status="ENGINEERING_FAILED", phase="STOPPED_ON_FAILURE", failure=message)
    _atomic_json(output / "parallel-campaign-state.json", state)


def _parallel_prepare_attempt(output: Path, state: dict, intent: dict,
                               base_manifest: dict, job: dict,
                               *, allow_retry: bool) -> dict:
    model_id = job["model_id"]
    entry = state["models"][model_id]
    attempts = entry["attempts"]
    if attempts and not allow_retry:
        raise RuntimeError(f"parallel job {model_id} already has an attempt; explicit retry is required")
    if len(attempts) >= 1 + intent["maximum_manual_retries_per_model"]:
        raise RuntimeError(f"parallel job {model_id} reached its one-manual-retry limit")
    attempt_id = f"attempt-{len(attempts) + 1:04d}"
    run_relative_path = f"runs/{model_id}/{attempt_id}"
    log_relative_path = f"logs/{model_id}-{attempt_id}.log"
    run_path = output / run_relative_path
    log_path = output / log_relative_path
    if run_path.exists() or log_path.exists():
        raise FileExistsError(f"parallel attempt output already exists: {model_id}/{attempt_id}")
    metadata_parent = output / "attempt-meta" / model_id
    metadata_parent.mkdir(parents=True, exist_ok=True)
    process_dir = metadata_parent / attempt_id
    process_dir.mkdir(exist_ok=False)
    prepared = {
        **job,
        "attempt_id": attempt_id,
        "run_relative_path": run_relative_path,
        "log_relative_path": log_relative_path,
        "job_identity_sha256": entry["job_identity_sha256"],
        "source_commit": base_manifest["source_commit"],
        "parallel_manifest_sha256": state.get("active_campaign_manifest_sha256", intent["sha256"]),
        "process_record_path": process_dir / "process-identity.json",
    }
    attempt = {
        "attempt_id": attempt_id, "job_identity_sha256": entry["job_identity_sha256"],
        "status": "LAUNCHING", "attempt_kind": "RETRY" if allow_retry else "INITIAL",
        "run_relative_path": run_relative_path,
        "log_relative_path": log_relative_path,
        "attempt_record_path": f"attempt-meta/{model_id}/{attempt_id}.json",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_new_json(output / attempt["attempt_record_path"], {
        "schema_version": "f1rl_wp3_parallel_job_attempt_v1",
        "model_id": model_id, "attempt_id": attempt_id,
        "job_identity_sha256": entry["job_identity_sha256"],
        "intent_sha256": intent["sha256"],
        "source_commit": base_manifest["source_commit"],
        "source_tree_sha256": base_manifest["source_tree_sha256"],
        "base_campaign_config_sha256": intent["base_campaign_config_sha256"],
        "parallel_config_sha256": intent["parallel_config_sha256"],
        "run_relative_path": run_relative_path, "log_relative_path": log_relative_path,
    })
    attempts.append(attempt)
    entry.update(status="RUNNING", latest_attempt_id=attempt_id)
    state.update(status="RUNNING", current_model_id=model_id)
    if allow_retry:
        state["recovery"] = {
            "status": "RETRY_RUNNING", "model_id": model_id,
            "attempt_id": attempt_id,
        }
    _atomic_json(output / "parallel-campaign-state.json", state)
    _append_parallel_event(output / "parallel-campaign-events.jsonl", {
        "event": "retry_running" if allow_retry else "job_dispatched",
        "model_id": model_id, "attempt_id": attempt_id,
        "job_identity_sha256": entry["job_identity_sha256"],
        "source_commit": base_manifest["source_commit"],
        "timestamp": attempt["started_at"],
    })
    return prepared


def _parallel_run_campaign_stage(root: Path, output: Path, state: dict, intent: dict,
                                 base_manifest: dict, campaign_config: dict,
                                 parallel_config: dict, jobs: list[dict], workers: int,
                                 *, retry_model_id: str | None = None,
                                 gate_result=None) -> dict:
    from .process_resources import sample_process_tree

    event_path = output / "parallel-campaign-events.jsonl"
    prior_metrics = state.get("runtime_metrics", {})
    stats = {
        "active_worker_seconds": prior_metrics.get("active_worker_seconds", 0.0),
        "worker_capacity_seconds": prior_metrics.get("worker_capacity_seconds", 0.0),
        "aggregate_peak_rss_bytes": prior_metrics.get("aggregate_peak_rss_bytes", 0),
        "aggregate_peak_process_count": prior_metrics.get("aggregate_peak_process_count", 0),
        "aggregate_peak_cpu_seconds": prior_metrics.get("aggregate_peak_cpu_seconds", 0.0),
        "aggregate_cpu_seconds_total": prior_metrics.get("aggregate_cpu_seconds_total", 0.0),
        "process_cleanup_seconds": prior_metrics.get("process_cleanup_seconds", 0.0),
    }
    stage_capacity = workers
    retry_authorized = retry_model_id

    def on_dispatch(job: dict) -> dict:
        return _parallel_prepare_attempt(
            output, state, intent, base_manifest, job,
            allow_retry=job["model_id"] == retry_authorized,
        )

    def run_child(job: dict, *, abort_event) -> dict:
        result = _run_child(
            root, output / job["run_relative_path"], output / job["log_relative_path"],
            job, campaign_config["training"], abort_event=abort_event,
            resources=parallel_config["resources"],
            process_record_path=job["process_record_path"],
        )
        return {
            **result, "model_id": job["model_id"], "attempt_id": job["attempt_id"],
            "run_relative_path": job["run_relative_path"],
            "log_relative_path": job["log_relative_path"],
        }

    def validate_job(job: dict, result: dict) -> dict:
        if result.get("returncode") != 0 or result.get("cancelled"):
            raise RuntimeError("parallel child did not exit successfully")
        return _parallel_validate_attempt(root, output, job, base_manifest, campaign_config)

    def on_event(event: str, job: dict, detail) -> None:
        model_id = job.get("model_id")
        timestamp = datetime.now(timezone.utc).isoformat()
        if event == "job_dispatched":
            return
        if model_id in state["models"] and job.get("attempt_id"):
            entry = state["models"][model_id]
            attempt = _parallel_find_attempt(state, model_id, job["attempt_id"])
            if event == "job_engineering_pass":
                validation = detail["validation"]
                result = detail["result"]
                attempt.update(
                    status="ENGINEERING_PASS", validation_sha256=validation["sha256"],
                    validation=validation, returncode=result["returncode"],
                    elapsed_seconds=result.get("elapsed_seconds"),
                    startup_seconds=result.get("startup_seconds"),
                    cleanup_seconds=result.get("cleanup_seconds"),
                    resource_usage=result.get("resource_usage"),
                    process_identity=result.get("process_identity"),
                    completed_at=timestamp,
                )
                entry.update(status="ENGINEERING_PASS", latest_attempt_id=job["attempt_id"],
                             validation=validation, output_path=job["run_relative_path"])
                if attempt.get("attempt_kind") == "RETRY":
                    state["recovery"] = {
                        "status": "RETRY_PASSED", "model_id": model_id,
                        "attempt_id": job["attempt_id"],
                    }
                if model_id not in state["completed_model_ids"]:
                    state["completed_model_ids"].append(model_id)
                state["completed_training_timesteps"] = (
                    len(state["completed_model_ids"]) * campaign_config["training"]["timesteps_per_model"])
                stats["process_cleanup_seconds"] += result.get("cleanup_seconds", 0.0)
                stats["aggregate_cpu_seconds_total"] += (
                    result.get("resource_usage", {}).get("final_process_tree_cpu_seconds") or 0.0)
            elif event == "job_failed":
                attempt.update(status="FAILED", error=str(detail), completed_at=timestamp)
                entry.update(status="FAILED", latest_attempt_id=job["attempt_id"], last_error=str(detail))
                if attempt.get("attempt_kind") == "RETRY":
                    state["recovery"] = {
                        "status": "RETRY_FAILED", "model_id": model_id,
                        "attempt_id": job["attempt_id"], "error": str(detail),
                    }
            elif event == "job_cancelled":
                attempt.update(status="CANCELLED", completed_at=timestamp,
                               elapsed_seconds=detail.get("elapsed_seconds"),
                               cleanup_seconds=detail.get("cleanup_seconds"),
                               resource_usage=detail.get("resource_usage"))
                entry.update(status="CANCELLED", latest_attempt_id=job["attempt_id"])
                if attempt.get("attempt_kind") == "RETRY":
                    state["recovery"] = {
                        "status": "RETRY_FAILED", "model_id": model_id,
                        "attempt_id": job["attempt_id"], "error": "retry attempt cancelled",
                    }
                stats["process_cleanup_seconds"] += detail.get("cleanup_seconds", 0.0)
                stats["aggregate_cpu_seconds_total"] += (
                    detail.get("resource_usage", {}).get("final_process_tree_cpu_seconds") or 0.0)
            elif event == "cancellation_requested":
                attempt["status"] = "CANCELLATION_REQUESTED"
        elif event == "resource_budget_exceeded":
            state["resource_budget_failure"] = str(detail)
        if event == "resource_budget_exceeded":
            state["status"] = "ENGINEERING_FAILED"
            state["phase"] = "STOPPED_ON_RESOURCE_BUDGET"
        _atomic_json(output / "parallel-campaign-state.json", state)
        _append_parallel_event(event_path, {
            "event": event, "model_id": model_id,
            "attempt_id": job.get("attempt_id"),
            "detail": (str(detail) if event in {"job_failed", "resource_budget_exceeded"} else None),
            "timestamp": timestamp,
        })

    def on_tick(active_jobs: tuple[dict, ...], delta_seconds: float) -> str | None:
        stats["active_worker_seconds"] += len(active_jobs) * delta_seconds
        stats["worker_capacity_seconds"] += stage_capacity * delta_seconds
        rss_total = 0
        cpu_total = 0.0
        process_count = 0
        for job in active_jobs:
            identity_path = job["process_record_path"]
            if not identity_path.is_file():
                continue
            identity, _ = _strict_json(identity_path)
            if (identity.get("model_id") != job["model_id"]
                    or identity.get("attempt_id") != job["attempt_id"]
                    or identity.get("source_commit") != base_manifest["source_commit"]
                    or identity.get("parallel_manifest_sha256") != job["parallel_manifest_sha256"]):
                return "child PID identity record differs from the dispatched job"
            sample = sample_process_tree(identity["pid"])
            if sample is None:
                continue
            root_row = next((row for row in sample["processes"] if row["pid"] == identity["pid"]), None)
            if (root_row is None or root_row["creation_time_100ns"] != identity["creation_time_100ns"]):
                continue
            rss_total += sample["rss_bytes"]
            cpu_total += sample["cpu_seconds"]
            process_count += sample["process_count"]
            attempt = _parallel_find_attempt(state, job["model_id"], job["attempt_id"])
            if not attempt.get("process_identity"):
                attempt["process_identity"] = identity
                _atomic_json(output / "parallel-campaign-state.json", state)
                _append_parallel_event(event_path, {
                    "event": "child_process_observed", "model_id": job["model_id"],
                    "attempt_id": job["attempt_id"], "pid": identity["pid"],
                    "creation_time_100ns": identity["creation_time_100ns"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
        stats["aggregate_peak_rss_bytes"] = max(stats["aggregate_peak_rss_bytes"], rss_total)
        stats["aggregate_peak_process_count"] = max(stats["aggregate_peak_process_count"], process_count + 1)
        stats["aggregate_peak_cpu_seconds"] = max(stats["aggregate_peak_cpu_seconds"], cpu_total)
        resources = parallel_config["resources"]
        if rss_total > min(resources["max_total_process_tree_rss_bytes"],
                           len(active_jobs) * resources["per_worker_process_tree_rss_limit_bytes"]):
            return "aggregate active process-tree RSS exceeded the admitted memory budget"
        if process_count + 1 > resources["max_total_processes"]:
            return "aggregate child and supervisor process count exceeded the admitted limit"
        return None

    def on_gate_result(stage: dict) -> None:
        if gate_result is not None:
            gate_result(stage)

    prepared_jobs = [dict(row) for row in jobs]
    if prepared_jobs and prepared_jobs[0]["model_id"] == "C-seed0":
        try:
            result = run_parallel_schedule(
                prepared_jobs, workers, run_child, validate_job,
                on_dispatch=on_dispatch, on_event=on_event, on_tick=on_tick,
                on_gate_result=on_gate_result,
                tick_interval_seconds=parallel_config["resources"]["memory_poll_interval_seconds"],
            )
        except KeyboardInterrupt:
            _parallel_mark_interrupted(output, state, event_path)
            raise
        except Exception as error:
            _parallel_mark_failed(output, state, event_path, error)
            raise
    else:
        try:
            result = _run_bounded_stage(
                prepared_jobs, workers, run_child, validate_job,
                on_dispatch=on_dispatch, on_event=on_event, on_tick=on_tick,
                tick_interval_seconds=parallel_config["resources"]["memory_poll_interval_seconds"],
            )
        except KeyboardInterrupt:
            _parallel_mark_interrupted(output, state, event_path)
            raise
        except Exception as error:
            _parallel_mark_failed(output, state, event_path, error)
            raise
    total_wall_seconds = (datetime.now(timezone.utc) - datetime.fromisoformat(state["created_at"])).total_seconds()
    per_job_metrics = {}
    failure_count = 0
    retry_count = 0
    for model_id, entry in state["models"].items():
        attempts = entry["attempts"]
        retry_count += max(0, len(attempts) - 1)
        failure_count += sum(attempt.get("status") in {"FAILED", "CANCELLED", "INTERRUPTED"}
                             for attempt in attempts)
        latest = attempts[-1] if attempts else {}
        per_job_metrics[model_id] = {
            "attempt_id": latest.get("attempt_id"),
            "status": entry.get("status"),
            "elapsed_seconds": latest.get("elapsed_seconds"),
            "startup_seconds": latest.get("startup_seconds"),
            "cleanup_seconds": latest.get("cleanup_seconds"),
            "resource_usage": latest.get("resource_usage"),
        }
    state["runtime_metrics"] = {
        **stats,
        "campaign_wall_seconds": max(0.0, total_wall_seconds),
        "completed_jobs_per_hour": (len(state["completed_model_ids"]) * 3600 /
                                     max(0.001, total_wall_seconds)),
        "scheduler_idle_seconds": max(0.0, stats["worker_capacity_seconds"] - stats["active_worker_seconds"]),
        "worker_utilization": (stats["active_worker_seconds"] / stats["worker_capacity_seconds"]
                               if stats["worker_capacity_seconds"] > 0 else 0.0),
        "training_transitions_per_second": (
            state["completed_training_timesteps"] /
            max(0.001, total_wall_seconds)),
        "aggregate_cpu_percent_of_host": (
            100.0 * stats["aggregate_cpu_seconds_total"] /
            max(0.001, total_wall_seconds * (os.cpu_count() or 1))),
        "ppo_rollout_seconds": None,
        "ppo_optimizer_seconds": None,
        "per_job": per_job_metrics,
        "failure_count": failure_count,
        "retry_count": retry_count,
    }
    _atomic_json(output / "parallel-campaign-state.json", state)
    return result


def run_parallel_campaign(repository_root: str | Path, output_directory: str | Path,
                          *, workers: int | None = None, resume: bool = False,
                          retry_model_id: str | None = None) -> dict:
    root = Path(repository_root).resolve(strict=True)
    output_arg = Path(output_directory)
    if not output_arg.is_absolute():
        raise ValueError("parallel campaign evidence root must be an absolute path")
    output = output_arg.resolve()
    if output.is_relative_to(root):
        raise ValueError("parallel campaign evidence must be outside the repository")
    base_manifest, campaign_config, schedule = build_campaign_manifest(root)
    parallel_config, parallel_config_sha256 = load_parallel_contract(root)
    if workers is None:
        if resume:
            prior_intent, _ = _strict_json(output / "WP3-PARALLEL-INTENT.json")
            workers = prior_intent.get("worker_count")
        else:
            workers = parallel_config["default_concurrency"]
    if type(workers) is not int or workers not in parallel_config["allowed_concurrency"]:
        raise ValueError("parallel worker count must be one of 1, 2, or 4")
    intent = _parallel_intent(
        base_manifest, schedule, parallel_config_sha256, workers,
        parallel_config["recovery"]["maximum_manual_retries_per_model"],
    )
    if retry_model_id is not None and retry_model_id not in {row["model_id"] for row in schedule}:
        raise ValueError("explicit retry must name one of the frozen campaign jobs")
    initial_preflight = None
    if not resume:
        if retry_model_id is not None:
            raise ValueError("a retry requires an existing parallel campaign and --resume")
        initial_preflight = _parallel_preflight(root, output, parallel_config, campaign_config, workers)
        state = _new_parallel_output(root, output, intent, schedule, initial_preflight,
                                     base_manifest, campaign_config)
    else:
        intent, state = _load_parallel_output(root, output, intent, base_manifest, schedule)

    lock_path = output / ".campaign.lock"
    event_path = output / "parallel-campaign-events.jsonl"
    with _CampaignLock(lock_path):
        _parallel_assert_no_live_children(output, state)
        _parallel_revalidate_state(root, output, state, base_manifest, campaign_config, schedule)
        if state.get("status") in ("COMPLETE", "EVALUATION_WITH_INVALID_CELLS"):
            if retry_model_id is not None:
                raise RuntimeError("a fully validated parallel job cannot be retried")
            if (output / "WP3-EVIDENCE-MANIFEST.json").is_file():
                _verify_evidence_manifest(output)
                return state

        has_activation = bool(state.get("active_c0_gate_path"))
        gate = decision = None
        if has_activation:
            gate, decision = _verify_parallel_activation(root, output, state, intent,
                                                          base_manifest, campaign_config, schedule)
        elif resume and retry_model_id not in (None, "C-seed0"):
            raise RuntimeError("C0 must establish its immutable gate before another job can be retried")
        elif resume and state["models"]["C-seed0"]["attempts"] and retry_model_id != "C-seed0":
            raise RuntimeError("C0 was interrupted or lacks its immutable gate; explicit --retry-model C-seed0 is required")

        if gate is not None and gate["status"] == "ENGINEERING_FAIL":
            if retry_model_id != "C-seed0":
                state.update(status="C0_ENGINEERING_FAILED", phase="STOPPED_BEFORE_QUEUE")
                c0_entry = state["models"]["C-seed0"]
                c0_entry.update(status="GATE_FAILED", last_error=gate.get("failure"))
                state["completed_model_ids"] = [model_id for model_id in state["completed_model_ids"]
                                                 if model_id != "C-seed0"]
                state["completed_training_timesteps"] = (
                    len(state["completed_model_ids"]) * campaign_config["training"]["timesteps_per_model"])
                _atomic_json(output / "parallel-campaign-state.json", state)
                return state
            has_activation = False
        if has_activation and gate and gate["status"] == "ENGINEERING_PASS" and retry_model_id == "C-seed0":
            raise RuntimeError("C0 already passed its immutable gate and cannot be retried")
        if retry_model_id == "C-seed0":
            state["recovery"] = {"status": "RETRY_REQUESTED", "model_id": "C-seed0",
                                 "unresolved_model_ids": ["C-seed0"]}
            _atomic_json(output / "parallel-campaign-state.json", state)
            _append_parallel_event(event_path, {
                "event": "retry_requested", "model_id": "C-seed0",
                "unresolved_model_ids": ["C-seed0"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        needs_worker_preflight = (not has_activation or retry_model_id is not None or any(
            entry.get("status") == "PENDING" for entry in state["models"].values()))
        if resume and needs_worker_preflight:
            preflight = _parallel_preflight(root, output, parallel_config, campaign_config, workers)
            state["host_preflight_latest"] = preflight
            _atomic_json(output / "parallel-campaign-state.json", state)

        def record_gate(stage: dict) -> None:
            gate_doc = _parallel_gate_payload(output, state, intent, base_manifest, stage)
            _parallel_write_activation(output, state, intent, base_manifest, workers, gate_doc)
            _append_parallel_event(event_path, {
                "event": "c0_gate_decision", "attempt_id": gate_doc["attempt_id"],
                "status": gate_doc["status"], "gate_sha256": gate_doc["sha256"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        if not has_activation:
            c0 = [dict(row) for row in schedule if row["model_id"] == "C-seed0"]
            gate_result = _parallel_run_campaign_stage(
                root, output, state, intent, base_manifest, campaign_config,
                parallel_config, c0, 1,
                retry_model_id=("C-seed0" if retry_model_id == "C-seed0" else None),
                gate_result=record_gate,
            )
            retry_model_id = None
            gate, decision = _verify_parallel_activation(root, output, state, intent,
                                                         base_manifest, campaign_config, schedule)
            if gate["status"] != "ENGINEERING_PASS":
                if state.get("recovery", {}).get("model_id") == "C-seed0":
                    state["recovery"].update(status="RETRY_FAILED", unresolved_model_ids=["C-seed0"])
                    _atomic_json(output / "parallel-campaign-state.json", state)
                return state
            if state.get("recovery", {}).get("model_id") == "C-seed0":
                state["recovery"].update(status="SAFE_TO_RESUME_PENDING_QUEUE", unresolved_model_ids=[])
                _atomic_json(output / "parallel-campaign-state.json", state)
                _append_parallel_event(event_path, {
                    "event": "pending_queue_resumption_allowed", "model_id": "C-seed0",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
            preflight = _parallel_preflight(root, output, parallel_config, campaign_config, workers)
            state["host_preflight_latest"] = preflight
            _atomic_json(output / "parallel-campaign-state.json", state)

        if gate is None or gate.get("status") != "ENGINEERING_PASS":
            raise RuntimeError("parallel scheduling cannot continue without a verified C0 engineering gate")

        if retry_model_id:
            retry_entry = state["models"][retry_model_id]
            if retry_model_id == "C-seed0":
                raise RuntimeError("C0 retry was already consumed")
            if retry_entry.get("status") == "ENGINEERING_PASS":
                raise RuntimeError("a fully validated parallel job cannot be retried")
            if not retry_entry.get("attempts") or retry_entry.get("status") not in {
                    "FAILED", "CANCELLED", "CANCELLATION_REQUESTED", "RUNNING", "INTERRUPTED"}:
                raise RuntimeError("explicit retry requires one failed, cancelled, or interrupted attempt")
            if len(retry_entry["attempts"]) >= 1 + intent["maximum_manual_retries_per_model"]:
                raise RuntimeError("parallel job has reached its one-manual-retry limit")
            unresolved_ids = [model_id for model_id, entry in state["models"].items()
                              if entry.get("status") in {"FAILED", "CANCELLED", "CANCELLATION_REQUESTED", "INTERRUPTED", "RUNNING"}]
            state["recovery"] = {
                "status": "RETRY_REQUESTED", "model_id": retry_model_id,
                "unresolved_model_ids": unresolved_ids,
            }
            _atomic_json(output / "parallel-campaign-state.json", state)
            _append_parallel_event(event_path, {
                "event": "retry_requested", "model_id": retry_model_id,
                "unresolved_model_ids": unresolved_ids,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        else:
            unresolved = [model_id for model_id, entry in state["models"].items()
                          if entry.get("status") in {"FAILED", "CANCELLED", "CANCELLATION_REQUESTED", "INTERRUPTED", "RUNNING"}]
            if unresolved:
                raise RuntimeError(f"parallel resume requires explicit retries for failed attempts: {unresolved}")

        if retry_model_id:
            retry_job = next(row for row in schedule if row["model_id"] == retry_model_id)
            stage_result = _parallel_run_campaign_stage(
                root, output, state, intent, base_manifest, campaign_config,
                parallel_config, [dict(retry_job)], workers,
                retry_model_id=retry_model_id,
            )
            if stage_result["status"] != "ENGINEERING_PASS":
                state.update(status="ENGINEERING_FAILED", phase="STOPPED_ON_FAILURE")
                _atomic_json(output / "parallel-campaign-state.json", state)
                return state
            unresolved_ids = [model_id for model_id, entry in state["models"].items()
                              if entry.get("status") in {"FAILED", "CANCELLED", "CANCELLATION_REQUESTED", "INTERRUPTED", "RUNNING"}]
            if unresolved_ids:
                state["recovery"] = {
                    "status": "UNRESOLVED_FAILURES_REMAIN",
                    "model_id": retry_model_id, "unresolved_model_ids": unresolved_ids,
                }
                state.update(status="WAITING_FOR_EXPLICIT_RETRY",
                             phase="WAITING_FOR_EXPLICIT_RETRY", current_model_id=None)
                _atomic_json(output / "parallel-campaign-state.json", state)
                _append_parallel_event(event_path, {
                    "event": "unresolved_failures_remain", "model_id": retry_model_id,
                    "unresolved_model_ids": unresolved_ids,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                return state
            state["recovery"] = {
                "status": "SAFE_TO_RESUME_PENDING_QUEUE", "model_id": retry_model_id,
                "unresolved_model_ids": [],
            }
            _atomic_json(output / "parallel-campaign-state.json", state)
            _append_parallel_event(event_path, {
                "event": "pending_queue_resumption_allowed", "model_id": retry_model_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        pending_jobs = [dict(row) for row in schedule
                        if row["model_id"] != "C-seed0"
                        and state["models"][row["model_id"]].get("status") == "PENDING"]
        if pending_jobs:
            stage_result = _parallel_run_campaign_stage(
                root, output, state, intent, base_manifest, campaign_config,
                parallel_config, pending_jobs, workers,
            )
            if stage_result["status"] != "ENGINEERING_PASS":
                state.update(status="ENGINEERING_FAILED", phase="STOPPED_ON_FAILURE")
                _atomic_json(output / "parallel-campaign-state.json", state)
                return state

        if len(state["completed_model_ids"]) != 12:
            state.update(status="ENGINEERING_FAILED", phase="WAITING_FOR_EXPLICIT_RETRY")
            _atomic_json(output / "parallel-campaign-state.json", state)
            return state

        effective_schedule = []
        for model in schedule:
            entry = state["models"][model["model_id"]]
            effective_schedule.append({
                **model,
                "run_relative_path": entry["output_path"],
            })
        state.update(status="TRAINING_COMPLETE", phase="STAGE_3_TRAIN_EVALUATION",
                     current_model_id=None)
        _atomic_json(output / "parallel-campaign-state.json", state)
        _write_training_deliverables(output, state, effective_schedule, base_manifest, campaign_config)
        from .evaluation import run_evaluation
        evaluation = run_evaluation(root, output, base_manifest, campaign_config, effective_schedule)
        state.update(
            status=("COMPLETE" if evaluation["status"] == "PASS" else "EVALUATION_WITH_INVALID_CELLS"),
            phase="COMPLETE" if evaluation["status"] == "PASS" else "EVALUATION_REVIEW",
            evaluation=evaluation, current_model_id=None,
        )
        _atomic_json(output / "parallel-campaign-state.json", state)
        _write_parallel_review_status(output, state, evaluation)
        _write_evidence_manifest(output, allow_incomplete_attempt_artifacts=True)
        return state

def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Explicit bounded WP3 parallel campaign supervisor")
    parser.add_argument("command", choices=("parallel-run", "parallel-resume", "parallel-retry"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--retry-model", choices=tuple(f"{candidate}-seed{seed}"
                                                         for candidate in "ABCD" for seed in range(3)))
    args = parser.parse_args(argv)
    if args.command == "parallel-retry" and args.retry_model is None:
        parser.error("parallel-retry requires --retry-model")
    if args.command != "parallel-retry" and args.retry_model is not None:
        parser.error("--retry-model is only valid with parallel-retry")
    root = Path(__file__).resolve().parents[2]
    state = run_parallel_campaign(
        root, args.output, workers=args.workers,
        resume=args.command != "parallel-run", retry_model_id=args.retry_model,
    )
    print(f"[WP3-PARALLEL] {state['status']} completed="
          f"{len(state['completed_model_ids'])}/12 "
          f"timesteps={state['completed_training_timesteps']}", flush=True)
    return 0 if state["status"] == "COMPLETE" else 2
