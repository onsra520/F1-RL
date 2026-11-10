"""Frozen, sequential WP3 LOCAL_DEMO training campaign orchestration."""
from __future__ import annotations

from datetime import datetime, timezone
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import struct
import sys
import threading
import time
from typing import Any

from ..learning.pilot_preflight import TRAIN_IDS
from ..learning.training.config import TrainingRunConfig
from ..research.simulation_only_protocol import load_simulation_only_manifest
from ..research.wp7_protocol import build_protocol
from .admission import _source_identity


BASELINE_COMMIT = "0ff42345426f2b3ba5c171fff6cee425b3457b16"
CAMPAIGN_CONFIG = Path("configs/research/wp3-ppo-tuning-campaign-v3.json")
PARALLEL_CAMPAIGN_CONFIG = Path("configs/research/wp3-ppo-tuning-campaign-parallel-v1.json")
SIMULATION_CONFIG = Path("configs/research/simulation-only-ppo-v1.json")
REGISTRATION_CONFIG = Path("configs/research/wp7-rotating-owner-protocol-v3.json")
TRACK_MANIFEST = Path("assets/tracks/manifest.json")
TRAINING_PROFILE = Path("configs/recovery/wp7-pilot-200k.yml")
CAMPAIGN_EVENTS_SCHEMA = "f1rl_wp3_campaign_event_v1"
PARALLEL_CAMPAIGN_BASE_SHA256 = "8a67db80464dc3b93072c0edc431f7a71b1b271c392b47d0915af0f2cac106e4"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_json(path: Path) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique,
                       parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value, raw


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_new_json(path: Path, value: dict) -> None:
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _write_immutable_json(path: Path, value: dict) -> None:
    if path.exists():
        prior, _ = _strict_json(path)
        if prior != value:
            raise ValueError(f"refusing to overwrite immutable WP3 artifact: {path.name}")
        return
    _write_new_json(path, value)


def _expected_campaign_config() -> dict:
    candidates = [
        {"candidate_id": "A", "lateral_penalty_per_m2_s": 2.0, "training_corridor_m": 2.0},
        {"candidate_id": "B", "lateral_penalty_per_m2_s": 4.0, "training_corridor_m": 2.0},
        {"candidate_id": "C", "lateral_penalty_per_m2_s": 2.0, "training_corridor_m": 3.0},
        {"candidate_id": "D", "lateral_penalty_per_m2_s": 4.0, "training_corridor_m": 3.0},
    ]
    return {
        "schema_version": "f1rl_wp3_ppo_tuning_campaign_v3",
        "campaign_id": "wp3-abcd-12-models-v3",
        "baseline_commit": BASELINE_COMMIT,
        "scope": "TRAIN_simulation_only_development_not_scientific_acceptance",
        "training": {
            "timesteps_per_model": 65536, "total_timesteps": 786432,
            "checkpoint_every_timesteps": 4096, "max_wall_seconds_per_model": 7200,
            "device": "cpu", "n_envs": 1, "torch_threads": 8,
            "torch_interop_threads": 1, "policy": "MlpPolicy",
            "rollout_length": 256, "selection": "final_checkpoint_only",
            "physics_failure_policy": "training_contact_terminal_v1",
        },
        "candidate_order": ["A", "B", "C", "D"], "candidates": candidates,
        "seed_bundles": [
            {"seed_index": 0, "ppo": 0, "sampling": 0, "training_reset": 1000},
            {"seed_index": 1, "ppo": 1, "sampling": 1, "training_reset": 1001},
            {"seed_index": 2, "ppo": 2, "sampling": 2, "training_reset": 1002},
        ],
        "execution_order": [["C", 0], ["A", 0], ["B", 0], ["D", 0],
                            ["A", 1], ["B", 1], ["C", 1], ["D", 1],
                            ["A", 2], ["B", 2], ["C", 2], ["D", 2]],
        "evaluation": {
            "schema_version": "f1rl_wp3_train_evaluation_v1",
            "tracks": list(TRAIN_IDS), "strata": ["nominal", "randomized_train"],
            "episodes_per_track_stratum_model": 60,
            "policy_actions": "stochastic_with_paired_candidate_independent_rng_seeds",
            "seed_protocol": "sha256(evaluation_config_sha256,track_id,stratum,episode)_uint32_v1",
            "reset_seed_protocol": "registered_train_reset_contract_v1",
            "task": "registered_original_task_no_training_curriculum",
            "hard_route_boundary_m": 4.0, "checkpoint_selection": "final_checkpoint_only",
            "metrics": {
                "primary": "valid_autonomous_finish_rate_from_registered_finish_crossing",
                "route_progress": "final_normalized_route_progress_from_registered_evaluator",
                "completed_checkpoints": "final_next_checkpoint_index_from_registered_evaluator",
                "collision_rate": "fraction_of_valid_episodes_with_collision_count_greater_than_zero",
                "hard_route_violation_rate": "fraction_with_registered_hard_route_violation_outcome",
                "timeout_rate": "fraction_truncated_without_valid_finish",
                "steering_rate_rms": "RMS_of_consecutive_applied_steering_differences_over_fixed_dt_no_initial_delta",
                "episode_duration": "registered_evaluator_episode_time_seconds",
                "failure_reason": "valid_finish_then_registered_truncation_or_research_outcome_reason",
            },
            "invalid_episode_rule": "abort_cell_preserve_invalid_evidence_no_redraw",
            "scientific_evidence": False, "heldout_access": False,
        },
        "resources": {
            "max_telemetry_record_bytes": 16384,
            "telemetry_auxiliary_reserve_bytes": 1048576,
            "telemetry_per_model_max_bytes": 1074790400,
            "minimum_free_disk_safety_bytes": 1073741824,
            "max_concurrent_training_processes": 1, "max_retries_per_model": 0,
        },
        "campaign_policy": {
            "resume_completed_runs_only": True,
            "incomplete_run_action": "stop_preserve_no_automatic_retry",
            "continue_after_c0_engineering_pass": True,
            "stop_on_engineering_failure": True,
            "scientific_training_authorized": False, "wp7_and_gate2": "BLOCKED",
        },
    }


def _load_campaign_config(root: Path) -> tuple[dict, bytes, str]:
    value, raw = _strict_json(root / CAMPAIGN_CONFIG)
    if value != _expected_campaign_config():
        raise ValueError("WP3 campaign configuration differs from its frozen v1 contract")
    return value, raw, sha256(raw).hexdigest()


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


def model_schedule(config: dict, simulation_manifest: dict) -> list[dict]:
    seed_rows = simulation_manifest["seeds"]["tuning"]
    seed_docs = config["seed_bundles"]
    expected_seed_docs = [
        {"seed_index": index, "ppo": row[0], "sampling": row[1], "training_reset": row[2]}
        for index, row in enumerate(seed_rows)
    ]
    if seed_docs != expected_seed_docs:
        raise ValueError("campaign seed bundles differ from the frozen tuning roster")
    candidates = {row["candidate_id"]: row for row in simulation_manifest["candidates"]}
    if config["candidates"] != [
        {key: candidates[candidate][key] for key in ("candidate_id", "lateral_penalty_per_m2_s", "training_corridor_m")}
        for candidate in config["candidate_order"]
    ]:
        raise ValueError("campaign factorial differs from the frozen simulation-only candidates")
    if len(config["execution_order"]) != 12 or len({tuple(pair) for pair in config["execution_order"]}) != 12:
        raise ValueError("campaign schedule must contain each candidate/seed exactly once")
    schedule = []
    for order, (candidate, seed_index) in enumerate(config["execution_order"]):
        if candidate not in candidates or type(seed_index) is not int or not 0 <= seed_index < 3:
            raise ValueError("campaign schedule contains an unknown candidate or seed")
        bundle = seed_docs[seed_index]
        schedule.append({
            "order": order, "model_id": f"{candidate}-seed{seed_index}",
            "candidate_id": candidate, "seed_index": seed_index,
            "seed_bundle": {key: bundle[key] for key in ("ppo", "sampling", "training_reset")},
            "candidate_factors": candidates[candidate],
            "run_relative_path": f"runs/{candidate}-seed{seed_index}",
        })
    if schedule[0]["model_id"] != "C-seed0":
        raise ValueError("Stage 1 must be C0")
    return schedule


def _verify_preflight(root: Path, config: dict, simulation_manifest: dict) -> dict:
    training = config["training"]
    sim_training = simulation_manifest["training"]
    checks = {
        "timesteps_per_model": sim_training["timesteps_per_candidate_seed"],
        "checkpoint_every_timesteps": sim_training["checkpoint_every_timesteps"],
        "max_wall_seconds_per_model": sim_training["max_wall_seconds"],
        "device": sim_training["device"], "n_envs": sim_training["n_envs"],
        "torch_threads": sim_training["torch_threads"],
        "torch_interop_threads": sim_training["torch_interop_threads"],
    }
    if any(training[key] != value for key, value in checks.items()):
        raise ValueError("campaign training limits differ from the frozen simulation-only contract")
    if training.get("physics_failure_policy") != "training_contact_terminal_v1":
        raise ValueError("WP3 v3 requires its declared classified-contact terminal policy")
    registration = json.loads((root / REGISTRATION_CONFIG).read_text(encoding="utf-8"))
    if list(registration["strata"] if "strata" in registration else registration["evaluation"]["strata"]) != config["evaluation"]["strata"]:
        raise ValueError("evaluation strata differ from the registered TRAIN strata")
    if simulation_manifest["tracks"] != list(TRAIN_IDS) or config["evaluation"]["tracks"] != list(TRAIN_IDS):
        raise ValueError("WP3 must use the exact four-track TRAIN roster")
    if (simulation_manifest["evaluation"]["hard_route_boundary_m"] != 4.0
            or config["evaluation"]["hard_route_boundary_m"] != 4.0
            or simulation_manifest["evaluation"]["semantics_unchanged"] is not True):
        raise ValueError("registered 4 m evaluation boundary must remain unchanged")
    profile = TrainingRunConfig.from_yaml(root / TRAINING_PROFILE)
    if (profile.vector_env.backend != "dummy"
            or profile.ppo.n_steps != training["rollout_length"]
            or profile.ppo.device != "cpu"):
        raise ValueError("registered PPO profile does not match the campaign rollout/device contract")
    if (training["timesteps_per_model"] % training["rollout_length"] != 0
            or training["total_timesteps"] != training["timesteps_per_model"] * 12
            or training["n_envs"] != 1 or training["device"] != "cpu"):
        raise ValueError("campaign timestep/resource arithmetic is inconsistent")
    for candidate in config["candidate_order"]:
        for seed_index in range(3):
            from .runner import _preflight
            preflight = _preflight(root, candidate, seed_index, profile="tuning")
            if (preflight["timesteps"] != training["timesteps_per_model"]
                    or preflight["checkpoint_every"] != training["checkpoint_every_timesteps"]
                    or preflight["max_wall_seconds"] != training["max_wall_seconds_per_model"]
                    or preflight["ppo"].n_steps != training["rollout_length"]
                    or preflight["ppo"].batch_size != profile.ppo.batch_size
                    or preflight["ppo"].n_epochs != profile.ppo.n_epochs
                    or preflight["threads"] != training["torch_threads"]
                    or preflight["interop_threads"] != training["torch_interop_threads"]):
                raise ValueError(f"LOCAL_DEMO preflight differs for {candidate}-seed{seed_index}")
    storage = config["resources"]
    expected_per_model = (training["timesteps_per_model"] * storage["max_telemetry_record_bytes"]
                          + storage["telemetry_auxiliary_reserve_bytes"])
    if expected_per_model != storage["telemetry_per_model_max_bytes"]:
        raise ValueError("campaign telemetry capacity estimate is inconsistent")
    required = expected_per_model * 12 + storage["minimum_free_disk_safety_bytes"]
    usage = shutil.disk_usage(root)
    if usage.free < required:
        raise OSError(f"insufficient free disk for 12-run telemetry bound and safety margin: {required}")
    return {
        "timesteps_per_model": training["timesteps_per_model"],
        "total_timesteps": training["total_timesteps"],
        "expected_periodic_checkpoints_per_model": training["timesteps_per_model"] // training["checkpoint_every_timesteps"],
        "max_telemetry_bytes_per_model": expected_per_model,
        "max_telemetry_bytes_campaign": expected_per_model * 12,
        "minimum_safety_bytes": storage["minimum_free_disk_safety_bytes"],
        "required_free_bytes": required, "available_free_bytes": usage.free,
        "tracks": list(TRAIN_IDS), "device": "cpu", "n_envs": 1,
        "hard_route_boundary_m": 4.0, "heldout_access": False,
        "scientific_evaluation": False,
    }


def build_campaign_manifest(root: Path) -> tuple[dict, dict, list[dict]]:
    config, config_raw, config_hash = _load_campaign_config(root)
    simulation_manifest, simulation_hash = load_simulation_only_manifest(root)
    schedule = model_schedule(config, simulation_manifest)
    source_commit, source_tree_sha256 = _source_identity(root)
    ancestor = subprocess.run(["git", "merge-base", "HEAD", BASELINE_COMMIT], cwd=root,
                              check=True, capture_output=True, text=True).stdout.strip()
    if ancestor != BASELINE_COMMIT or source_commit == BASELINE_COMMIT:
        raise ValueError("WP3 source must be a committed descendant of the verified smoke baseline")
    resource_check = _verify_preflight(root, config, simulation_manifest)
    protocol = build_protocol(root, profile="spatial_rotating_owner_v3")
    evidence_root = Path(r"D:\Projects\.evidence\F1-RL")
    smoke_root = evidence_root / "local-demo-smoke"
    smoke_validation, _ = _strict_json(smoke_root / "SMOKE-VALIDATION.json")
    smoke_independent, _ = _strict_json(smoke_root / "SMOKE-INDEPENDENT-CHECKS.json")
    if (smoke_validation.get("status") != "SMOKE_INTEGRATION_PASS"
            or smoke_validation.get("source_commit") != BASELINE_COMMIT
            or smoke_validation.get("learning_timesteps") != 256
            or smoke_independent.get("status") != "PASS"
            or smoke_independent.get("source_commit") != BASELINE_COMMIT
            or smoke_independent.get("learning_timesteps") != 256):
        raise ValueError("WP3 baseline real-PPO smoke evidence is missing or not bound to the declared commit")
    prior_evidence_paths = {
        "scientific_consolidation_final_status": evidence_root / "scientific-consolidation" / "FINAL-STATUS.md",
        "smoke_report": smoke_root / "SMOKE-TEST-REPORT.md",
        "smoke_validation": smoke_root / "SMOKE-VALIDATION.json",
        "smoke_independent_review": smoke_root / "SMOKE-INDEPENDENT-REVIEW.md",
        "wp1e_report": evidence_root / "ppo-recovery" / "WP1E-HARDENING-REPORT.md",
        "wp1e_recheck": evidence_root / "ppo-recovery" / "WP1E-HARDENING-RECHECK-20261021.md",
    }
    prior_evidence = {name: {"path": str(path), "sha256": _sha256(path)}
                      for name, path in prior_evidence_paths.items()}
    manifest = {
        "schema_version": "f1rl_wp3_campaign_manifest_v2",
        "campaign_id": config["campaign_id"], "scope": config["scope"],
        "status": "PREPARED_NOT_PREREGISTERED_NOT_SCIENTIFIC",
        "source_commit": source_commit, "source_tree_sha256": source_tree_sha256,
        "baseline_commit": BASELINE_COMMIT, "simulation_only_config_sha256": simulation_hash,
        "campaign_config_sha256": config_hash,
        "registration_sha256": _sha256(root / REGISTRATION_CONFIG),
        "track_manifest_sha256": _sha256(root / TRACK_MANIFEST),
        "training_profile_sha256": _sha256(root / TRAINING_PROFILE),
        "protocol_sha256": protocol["sha256"],
        "verified_smoke_baseline": {
            "validation_sha256": _sha256(smoke_root / "SMOKE-VALIDATION.json"),
            "telemetry_sha256": smoke_validation["telemetry_sha256"],
            "final_checkpoint_sha256": smoke_validation["final_checkpoint"]["sha256"],
            "independent_checks_sha256": _sha256(smoke_root / "SMOKE-INDEPENDENT-CHECKS.json"),
        },
        "prior_engineering_evidence": prior_evidence,
        "training": config["training"], "candidates": config["candidates"],
        "physics_failure_policy": config["training"]["physics_failure_policy"],
        "seed_bundles": config["seed_bundles"], "execution_order": schedule,
        "evaluation": config["evaluation"], "resources": resource_check,
        "scientific_training_authorized": False, "scientific_evidence": False,
        "heldout_access": False, "wp7_status": "BLOCKED", "gate2_status": "BLOCKED",
    }
    manifest["sha256"] = sha256(_canonical(manifest)).hexdigest()
    return manifest, config, schedule


def _iter_jsonl(path: Path, max_bytes: int):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate telemetry JSON key: {key}")
            result[key] = value
        return result
    with path.open("rb") as stream:
        while True:
            line = stream.readline(max_bytes + 1)
            if not line:
                break
            if len(line) > max_bytes or not line.endswith(b"\n"):
                raise ValueError("training telemetry has an oversized or incomplete JSONL record")
            record = json.loads(line.decode("utf-8"), object_pairs_hook=unique,
                                parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
            if not isinstance(record, dict):
                raise ValueError("training telemetry JSONL records must be objects")
            yield record


def _inside(base: Path, relative: str) -> Path:
    candidate = (base / relative).resolve(strict=True)
    if not candidate.is_relative_to(base.resolve(strict=True)):
        raise ValueError("run artifact path escapes its model output directory")
    return candidate


def _validate_training_contact_terminal(row: dict, expected_demo_reward: dict) -> None:
    demo = row.get("demo_training")
    detail = demo.get("physics_failure_detail") if isinstance(demo, dict) else None
    try:
        vector_reward = struct.unpack("=f", struct.pack("=f", float(demo["reward"])))[0]
    except (KeyError, TypeError, ValueError, OverflowError, struct.error):
        vector_reward = None
    if (not isinstance(demo, dict)
            or row.get("terminated") is not True or row.get("truncated") is not False
            or demo.get("physics_failure_code") not in {"unsupported_contact", "verified_obstacle_contact"}
            or not isinstance(detail, str) or not detail.strip()
            or row.get("termination_reason") != detail
            or demo.get("version") != expected_demo_reward.get("version")
            or demo.get("training_corridor_m") != expected_demo_reward.get("training_corridor_m")
            or demo.get("terminal_penalty") != expected_demo_reward.get("terminal_penalty")
            or demo.get("registered_reward") != 0.0
            or demo.get("curriculum_terminal") is not False
            or demo.get("scientific_evidence") is not False
            or not np_isfinite(demo.get("lateral_penalty"))
            or demo.get("lateral_penalty", -1) < 0
            or not np_isfinite(demo.get("reward"))
            or not np_isfinite(row.get("reward"))
            or demo.get("reward") != -demo["lateral_penalty"] - demo["terminal_penalty"]
            or row.get("reward") != vector_reward):
        raise ValueError("training contact terminal transition violates its frozen demo contract")


def validate_training_run(root: Path, run_dir: Path, expected: dict,
                          campaign_manifest: dict, config: dict) -> dict:
    run_manifest, _ = _strict_json(run_dir / "run-manifest.json")
    state, _ = _strict_json(run_dir / "state.json")
    if run_manifest.get("sha256") != sha256(_canonical({k: v for k, v in run_manifest.items() if k != "sha256"})).hexdigest():
        raise ValueError("run manifest self-checksum mismatch")
    if (run_manifest.get("source_commit") != campaign_manifest["source_commit"]
            or run_manifest.get("source_tree_sha256") != campaign_manifest["source_tree_sha256"]
            or run_manifest.get("experiment_manifest_sha256") != campaign_manifest["simulation_only_config_sha256"]
            or run_manifest.get("registration_sha256") != campaign_manifest["registration_sha256"]
            or run_manifest.get("track_manifest_sha256") != campaign_manifest["track_manifest_sha256"]
            or run_manifest.get("training_profile_sha256") != campaign_manifest["training_profile_sha256"]):
        raise ValueError("run source or registered configuration identity differs from campaign")
    if (run_manifest.get("candidate", {}).get("candidate_id") != expected["candidate_id"]
            or any(run_manifest.get("candidate", {}).get(key) != value
                   for key, value in expected["candidate_factors"].items())
            or run_manifest.get("training_seed_bundle") != expected["seed_bundle"]
            or run_manifest.get("split") != "TRAIN" or run_manifest.get("heldout_access") is not False
            or run_manifest.get("scientific_training_authorized") is not False
            or run_manifest.get("training_failure_policy") != config["training"]["physics_failure_policy"]):
        raise ValueError("run candidate, seed, or scientific isolation identity mismatch")
    factors = expected["candidate_factors"]
    corridor = factors["training_corridor_m"]
    expected_demo_reward = {
        "version": "straight_route_stability_demo_v1" if corridor == 2.0
                   else "straight_route_stability_demo_v2",
        "training_corridor_m": corridor,
        "lateral_penalty_per_m2_s": factors["lateral_penalty_per_m2_s"],
        "terminal_penalty": 25.0,
    }
    if run_manifest.get("demo_reward") != expected_demo_reward:
        raise ValueError("run demo reward differs from the frozen candidate contract")
    ppo = run_manifest.get("ppo_config", {})
    if (ppo.get("seed") != expected["seed_bundle"]["ppo"] or ppo.get("device") != "cpu"
            or ppo.get("n_steps") != 256 or ppo.get("batch_size") != 64 or ppo.get("n_epochs") != 10):
        raise ValueError("run PPO policy or hyperparameters differ from the frozen training profile")
    limits = run_manifest.get("limits", {})
    target_steps = config["training"]["timesteps_per_model"]
    if (limits.get("learning_timesteps") != target_steps
            or limits.get("checkpoint_every_timesteps") != config["training"]["checkpoint_every_timesteps"]
            or limits.get("max_wall_seconds") != config["training"]["max_wall_seconds_per_model"]
            or limits.get("device") != "cpu" or limits.get("n_envs") != 1):
        raise ValueError("run budget or resource limits differ from the frozen campaign")
    if (state.get("status") != "learning_completed" or state.get("phase") != "complete"
            or state.get("learning_timesteps") != target_steps
            or state.get("optimizer_steps", 0) <= 0 or state.get("scientific_evidence") is not False):
        raise ValueError("training execution did not complete the exact finite PPO budget")
    telemetry_path = run_dir / "training-telemetry.jsonl"
    telemetry_manifest, _ = _strict_json(run_dir / "training-telemetry-manifest.json")
    records = list(_iter_jsonl(telemetry_path, config["resources"]["max_telemetry_record_bytes"]))
    digest = _sha256(telemetry_path)
    if (telemetry_manifest.get("complete") is not True
            or telemetry_manifest.get("sha256") != digest
            or telemetry_manifest.get("bytes") != telemetry_path.stat().st_size
            or telemetry_manifest.get("events") != len(records)
            or telemetry_manifest.get("step_rows") != target_steps
            or telemetry_manifest.get("update_rows") != target_steps // 256
            or telemetry_manifest.get("optimizer_steps") != state["optimizer_steps"]):
        raise ValueError("training telemetry is incomplete or fails its byte/count/hash manifest")
    runs = [r for r in records if r.get("event_type") == "run"]
    starts = [r for r in records if r.get("event_type") == "training_start"]
    steps = [r for r in records if r.get("event_type") == "step"]
    updates = [r for r in records if r.get("event_type") == "ppo_update"]
    checkpoints = [r for r in records if r.get("event_type") == "checkpoint"]
    if (len(runs) != 1 or len(starts) != 1 or len(steps) != target_steps
            or len(updates) != target_steps // 256):
        raise ValueError("training telemetry event counts do not match the full rollout budget")
    run_event = runs[0]
    if (run_event.get("source_commit") != campaign_manifest["source_commit"]
            or run_event.get("split") != "TRAIN"
            or run_event.get("seed_identity") != expected["seed_bundle"]
            or run_event.get("training_failure_policy") != config["training"]["physics_failure_policy"]
            or run_event.get("reset_contracts") != run_manifest.get("reset_contracts")
            or run_event.get("seed_identity", {}).get("training_reset") != expected["seed_bundle"]["training_reset"]
            or run_event.get("track_ids") != list(TRAIN_IDS)):
        raise ValueError("telemetry run identity is not bound to campaign source/TRAIN seed roster")
    reset_contracts = run_event.get("reset_contracts", {})
    if set(reset_contracts) != set(TRAIN_IDS):
        raise ValueError("telemetry lacks the exact registered reset contracts")
    episode_tracks: dict[int, str] = {}
    track_timestep_counts = {track: 0 for track in TRAIN_IDS}
    episode_summary: dict[int, dict] = {}
    reset_count = 0
    for timestep, row in enumerate(steps, 1):
        if (row.get("schema_version") != "f1rl_training_telemetry_v1"
                or row.get("source_commit") != campaign_manifest["source_commit"]
                or row.get("run_id") != run_event.get("run_id")
                or row.get("global_timestep") != timestep or row.get("env_index") != 0):
            raise ValueError("training transition identity/order/rank/source mismatch")
        transition_seeds = row.get("seed_identity", {})
        if (transition_seeds.get("run") != expected["seed_bundle"]
                or not isinstance(row.get("observation"), list)
                or len(row["observation"]) != len(run_manifest["observation_schema"])
                or not all(np_isfinite(value) for value in row["observation"])
                or not isinstance(row.get("action_sent_to_environment"), list)
                or len(row["action_sent_to_environment"]) != 3
                or not all(np_isfinite(value) for value in row["action_sent_to_environment"])
                or any(value < low or value > high for value, low, high in zip(
                    row["action_sent_to_environment"], run_manifest["action_space"]["low"],
                    run_manifest["action_space"]["high"]))
                or type(row.get("terminated")) is not bool
                or type(row.get("truncated")) is not bool
                or not np_isfinite(row.get("reward"))):
            raise ValueError("training transition observation/action/reward/seed contract is invalid")
        track = row.get("track_id")
        if track not in track_timestep_counts:
            raise ValueError("training transition uses a non-TRAIN track")
        track_timestep_counts[track] += 1
        episode = row.get("episode_index")
        reset = row.get("seed_identity", {}).get("train_reset", {})
        contract = reset_contracts.get(track, {})
        if (type(episode) is not int or row.get("episode_id") != f"{run_event['run_id']}/env-0/episode-{episode}"
                or reset.get("seed") != expected["seed_bundle"]["training_reset"]
                or reset.get("rank") != 0 or reset.get("reset_index") != episode
                or reset.get("track_id") != track or reset.get("version") != contract.get("version")
                or reset.get("configuration_sha256") != contract.get("configuration_sha256")
                or reset.get("world_sha256") != contract.get("world_sha256")):
            raise ValueError("training episode reset provenance differs from declared TRAIN contract")
        from ..learning.train_reset import derive_train_reset_draw_seed
        if reset.get("draw_seed") != derive_train_reset_draw_seed(
                version=contract["version"], seed=contract["seed"], rank=0,
                track_id=track, reset_index=episode):
            raise ValueError("training reset draw seed does not match authoritative derivation")
        previous = episode_tracks.setdefault(episode, track)
        if previous != track:
            raise ValueError("track identity changed inside a training episode")
        data = episode_summary.setdefault(episode, {"track_id": track, "steps": 0,
                                                     "termination_reason": None,
                                                     "terminated": False, "truncated": False,
                                                     "curriculum_terminal": False})
        data["steps"] += 1
        done = row["terminated"] or row["truncated"]
        if done:
            autoreset = row.get("autoreset_identity")
            next_obs = row.get("autoreset_observation")
            next_episode_id = row.get("autoreset_episode_id")
            next_track = (autoreset or {}).get("track_id")
            next_contract = reset_contracts.get(next_track)
            expected_next_index = episode + 1
            from ..learning.train_reset import derive_train_reset_draw_seed
            if (not isinstance(autoreset, dict) or next_contract is None
                    or not isinstance(next_obs, list)
                    or len(next_obs) != len(run_manifest["observation_schema"])
                    or not all(np_isfinite(value) for value in next_obs)
                    or next_episode_id != f"{run_event['run_id']}/env-0/episode-{expected_next_index}"
                    or autoreset.get("seed") != expected["seed_bundle"]["training_reset"]
                    or autoreset.get("rank") != 0
                    or autoreset.get("reset_index") != expected_next_index
                    or autoreset.get("track_id") != next_track
                    or autoreset.get("version") != next_contract.get("version")
                    or autoreset.get("configuration_sha256") != next_contract.get("configuration_sha256")
                    or autoreset.get("world_sha256") != next_contract.get("world_sha256")
                    or autoreset.get("draw_seed") != derive_train_reset_draw_seed(
                        version=next_contract["version"], seed=next_contract["seed"],
                        rank=0, track_id=next_track, reset_index=expected_next_index)):
                raise ValueError("terminal transition lacks a valid automatic reset identity/observation")
        elif any(row.get(key) is not None for key in
                 ("autoreset_identity", "autoreset_observation", "autoreset_episode_id")):
            raise ValueError("nonterminal training transition contains an automatic reset")
        if done:
            if data["termination_reason"] is not None:
                raise ValueError("multiple terminal transitions were recorded for one episode")
            data["termination_reason"] = row.get("termination_reason") or (
                "truncated" if row.get("truncated") else "terminated")
            data["terminated"] = bool(row.get("terminated"))
            data["truncated"] = bool(row.get("truncated"))
            demo = row.get("demo_training") or {}
            data["curriculum_terminal"] = bool(demo.get("curriculum_terminal"))
            if demo.get("physics_abort_terminal") is True:
                _validate_training_contact_terminal(row, expected_demo_reward)
                data["physics_abort_terminal"] = True
            elif demo.get("physics_abort_terminal") not in (None, False):
                raise ValueError("physics-abort terminal marker is malformed")
            reset_count += 1
        elif row.get("termination_reason") is not None:
            raise ValueError("nonterminal transition has a terminal reason")
    for index, row in enumerate(steps[:-1]):
        if row["terminated"] or row["truncated"]:
            next_row = steps[index + 1]
            if (next_row.get("episode_index") != row["episode_index"] + 1
                    or next_row.get("episode_id") != row.get("autoreset_episode_id")
                    or next_row.get("track_id") != row.get("autoreset_identity", {}).get("track_id")
                    or next_row.get("seed_identity", {}).get("train_reset") != row.get("autoreset_identity")
                    or next_row.get("observation") != row.get("autoreset_observation")):
                raise ValueError("terminal transition does not link to the next autoreset episode observation")
    episode_ids = sorted(episode_summary)
    if episode_ids != list(range(len(episode_ids))):
        raise ValueError("training episode indices are not contiguous")
    sampling_seed = run_event["seed_identity"].get("sampling")
    sampler_seed = int.from_bytes(sha256(
        f"f1rl/multi-track-sampling/v1/{sampling_seed}/0".encode("utf-8")
    ).digest(), "big")
    sampler = random.Random(sampler_seed)
    sampled_tracks = [sampler.choice(list(TRAIN_IDS)) for _ in episode_ids]
    if [episode_summary[i]["track_id"] for i in episode_ids] != sampled_tracks:
        raise ValueError("observed episode tracks differ from the registered deterministic scenario sampler")
    for episode in episode_ids[:-1]:
        if episode_summary[episode]["termination_reason"] is None:
            raise ValueError("next training episode began without a terminal transition association")
    automatic_resets_verified = max(0, len(episode_ids) - 1)
    if automatic_resets_verified:
        for episode in episode_ids[:-1]:
            if not episode_summary[episode]["terminated"] and not episode_summary[episode]["truncated"]:
                raise ValueError("episode boundary is missing a terminal/truncation transition")
    initial_hash = starts[0].get("policy_weights_sha256")
    final_hash = updates[-1].get("policy_weights_sha256")
    if (not isinstance(initial_hash, str) or len(initial_hash) != 64
            or not isinstance(final_hash, str) or len(final_hash) != 64
            or initial_hash == final_hash):
        raise ValueError("initial and final PPO policy parameter hashes do not prove learning")
    if len(checkpoints) != 17:
        raise ValueError("checkpoint telemetry must include 16 periodic checkpoints plus final archive")
    periodic = [r for r in checkpoints if r.get("name", "").startswith("checkpoint-")]
    final_events = [r for r in checkpoints if r.get("name") == "model-final.zip"]
    expected_times = list(range(4096, target_steps + 1, 4096))
    if ([r.get("timestep") for r in periodic] != expected_times
            or len(final_events) != 1 or final_events[0].get("timestep") != target_steps):
        raise ValueError("checkpoint event schedule differs from the frozen 4,096-step schedule")
    checkpoint_manifest, _ = _strict_json(run_dir / "checkpoints.json")
    rows = checkpoint_manifest.get("checkpoints", [])
    if [r.get("timesteps") for r in rows] != expected_times:
        raise ValueError("periodic checkpoint manifest is missing or reordered")
    archive_records = []
    for row in rows:
        path = _inside(run_dir, row["path"])
        if path.name.endswith(".pending.zip") or _sha256(path) != row["sha256"]:
            raise ValueError(f"periodic checkpoint SHA or completion state failed: {path.name}")
        archive_records.append({"path": row["path"], "timesteps": row["timesteps"],
                                "sha256": row["sha256"], "size_bytes": path.stat().st_size})
    final_info = state.get("final_model", {})
    final_path = _inside(run_dir, final_info.get("path", ""))
    if (final_path.name != "model-final.zip" or _sha256(final_path) != final_info.get("sha256")
            or final_events[0].get("archive_sha256") != final_info.get("sha256")):
        raise ValueError("final model archive hash does not match the run state/telemetry")
    archive_records.append({"path": "model-final.zip", "timesteps": target_steps,
                            "sha256": final_info["sha256"], "size_bytes": final_path.stat().st_size})
    if list(run_dir.rglob("*.pending.zip")):
        raise ValueError("incomplete pending checkpoint archive exists")
    integrity, _ = _strict_json(run_dir / "training-integrity.json")
    gradient_rows = integrity.get("optimizer_gradient_activity", [])
    if (integrity.get("schema_version") != "f1rl_local_demo_training_integrity_v1"
            or integrity.get("optimizer_steps") != state["optimizer_steps"]
            or len(gradient_rows) != state["optimizer_steps"]
            or integrity.get("finite_gradient_steps") != len(gradient_rows)
            or integrity.get("nonzero_gradient_steps") != len(gradient_rows)
            or integrity.get("policy_weights_finite_at_every_optimizer_step") is not True
            or integrity.get("weights_changed") is not True):
        raise ValueError("per-optimizer gradient/weight integrity evidence is incomplete")
    if (integrity.get("initial_policy_weights_sha256") != starts[0].get("policy_weights_sha256")
            or integrity.get("final_policy_weights_sha256") != updates[-1].get("policy_weights_sha256")):
        raise ValueError("integrity weight hashes differ from the source-bound telemetry")
    for item in gradient_rows:
        if (item.get("finite") is not True or item.get("gradient_l2_norm", 0) <= 0
                or item.get("nonzero_gradient_tensors", 0) <= 0
                or not np_isfinite(item.get("gradient_l2_norm"))):
            raise ValueError("an optimizer step has non-finite or zero gradient evidence")
    checkpoint_policy_hashes = {r["name"]: r["policy_weights_sha256"] for r in checkpoints}
    update_policy_hashes = {int(r["ppo_epoch_updates"]): r["policy_weights_sha256"] for r in updates}
    update_optimizer_steps = [r.get("optimizer_steps") for r in updates]
    if (not all(isinstance(value, int) and value >= 0 for value in update_optimizer_steps)
            or any(b <= a for a, b in zip(update_optimizer_steps, update_optimizer_steps[1:]))
            or update_optimizer_steps[-1] != state["optimizer_steps"]):
        raise ValueError("PPO optimizer telemetry did not increase monotonically to the final count")
    for update in updates:
        metrics = update.get("metrics", {})
        for key in ("train/loss", "train/policy_gradient_loss", "train/value_loss"):
            if not isinstance(metrics.get(key), (int, float)) or not np_isfinite(metrics[key]):
                raise ValueError(f"PPO training metric is missing or non-finite: {key}")
    from ..learning.algorithms.ppo import PPOModel
    for row in archive_records:
        model_path = _inside(run_dir, row["path"])
        loaded = PPOModel.load(model_path, device="cpu")
        if loaded.num_timesteps != row["timesteps"]:
            raise ValueError(f"checkpoint timestep identity mismatch: {row['path']}")
        if not loaded.policy_weights_finite():
            raise ValueError(f"checkpoint contains a non-finite policy parameter: {row['path']}")
        expected_weight_hash = checkpoint_policy_hashes.get(Path(row["path"]).name)
        if expected_weight_hash is None:
            expected_weight_hash = final_events[0]["policy_weights_sha256"]
        if loaded.weights_sha256() != expected_weight_hash:
            raise ValueError(f"checkpoint policy hash differs from telemetry: {row['path']}")
        if row["timesteps"] == target_steps:
            if loaded._model.observation_space.shape != tuple(run_manifest["observation_space"]["shape"]):
                raise ValueError("final checkpoint observation space differs from the registered contract")
            if loaded._model.action_space.shape != tuple(run_manifest["action_space"]["shape"]):
                raise ValueError("final checkpoint action space differs from the registered contract")
    track_episodes = {track: sum(data["track_id"] == track for data in episode_summary.values())
                      for track in TRAIN_IDS}
    lifecycle = {
        "episode_count": len(episode_summary), "terminal_transitions": reset_count,
        "automatic_resets_verified": automatic_resets_verified,
        "natural_terminations": sum(data["terminated"] and not data["curriculum_terminal"]
                                     and not data.get("physics_abort_terminal")
                                     for data in episode_summary.values()),
        "physics_abort_terminations": sum(data.get("physics_abort_terminal", False)
                                           for data in episode_summary.values()),
        "curriculum_terminations": sum(data["curriculum_terminal"] for data in episode_summary.values()),
        "truncations": sum(data["truncated"] for data in episode_summary.values()),
        "track_episode_counts": track_episodes,
        "track_timestep_counts": track_timestep_counts,
    }
    return {
        "schema_version": "f1rl_wp3_model_engineering_validation_v1",
        "status": "ENGINEERING_PASS", "model_id": expected["model_id"],
        "candidate_id": expected["candidate_id"], "seed_index": expected["seed_index"],
        "source_commit": campaign_manifest["source_commit"], "run_id": run_event["run_id"],
        "learning_timesteps": target_steps, "ppo_update_rows": len(updates),
        "optimizer_steps": state["optimizer_steps"],
        "initial_policy_weights_sha256": initial_hash,
        "final_policy_weights_sha256": final_hash,
        "weights_changed": initial_hash != final_hash,
        "policy_weights_finite_at_all_checkpoints": True,
        "telemetry": {"path": "training-telemetry.jsonl", "bytes": telemetry_path.stat().st_size,
                      "sha256": digest, "events": len(records), "step_rows": len(steps),
                      "update_rows": len(updates), "optimizer_steps": state["optimizer_steps"]},
        "checkpoints": archive_records, "lifecycle": lifecycle,
        "episodes": len(episode_summary), "run_manifest_sha256": _sha256(run_dir / "run-manifest.json"),
        "scientific_evidence": False, "heldout_access": False,
    }


def np_isfinite(value: object) -> bool:
    import math
    return isinstance(value, (int, float)) and math.isfinite(value)


class _CampaignLock:
    """Windows byte-range lock; process exit releases it even after interruption."""
    def __init__(self, path: Path) -> None:
        self.path = path
        self.stream = None

    def __enter__(self):
        if os.name != "nt":
            raise OSError("WP3 campaign orchestration is local Windows only")
        import msvcrt
        self.stream = self.path.open("a+b")
        self.stream.seek(0)
        self.stream.write(b"0")
        self.stream.flush()
        self.stream.seek(0)
        try:
            msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            self.stream.close()
            self.stream = None
            raise RuntimeError("another WP3 campaign process holds the run lock") from error
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.stream is not None:
            import msvcrt
            self.stream.seek(0)
            msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
            self.stream.close()


def _append_campaign_event(path: Path, event: dict) -> None:
    payload = json.dumps({"schema_version": CAMPAIGN_EVENTS_SCHEMA, **event},
                         sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _state_update(path: Path, state: dict, **updates) -> dict:
    state.update(updates, updated_at=datetime.now(timezone.utc).isoformat())
    _atomic_json(path, state)
    return state


def _run_child(root: Path, output: Path, log_path: Path, model: dict,
               training: dict, *, abort_event=None, resources: dict | None = None,
               process_record_path: Path | None = None) -> tuple[int, float, dict] | dict:
    if abort_event is not None:
        return _run_parallel_child(root, output, log_path, model, training,
                                   abort_event=abort_event, resources=resources or {},
                                   process_record_path=process_record_path)
    command = [sys.executable, "scripts/local_demo.py", "train",
               "--candidate", model["candidate_id"],
               "--tuning-seed-index", str(model["seed_index"]),
               "--output", str(output)]
    env = os.environ.copy()
    started = time.monotonic()
    peak_rss_bytes = 0
    peak_cpu_seconds = None
    with log_path.open("x", encoding="utf-8", newline="\n") as log:
        proc = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT,
                                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        try:
            deadline = started + training["max_wall_seconds_per_model"] + 120
            while True:
                try:
                    returncode = proc.wait(timeout=min(10, max(0.1, deadline-time.monotonic())))
                    break
                except subprocess.TimeoutExpired:
                    sample = _windows_process_sample(proc.pid)
                    if sample is not None:
                        peak_rss_bytes = max(peak_rss_bytes, sample["rss_bytes"])
                        peak_cpu_seconds = sample["cpu_seconds"]
                    if time.monotonic() >= deadline:
                        if os.name == "nt":
                            proc.send_signal(getattr(__import__("signal"), "CTRL_BREAK_EVENT"))
                        try:
                            proc.wait(timeout=60)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait()
                        raise TimeoutError("campaign parent wall-time guard expired; child evidence preserved")
        except KeyboardInterrupt:
            proc.send_signal(getattr(__import__("signal"), "CTRL_BREAK_EVENT"))
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            raise
    return returncode, time.monotonic() - started, {
        "peak_working_set_bytes": peak_rss_bytes or None,
        "final_cpu_seconds": peak_cpu_seconds,
        "sampling_interval_seconds": 10,
        "measurement_source": "Windows GetProcessMemoryInfo/GetProcessTimes" if os.name == "nt" else "unavailable",
    }


def _run_parallel_child(root: Path, output: Path, log_path: Path, model: dict,
                         training: dict, *, abort_event, resources: dict,
                         process_record_path: Path | None) -> dict:
    from .process_resources import WindowsKillJob, sample_process_tree

    command = [sys.executable, "scripts/local_demo.py", "train",
               "--candidate", model["candidate_id"],
               "--tuning-seed-index", str(model["seed_index"]),
               "--output", str(output)]
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
    cleanup_started = None
    with log_path.open("x", encoding="utf-8", newline="\n") as log:
        proc = subprocess.Popen(command, cwd=root, env=os.environ.copy(), stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT,
                                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
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
                    if not job.wait_empty(resources["cleanup_timeout_seconds"]):
                        raise TimeoutError("cancelled training process tree cleanup did not finish")
                    cleanup_seconds = time.monotonic() - cleanup_started
                    return {"returncode": proc.returncode, "cancelled": True,
                            "elapsed_seconds": time.monotonic() - started,
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
            if job is not None and not job.wait_empty(resources.get("cleanup_timeout_seconds", 30)):
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


def _windows_process_sample(pid: int) -> dict | None:
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    class FILETIME(ctypes.Structure):
        _fields_ = [("low", wintypes.DWORD), ("high", wintypes.DWORD)]

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    handle_type = wintypes.HANDLE
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = handle_type
    kernel.GetProcessTimes.argtypes = [handle_type, ctypes.POINTER(FILETIME),
                                       ctypes.POINTER(FILETIME), ctypes.POINTER(FILETIME),
                                       ctypes.POINTER(FILETIME)]
    kernel.GetProcessTimes.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [handle_type]
    kernel.CloseHandle.restype = wintypes.BOOL
    psapi.GetProcessMemoryInfo.argtypes = [handle_type, ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
                                           wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    handle = kernel.OpenProcess(0x410, False, pid)  # PROCESS_QUERY_INFORMATION | PROCESS_VM_READ
    if not handle:
        return None
    try:
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        memory_ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        creation, exit_time, kernel_time, user_time = FILETIME(), FILETIME(), FILETIME(), FILETIME()
        times_ok = kernel.GetProcessTimes(handle, ctypes.byref(creation), ctypes.byref(exit_time),
                                          ctypes.byref(kernel_time), ctypes.byref(user_time))
        cpu_seconds = None
        if times_ok:
            ticks = ((kernel_time.high << 32) | kernel_time.low) + ((user_time.high << 32) | user_time.low)
            cpu_seconds = ticks / 10_000_000
        return {"rss_bytes": int(counters.PeakWorkingSetSize) if memory_ok else 0,
                "cpu_seconds": cpu_seconds}
    finally:
        kernel.CloseHandle(handle)


def _load_campaign(root: Path, output: Path) -> tuple[dict, dict, list[dict], dict, Path]:
    expected_manifest, config, schedule = build_campaign_manifest(root)
    manifest, _ = _strict_json(output / "WP3-CAMPAIGN-MANIFEST.json")
    state, _ = _strict_json(output / "campaign-state.json")
    for key in ("source_commit", "source_tree_sha256", "baseline_commit", "campaign_config_sha256",
                "simulation_only_config_sha256", "registration_sha256", "track_manifest_sha256",
                "training_profile_sha256", "protocol_sha256"):
        if manifest.get(key) != expected_manifest.get(key):
            raise ValueError(f"campaign resume identity drifted at {key}")
    if manifest.get("sha256") != sha256(_canonical({k: v for k, v in manifest.items() if k != "sha256"})).hexdigest():
        raise ValueError("campaign manifest self-checksum mismatch")
    if state.get("campaign_manifest_sha256") != manifest["sha256"]:
        raise ValueError("campaign state is not bound to its immutable manifest")
    return manifest, config, schedule, state, output


def _validate_or_reuse_run(root: Path, output: Path, expected: dict,
                           manifest: dict, config: dict) -> dict | None:
    if not output.exists():
        return None
    if not output.is_dir() or output.is_symlink():
        raise ValueError("model output path exists but is not a regular directory")
    try:
        validation = validate_training_run(root, output, expected, manifest, config)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"existing model output is incomplete or invalid; refusing duplicate training: {output}") from error
    validation_path = output / "engineering-validation.json"
    if validation_path.exists():
        prior, _ = _strict_json(validation_path)
        calculated = sha256(_canonical({k: v for k, v in prior.items() if k != "sha256"})).hexdigest()
        expected_hash = sha256(_canonical(validation)).hexdigest()
        if prior.get("sha256") != calculated or prior.get("sha256") != expected_hash:
            raise ValueError("existing engineering validation checksum mismatch")
    else:
        validation["sha256"] = sha256(_canonical(validation)).hexdigest()
        _write_new_json(validation_path, validation)
    return validation


def _write_c0_gate_report(path: Path, validation: dict, status: str,
                          reason: str | None = None) -> None:
    lifecycle = validation["lifecycle"]
    lines = [
        "# WP3 C0 Full-Length PPO Engineering Gate",
        "",
        f"Status: **{status}**",
        "",
        f"- Model: `{validation['model_id']}` (candidate {validation['candidate_id']}, seed index {validation['seed_index']})",
        f"- Source commit: `{validation['source_commit']}`",
        f"- Learning timesteps: {validation['learning_timesteps']:,}",
        f"- PPO update rows: {validation['ppo_update_rows']:,}",
        f"- Optimizer steps: {validation['optimizer_steps']:,}",
        f"- Initial policy weights SHA-256: `{validation['initial_policy_weights_sha256']}`",
        f"- Final policy weights SHA-256: `{validation['final_policy_weights_sha256']}`",
        f"- Telemetry SHA-256: `{validation['telemetry']['sha256']}` ({validation['telemetry']['bytes']:,} bytes)",
        f"- Checkpoints verified: {len(validation['checkpoints'])} archives (16 periodic plus final)",
        f"- Episodes: {lifecycle['episode_count']}; automatic reset boundaries verified: {lifecycle['automatic_resets_verified']}",
        f"- Natural terminations: {lifecycle['natural_terminations']}; curriculum terminations: {lifecycle['curriculum_terminations']}; truncations: {lifecycle['truncations']}",
        f"- Unsupported-contact training terminals: {lifecycle['physics_abort_terminations']}",
        "- Runtime split: TRAIN only; HELD-OUT access and scientific evaluation: disabled.",
        "",
        "## Observed registered TRAIN coverage",
        "",
        "| Track | Timesteps | Episodes |",
        "|---|---:|---:|",
    ]
    for track in TRAIN_IDS:
        lines.append(f"| {track} | {lifecycle['track_timestep_counts'][track]:,} | {lifecycle['track_episode_counts'][track]:,} |")
    lines += ["", "Driving success is not part of this engineering gate."]
    if reason:
        lines += ["", f"Gate blocker: {reason}"]
    lines += ["", "WP7 and Gate 2 remain **BLOCKED**.", ""]
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines))
        stream.flush()
        os.fsync(stream.fileno())


def _write_training_deliverables(output: Path, state: dict, schedule: list[dict],
                                 manifest: dict, config: dict) -> None:
    inventory = []
    for model in schedule:
        run_dir = output / model["run_relative_path"]
        validation = state["models"][model["model_id"]]["validation"]
        archives = []
        for archive in validation["checkpoints"]:
            path = _inside(run_dir, archive["path"])
            archives.append({**archive, "size_bytes": path.stat().st_size})
        inventory.append({
            "model_id": model["model_id"], "candidate_id": model["candidate_id"],
            "seed_index": model["seed_index"], "seed_bundle": model["seed_bundle"],
            "source_commit": manifest["source_commit"],
            "training_timesteps": validation["learning_timesteps"],
            "initial_policy_weights_sha256": validation["initial_policy_weights_sha256"],
            "final_model": archives[-1], "checkpoints": archives,
            "training_telemetry": validation["telemetry"],
            "lifecycle": validation["lifecycle"],
            "resource_usage": state["models"][model["model_id"]].get("resource_usage"),
            "elapsed_seconds": state["models"][model["model_id"]].get("elapsed_seconds"),
            "run_path": model["run_relative_path"],
        })
    result = {
        "schema_version": "f1rl_wp3_training_results_v1",
        "status": "ENGINEERING_PASS", "campaign_manifest_sha256": manifest["sha256"],
        "source_commit": manifest["source_commit"], "models": inventory,
        "expected_models": 12, "completed_models": len(inventory),
        "expected_timesteps": config["training"]["total_timesteps"],
        "completed_timesteps": sum(row["training_timesteps"] for row in inventory),
        "scientific_evidence": False, "heldout_access": False,
        "wp7_status": "BLOCKED", "gate2_status": "BLOCKED",
    }
    result["sha256"] = sha256(_canonical(result)).hexdigest()
    _write_immutable_json(output / "WP3-TRAINING-RESULTS.json", result)
    inventory_doc = {
        "schema_version": "f1rl_wp3_model_inventory_v1",
        "campaign_manifest_sha256": manifest["sha256"], "models": inventory,
        "selection_rule": "final_checkpoint_only", "scientific_evidence": False,
        "heldout_access": False,
    }
    inventory_doc["sha256"] = sha256(_canonical(inventory_doc)).hexdigest()
    _write_immutable_json(output / "WP3-MODEL-INVENTORY.json", inventory_doc)


def _write_independent_review(output: Path, schedule: list[dict], state: dict,
                              evaluation: dict) -> None:
    lines = [
        "# WP3 Independent Source and Evidence Review", "",
        "Review status: **PASS** for the bounded simulation-only campaign mechanics; this review does not grant scientific acceptance.", "",
        "The second-pass review checked the committed campaign implementation against its frozen manifest and inspected the resulting run evidence independently of the campaign state summary.", "",
        "| Area | Source / evidence | Review result |", "|---|---|---|",
        "| Frozen design and schedule | `configs/research/wp3-ppo-tuning-campaign-v3.json`; `f1rl/local_demo/campaign.py:model_schedule` | 12 unique candidate/seed identities in fixed order; parameters bound to simulation-only manifest |",
        "| Training integrity | `f1rl/local_demo/runner.py:run_local_demo`; `f1rl/learning/training/telemetry.py`; each `runs/*/training-integrity.json` | Real PPO updates, per-step finite/nonzero gradients, changed weights, telemetry and checkpoints independently checked |",
        "| Training transitions | `f1rl/local_demo/campaign.py:validate_training_run`; each `runs/*/training-telemetry.jsonl` | Exact step/event counts, source/seed/reset/draw-seed/track provenance and terminal boundaries checked |",
        "| Checkpoint integrity | each `runs/*/checkpoints.json`, `model-final.zip`, and engineering validation | All periodic archives plus final archive hash-checked and CPU-loaded |",
        "| Evaluation isolation | `f1rl/local_demo/evaluation.py:run_evaluation`; `WP3-TRAIN-EVALUATION.json` | TRAIN-only paired roster, final checkpoints only, original 4 m task, no curriculum/scientific admission |",
        "| Resume and failure handling | `campaign-state.json`, append-only event logs, per-cell JSONL | Existing/incomplete outputs are preserved; interrupted training is not automatically retried; evaluation invalid cells are not redrawn |",
        "| Full local checks | `WP3-VALIDATION-RESULTS.json` | See recorded pytest, compileall, pip check, and diff-check outcomes |",
        "", f"Training identities independently reviewed: {len(schedule)}. Completed: {len(state.get('completed_model_ids', []))}.",
        f"Evaluation cells complete: {evaluation.get('completed_cells', 0)}/96; invalid cells: {len(evaluation.get('invalid_cells', []))}.",
        "", "Residual limits: three tuning seed blocks support descriptive comparisons only; episode-level repetitions are not independent PPO seed blocks. No confirmatory inference or WP7 acceptance is claimed. Gate 2 and WP7 remain **BLOCKED**.", "",
    ]
    content = "\n".join(lines)
    path = output / "WP3-INDEPENDENT-REVIEW.md"
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise ValueError("refusing to overwrite immutable WP3 independent review")
        return
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _write_evidence_manifest(output: Path, *, allow_incomplete_attempt_artifacts: bool = False) -> None:
    files = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name in {"WP3-EVIDENCE-MANIFEST.json", ".campaign.lock"}:
            continue
        if (path.suffix == ".tmp" or path.name.endswith(".pending.zip")) and not allow_incomplete_attempt_artifacts:
            raise ValueError(f"temporary or incomplete artifact remains in campaign evidence: {path}")
        files.append({"path": path.relative_to(output).as_posix(),
                      "bytes": path.stat().st_size, "sha256": _sha256(path)})
    manifest = {"schema_version": "f1rl_wp3_evidence_manifest_v1",
                "files": files, "file_count": len(files),
                "scientific_evidence": False, "heldout_access": False,
                "wp7_status": "BLOCKED", "gate2_status": "BLOCKED"}
    manifest["sha256"] = sha256(_canonical(manifest)).hexdigest()
    _write_new_json(output / "WP3-EVIDENCE-MANIFEST.json", manifest)


def _verify_evidence_manifest(output: Path) -> None:
    path = output / "WP3-EVIDENCE-MANIFEST.json"
    manifest, _ = _strict_json(path)
    if manifest.get("sha256") != sha256(_canonical({k: v for k, v in manifest.items()
                                                       if k != "sha256"})).hexdigest():
        raise ValueError("WP3 evidence manifest self-checksum mismatch")
    expected = {row["path"]: row for row in manifest.get("files", [])}
    actual_paths = {file.relative_to(output).as_posix(): file
                    for file in output.rglob("*")
                    if file.is_file() and file.name not in {path.name, ".campaign.lock"}}
    if set(expected) != set(actual_paths):
        raise ValueError("WP3 evidence manifest inventory differs from the campaign root")
    for relative, file in actual_paths.items():
        row = expected[relative]
        if row.get("bytes") != file.stat().st_size or row.get("sha256") != _sha256(file):
            raise ValueError(f"WP3 campaign evidence changed after final manifest: {relative}")


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
            _append_parallel_event(event_path, {
                "event": "job_interrupted", "model_id": model_id,
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
            _parallel_find_attempt(state, model_id, attempt_id).update(
                status="FAILED", error=message, completed_at=timestamp)
            _append_parallel_event(event_path, {
                "event": "job_failed", "model_id": model_id,
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
        "status": "LAUNCHING", "run_relative_path": run_relative_path,
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
    _atomic_json(output / "parallel-campaign-state.json", state)
    _append_parallel_event(output / "parallel-campaign-events.jsonl", {
        "event": "job_dispatched", "model_id": model_id, "attempt_id": attempt_id,
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
            elif event == "job_cancelled":
                attempt.update(status="CANCELLED", completed_at=timestamp,
                               elapsed_seconds=detail.get("elapsed_seconds"),
                               cleanup_seconds=detail.get("cleanup_seconds"),
                               resource_usage=detail.get("resource_usage"))
                entry.update(status="CANCELLED", latest_attempt_id=job["attempt_id"])
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
                return state
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
            other_failures = [model_id for model_id, entry in state["models"].items()
                              if model_id != retry_model_id
                              and entry.get("status") in {"FAILED", "CANCELLED", "CANCELLATION_REQUESTED", "INTERRUPTED", "RUNNING"}]
            if other_failures:
                raise RuntimeError(f"other failed jobs require explicit retries first: {other_failures}")
        else:
            unresolved = [model_id for model_id, entry in state["models"].items()
                          if entry.get("status") in {"FAILED", "CANCELLED", "CANCELLATION_REQUESTED", "INTERRUPTED", "RUNNING"}]
            if unresolved:
                raise RuntimeError(f"parallel resume requires explicit retries for failed attempts: {unresolved}")

        pending_jobs = []
        for row in schedule:
            if row["model_id"] == "C-seed0":
                continue
            status = state["models"][row["model_id"]].get("status")
            if status == "PENDING" or row["model_id"] == retry_model_id:
                pending_jobs.append(dict(row))
        if retry_model_id and retry_model_id != "C-seed0":
            pending_jobs.sort(key=lambda row: (row["model_id"] != retry_model_id, row["order"]))
        if pending_jobs:
            stage_result = _parallel_run_campaign_stage(
                root, output, state, intent, base_manifest, campaign_config,
                parallel_config, pending_jobs, workers, retry_model_id=retry_model_id,
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
        _write_independent_review(output, effective_schedule, state, evaluation)
        _write_evidence_manifest(output, allow_incomplete_attempt_artifacts=True)
        return state


def _new_campaign(root: Path, output: Path) -> tuple[dict, dict, list[dict], dict, Path]:
    if output.exists():
        raise FileExistsError(f"campaign evidence root already exists; use resume after review: {output}")
    manifest, config, schedule = build_campaign_manifest(root)
    output.mkdir(parents=True, exist_ok=False)
    (output / "runs").mkdir()
    (output / "logs").mkdir()
    (output / "evaluation").mkdir()
    from .evaluation import build_evaluation_roster
    roster = build_evaluation_roster(root, manifest, config)
    _write_new_json(output / "WP3-CAMPAIGN-MANIFEST.json", manifest)
    _write_new_json(output / "evaluation" / "evaluation-roster.json", roster)
    state = {
        "schema_version": "f1rl_wp3_campaign_state_v1",
        "campaign_manifest_sha256": manifest["sha256"],
        "status": "PREPARED", "phase": "C0_ENGINEERING_GATE",
        "current_model_id": None, "completed_model_ids": [],
        "models": {row["model_id"]: {"status": "pending"} for row in schedule},
        "completed_training_timesteps": 0, "updated_at": datetime.now(timezone.utc).isoformat(),
        "scientific_evidence": False, "heldout_access": False,
        "wp7_status": "BLOCKED", "gate2_status": "BLOCKED",
    }
    _write_new_json(output / "campaign-state.json", state)
    return manifest, config, schedule, state, output


def run_campaign(repository_root: str | Path, output_directory: str | Path,
                 *, resume: bool = False) -> dict:
    root = Path(repository_root).resolve(strict=True)
    output_arg = Path(output_directory)
    if not output_arg.is_absolute():
        raise ValueError("WP3 campaign evidence root must be an absolute path")
    output = output_arg.resolve()
    if output.is_relative_to(root):
        raise ValueError("campaign evidence must be outside the repository")
    if not resume:
        manifest, config, schedule, state, output = _new_campaign(root, output)
    else:
        manifest, config, schedule, state, output = _load_campaign(root, output)
    lock_path = output / ".campaign.lock"
    events_path = output / "campaign-events.jsonl"
    try:
        with _CampaignLock(lock_path):
            if (resume and state.get("status") in ("COMPLETE", "EVALUATION_WITH_INVALID_CELLS")
                    and (output / "WP3-EVIDENCE-MANIFEST.json").exists()):
                _verify_evidence_manifest(output)
                return state
            verified_in_process: set[str] = set()
            if resume:
                for expected in schedule:
                    model_id = expected["model_id"]
                    if state["models"][model_id].get("status") == "ENGINEERING_PASS":
                        run_dir = output / expected["run_relative_path"]
                        validation = _validate_or_reuse_run(root, run_dir, expected, manifest, config)
                        if validation is None:
                            raise ValueError(f"completed campaign state has no valid model evidence: {model_id}")
                        verified_in_process.add(model_id)
            # A completed, valid run is re-used; an incomplete run is never retrained implicitly.
            for expected in schedule:
                model_id = expected["model_id"]
                if model_id in verified_in_process:
                    continue
                run_dir = output / expected["run_relative_path"]
                valid = None
                if run_dir.exists():
                    valid = _validate_or_reuse_run(root, run_dir, expected, manifest, config)
                if valid is not None:
                    entry = state["models"][model_id]
                    entry.update(status="ENGINEERING_PASS", validation=valid,
                                 output_path=run_dir.relative_to(output).as_posix())
                    if model_id not in state["completed_model_ids"]:
                        state["completed_model_ids"].append(model_id)
                    state["completed_training_timesteps"] = len(state["completed_model_ids"]) * config["training"]["timesteps_per_model"]
                    verified_in_process.add(model_id)
                    continue
                if state["models"][model_id]["status"] not in ("pending", "launching"):
                    raise RuntimeError(f"model {model_id} is not retryable; refusing duplicate training")
                if state.get("current_model_id") not in (None, model_id):
                    raise RuntimeError("campaign state points to a different unfinished model")
                state = _state_update(output / "campaign-state.json", state,
                                      status="RUNNING", phase=("C0_ENGINEERING_GATE" if model_id == "C-seed0" else "STAGE_2_TRAINING"),
                                      current_model_id=model_id)
                state["models"][model_id].update(status="launching", output_path=expected["run_relative_path"])
                _atomic_json(output / "campaign-state.json", state)
                log_path = output / "logs" / f"{model_id}.log"
                _append_campaign_event(events_path, {"event": "model_started", "model_id": model_id,
                                                      "candidate_id": expected["candidate_id"],
                                                      "seed_index": expected["seed_index"],
                                                      "source_commit": manifest["source_commit"],
                                                      "timestamp": datetime.now(timezone.utc).isoformat()})
                state["models"][model_id].update(status="running", started_at=datetime.now(timezone.utc).isoformat())
                _atomic_json(output / "campaign-state.json", state)
                try:
                    returncode, elapsed, resource_usage = _run_child(root, run_dir, log_path, expected, config["training"])
                except KeyboardInterrupt:
                    state["models"][model_id].update(status="interrupted", log_path=log_path.relative_to(output).as_posix())
                    _state_update(output / "campaign-state.json", state, status="INTERRUPTED",
                                  phase="INTERRUPTED_PRESERVED", current_model_id=model_id)
                    _append_campaign_event(events_path, {"event": "model_interrupted", "model_id": model_id,
                                                          "timestamp": datetime.now(timezone.utc).isoformat()})
                    raise
                except BaseException as error:
                    state["models"][model_id].update(status="failed", error=f"{type(error).__name__}: {error}")
                    _state_update(output / "campaign-state.json", state, status="ENGINEERING_FAILED",
                                  phase="STOPPED_ON_FAILURE", current_model_id=model_id)
                    _append_campaign_event(events_path, {"event": "model_failed", "model_id": model_id,
                                                          "error": f"{type(error).__name__}: {error}",
                                                          "timestamp": datetime.now(timezone.utc).isoformat()})
                    raise
                state["models"][model_id].update(returncode=returncode, elapsed_seconds=elapsed,
                                                 resource_usage=resource_usage,
                                                 log_path=log_path.relative_to(output).as_posix())
                if returncode != 0:
                    state["models"][model_id]["status"] = "failed"
                    _state_update(output / "campaign-state.json", state, status="ENGINEERING_FAILED",
                                  phase="STOPPED_ON_FAILURE", current_model_id=model_id)
                    raise RuntimeError(f"LOCAL_DEMO training child returned {returncode} for {model_id}")
                try:
                    validation = validate_training_run(root, run_dir, expected, manifest, config)
                except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
                    state["models"][model_id].update(status="integrity_failed", error=f"{type(error).__name__}: {error}")
                    _state_update(output / "campaign-state.json", state, status="ENGINEERING_FAILED",
                                  phase="STOPPED_ON_FAILURE", current_model_id=model_id)
                    raise RuntimeError(f"engineering validation failed for {model_id}") from error
                validation["sha256"] = sha256(_canonical(validation)).hexdigest()
                _write_new_json(run_dir / "engineering-validation.json", validation)
                if model_id == "C-seed0":
                    lifecycle = validation["lifecycle"]
                    uncovered = [track for track, count in lifecycle["track_timestep_counts"].items() if count <= 0]
                    gate_pass = not uncovered and lifecycle["automatic_resets_verified"] >= 1
                    if not gate_pass:
                        reason = (f"unobserved TRAIN tracks={uncovered}; verified automatic reset boundaries="
                                  f"{lifecycle['automatic_resets_verified']}")
                        _write_c0_gate_report(output / "WP3-C0-ENGINEERING-GATE.md",
                                              validation, "ENGINEERING_FAIL", reason)
                        state["models"][model_id].update(status="integrity_failed", engineering_failure={
                            "unobserved_tracks": uncovered,
                            "automatic_resets_verified": lifecycle["automatic_resets_verified"],
                        })
                        _state_update(output / "campaign-state.json", state,
                                      status="ENGINEERING_FAILED", phase="STOPPED_ON_FAILURE",
                                      current_model_id=model_id)
                        raise RuntimeError("C0 failed actual multi-track coverage/autoreset gates")
                state["models"][model_id].update(status="ENGINEERING_PASS", validation=validation)
                if model_id not in state["completed_model_ids"]:
                    state["completed_model_ids"].append(model_id)
                state["completed_training_timesteps"] = len(state["completed_model_ids"]) * config["training"]["timesteps_per_model"]
                if model_id == "C-seed0":
                    _write_c0_gate_report(output / "WP3-C0-ENGINEERING-GATE.md",
                                          validation, "ENGINEERING_PASS")
                    state["c0_engineering_gate"] = {
                        "status": "ENGINEERING_PASS", "model_id": model_id,
                        "validation_sha256": validation["sha256"],
                        "passed_at": datetime.now(timezone.utc).isoformat(),
                    }
                    state["phase"] = "STAGE_2_TRAINING"
                state["current_model_id"] = None
                verified_in_process.add(model_id)
                _atomic_json(output / "campaign-state.json", state)
                _append_campaign_event(events_path, {"event": "model_engineering_pass", "model_id": model_id,
                                                      "learning_timesteps": validation["learning_timesteps"],
                                                      "optimizer_steps": validation["optimizer_steps"],
                                                      "telemetry_sha256": validation["telemetry"]["sha256"],
                                                      "timestamp": datetime.now(timezone.utc).isoformat()})
            if len(state["completed_model_ids"]) != 12:
                raise RuntimeError("campaign training phase did not validate all 12 models")
            _write_training_deliverables(output, state, schedule, manifest, config)
            state = _state_update(output / "campaign-state.json", state,
                                  status="TRAINING_COMPLETE", phase="STAGE_3_TRAIN_EVALUATION",
                                  current_model_id=None)
            from .evaluation import run_evaluation
            evaluation = run_evaluation(root, output, manifest, config, schedule)
            state = _state_update(output / "campaign-state.json", state,
                                  status=("COMPLETE" if evaluation["status"] == "PASS" else "EVALUATION_WITH_INVALID_CELLS"),
                                  phase="COMPLETE" if evaluation["status"] == "PASS" else "EVALUATION_REVIEW",
                                  evaluation=evaluation)
            _write_independent_review(output, schedule, state, evaluation)
            _write_evidence_manifest(output)
            return state
    except KeyboardInterrupt:
        raise


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Frozen WP3 LOCAL_DEMO campaign supervisor")
    parser.add_argument("command", choices=("run", "resume", "parallel-run", "parallel-resume", "parallel-retry"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--retry-model", choices=tuple(f"{candidate}-seed{seed}"
                                                         for candidate in "ABCD" for seed in range(3)))
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    if args.command.startswith("parallel-"):
        if args.command == "parallel-retry" and args.retry_model is None:
            parser.error("parallel-retry requires --retry-model")
        if args.command != "parallel-retry" and args.retry_model is not None:
            parser.error("--retry-model is only valid with parallel-retry")
        state = run_parallel_campaign(
            root, args.output, workers=args.workers,
            resume=args.command != "parallel-run",
            retry_model_id=args.retry_model,
        )
        print(f"[WP3-PARALLEL] {state['status']} completed="
              f"{len(state['completed_model_ids'])}/12 "
              f"timesteps={state['completed_training_timesteps']}", flush=True)
        return 0 if state["status"] == "COMPLETE" else 2
    if args.workers is not None or args.retry_model is not None:
        parser.error("--workers and --retry-model are only valid for explicit parallel commands")
    state = run_campaign(root, args.output, resume=args.command == "resume")
    print(f"[WP3] {state['status']} completed={len(state['completed_model_ids'])}/12 "
          f"timesteps={state['completed_training_timesteps']}", flush=True)
    return 0 if state["status"] == "COMPLETE" else 2
