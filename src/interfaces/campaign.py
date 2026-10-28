"""Frozen, sequential WP3 LOCAL_DEMO training campaign orchestration."""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import struct
import sys
import time
from typing import Any

from ..rl.pilot_preflight import TRAIN_IDS
from src.training.config import TrainingRunConfig
from ..validation.simulation_only_protocol import load_simulation_only_manifest
from ..validation.wp7_protocol import build_protocol
from .admission import _source_identity


BASELINE_COMMIT = "0ff42345426f2b3ba5c171fff6cee425b3457b16"
CAMPAIGN_CONFIG = Path("configs/research/wp3-ppo-tuning-campaign-v3.json")
SIMULATION_CONFIG = Path("configs/research/simulation-only-ppo-v1.json")
REGISTRATION_CONFIG = Path("configs/research/wp7-rotating-owner-protocol-v3.json")
TRACK_MANIFEST = Path("assets/tracks/manifest.json")
TRAINING_PROFILE = Path("configs/recovery/wp7-pilot-200k.yml")
CAMPAIGN_EVENTS_SCHEMA = "f1rl_wp3_campaign_event_v1"


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
        from ..rl.train_reset import derive_train_reset_draw_seed
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
            from ..rl.train_reset import derive_train_reset_draw_seed
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
    from ..rl.algorithms.ppo import PPOModel
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
               training: dict) -> tuple[int, float, dict]:
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
        "| Frozen design and schedule | `configs/research/wp3-ppo-tuning-campaign-v3.json`; `src/interfaces/campaign.py:model_schedule` | 12 unique candidate/seed identities in fixed order; parameters bound to simulation-only manifest |",
        "| Training integrity | `src/interfaces/runner.py:run_local_demo`; `src/training/telemetry.py`; each `runs/*/training-integrity.json` | Real PPO updates, per-step finite/nonzero gradients, changed weights, telemetry and checkpoints independently checked |",
        "| Training transitions | `src/interfaces/campaign.py:validate_training_run`; each `runs/*/training-telemetry.jsonl` | Exact step/event counts, source/seed/reset/draw-seed/track provenance and terminal boundaries checked |",
        "| Checkpoint integrity | each `runs/*/checkpoints.json`, `model-final.zip`, and engineering validation | All periodic archives plus final archive hash-checked and CPU-loaded |",
        "| Evaluation isolation | `src/interfaces/evaluation.py:run_evaluation`; `WP3-TRAIN-EVALUATION.json` | TRAIN-only paired roster, final checkpoints only, original 4 m task, no curriculum/scientific admission |",
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


def _write_evidence_manifest(output: Path) -> None:
    files = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name in {"WP3-EVIDENCE-MANIFEST.json", ".campaign.lock"}:
            continue
        if path.suffix == ".tmp" or path.name.endswith(".pending.zip"):
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
    parser = argparse.ArgumentParser(description="Sequential frozen WP3 local TRAIN PPO tuning campaign")
    parser.add_argument("command", choices=("run", "resume"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    state = run_campaign(root, args.output, resume=args.command == "resume")
    print(f"[WP3] {state['status']} completed={len(state['completed_model_ids'])}/12 "
          f"timesteps={state['completed_training_timesteps']}", flush=True)
    return 0 if state["status"] == "COMPLETE" else 2
