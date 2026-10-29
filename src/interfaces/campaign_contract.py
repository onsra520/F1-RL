"""Frozen WP3 campaign configuration, schedule and manifest contracts."""
from __future__ import annotations

import json
import shutil
import subprocess
from hashlib import sha256
from pathlib import Path

from src.training.config import TrainingRunConfig

from ..rl.pilot_preflight import TRAIN_IDS
from ..validation.simulation_only_protocol import load_simulation_only_manifest
from ..validation.wp7_protocol import build_protocol
from ._artifacts import _canonical, _sha256, _strict_json
from .admission import _source_identity

BASELINE_COMMIT = "0ff42345426f2b3ba5c171fff6cee425b3457b16"
CAMPAIGN_CONFIG = Path("configs/research/wp3-ppo-tuning-campaign-v3.json")
SIMULATION_CONFIG = Path("configs/research/simulation-only-ppo-v1.json")
REGISTRATION_CONFIG = Path("configs/research/wp7-rotating-owner-protocol-v3.json")
TRACK_MANIFEST = Path("assets/tracks/manifest.json")
TRAINING_PROFILE = Path("configs/recovery/wp7-pilot-200k.yml")
CAMPAIGN_EVENTS_SCHEMA = "f1rl_wp3_campaign_event_v1"


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
