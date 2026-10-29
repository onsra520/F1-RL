"""Executable registration preparation, never training authorization.

Historical registrations retain their original domains. The explicit owner-v2
profile binds bounded compliant lifecycle/race/reset; broader required domains
and independent scientific admission remain incomplete.
"""
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
from datetime import datetime, timezone

from ..rl.provenance import _capture_git_provenance
from ..rl.pilot_preflight import TRAIN_IDS
from ..rl.route_navigation import AuthoredRoute
from src.training.config import TrainingRunConfig
from src.training.env_factory import TrainingScenarioSet
from ..rl.action import ActionContract
from . import wp7_runtime
from .wp7_runtime import hash_model_artifacts
from .wp7_registration import (
    PROFILES,
    _registration_path,
    _unique,
    load_registration,
    protocol_profile,
)
from .wp7_scenarios import prepare_scenarios

# Kept as a module seam for existing protocol/restore callers and tests.
platform = wp7_runtime.platform
_cpu_model = wp7_runtime.cpu_model_name
_runtime = wp7_runtime.capture_runtime_fingerprint


def digest(document):
    return sha256(json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def build_protocol(repository_root,*,profile='pv5_v1'):
    root = Path(repository_root).resolve(strict=True)
    registration = load_registration(root,profile=profile)
    scenarios = prepare_scenarios(root, registration, seed_index=0)
    config = TrainingRunConfig.from_yaml(root / registration["training_profile"])
    reference = scenarios.scenarios[0].factory
    policies = []
    for index, seed in enumerate(registration["seed_roster"]):
        for cohort, tracks in [("multitrack", TRAIN_IDS), *((f"singletrack-{t}", (t,)) for t in TRAIN_IDS)]:
            policies.append({"policy_id": f"{cohort}-seed{index}", "training_ids": list(tracks),
                             "seeds": seed, "ppo": asdict(replace(config.ppo, seed=seed["ppo"]))})
    tracks = []
    for row in scenarios.scenarios:
        factory = row.factory
        entry = next(e for e in factory.reset_config.training_registry if e.track_id == row.track_id)
        tracks.append({"track_id": row.track_id, "world_sha256": entry.world_sha256,
            "content_sha256": entry.content_sha256,
            "initial_state_sha256": digest(json.loads(json.dumps(asdict(factory.initial_state),default=sorted))),
            "race_sha256": factory.race_config.provenance(factory.track, AuthoredRoute(factory.navigation))["sha256"]})
    files = sorted([*root.joinpath("src").rglob("*.py"), *root.joinpath("scripts").glob("*.py"),
        *root.joinpath("tests").rglob("*.py"),
        root / "pyproject.toml", _registration_path(root,profile),
        root / "configs/experiments/ppo-generalization-v1.yml", root / registration["training_profile"],
        root / "configs/recovery/wp7-c2-candidate.yml", root / "configs/recovery/wp7-owner-decisions-preregistered-v1.yml",
        root / "assets/tracks/manifest.json",
        *(root / f"assets/tracks/training/{t}.json" for t in TRAIN_IDS)])
    if profile in ('spatial_owner_v2','spatial_rotating_owner_v3'):
        files.append(root/registration['owner_outcome_contract'])
    source = {p.relative_to(root).as_posix(): sha256(p.read_bytes()).hexdigest() for p in files}
    document = {
        "schema_version": "wp7_bound_protocol_v4" if profile=='spatial_rotating_owner_v3' else "wp7_bound_protocol_v3" if profile=='spatial_owner_v2' else "wp7_bound_protocol_v2" if profile=='spatial_ground_v1' else "wp7_bound_protocol_v1", "registration": registration,
        "source": asdict(_capture_git_provenance(root)), "source_files": source,
        "physics": reference.physics_config.document(), "physics_sha256": reference.physics_config.sha256,
        "tracks": tracks,
        "resets": [{"seeds": row, "sha256": replace(reference.reset_config, seed=row["perturbation"]).sha256}
                   for row in registration["seed_roster"]],
        "vehicle": asdict(reference.vehicle_parameters), "simulation": asdict(reference.simulation_config),
        "observation": {"features": list(reference.observation_encoder.feature_names),
                        "configuration": asdict(reference.observation_encoder.config)},
        "action": {"version": (reference.action_contract or ActionContract()).version,
                   "low":list((reference.action_contract or ActionContract()).low),
                   "high":list((reference.action_contract or ActionContract()).high)},
        "rewards": [{"name": r.name, "configuration": asdict(r)} for r in reference.reward_composer.terms],
        "episode": asdict(reference.episode_config), "policies": policies,
        "vector_environment": asdict(config.vector_env), "requested_device": config.ppo.device,
        "training_steps": config.total_timesteps, "checkpoint_every_steps": config.checkpoint_every_steps,
        "required_cells": len(policies)*len(TRAIN_IDS)*2,
        "minimum_evaluation_episodes": len(policies)*len(TRAIN_IDS)*2*60,
        "evaluator": "f1rl.learning.evaluation.evaluator:evaluate_episode/stochastic",
        "split_identity": "four_train_only_no_heldout_rollout",
        "registration_blockers": (["broader_lifecycle_domains_and_profile_robustness_unverified",
                                   "cohort_and_multiplicity_review_required", "engineering_parameters_uncalibrated"]
                                  if profile in ('spatial_owner_v2','spatial_rotating_owner_v3') else
                                  ["flight_race_reset_integration_missing", "owner_failure_policy_missing",
                                   "cohort_and_multiplicity_review_required", "engineering_parameters_uncalibrated"]),
        "training_authorized": False,
    }
    document = json.loads(json.dumps(document, allow_nan=False))
    return {**document, "sha256": digest(document)}


def validate_protocol(document, repository_root):
    if not isinstance(document, dict) or digest(document) != digest(build_protocol(repository_root,profile=protocol_profile(document))):
        raise ValueError("protocol is missing, inconsistent or stale against actual source/configuration")


def evaluation_seed(protocol, policy_id, track_id, stratum, episode):
    if (policy_id not in {p["policy_id"] for p in protocol["policies"]} or track_id not in TRAIN_IDS
            or stratum not in ("nominal", "randomized_train") or type(episode) is not int or episode < 0):
        raise ValueError("unregistered evaluation cell")
    return int(digest([protocol["sha256"], policy_id, track_id, stratum, episode])[:8], 16)


def _scenarios_for_policy(root, protocol, policy_id):
    policy = next((p for p in protocol["policies"] if p["policy_id"] == policy_id), None)
    if policy is None:
        raise ValueError("unregistered training policy identity")
    seed_index = next(
        (i for i, bundle in enumerate(protocol["registration"]["seed_roster"])
         if bundle == policy["seeds"]),
        None,
    )
    if seed_index is None:
        raise ValueError("policy lacks an exact registered paired seed bundle")
    all_scenarios = prepare_scenarios(root, protocol["registration"], seed_index=seed_index)
    scenarios = tuple(
        scenario for scenario in all_scenarios.scenarios
        if scenario.track_id in policy["training_ids"]
    )
    if tuple(s.track_id for s in scenarios) != tuple(policy["training_ids"]):
        raise ValueError("policy track roster is inconsistent with TRAIN registry")
    return TrainingScenarioSet(scenarios), seed_index, policy, all_scenarios


def prepare_training_scenarios(repository_root, protocol, policy_id):
    """Reconstruct a policy-specific TRAIN scenario set without training.

    Fail closed on policy/seed mismatch. This API does NOT grant admission.
    """
    validate_protocol(protocol, repository_root)
    scenarios, index, policy, _ = _scenarios_for_policy(
        Path(repository_root).resolve(strict=True), protocol, policy_id
    )
    return scenarios, {
        "policy_id": policy_id,
        "paired_seed_index": index,
        "seed_bundle": dict(policy["seeds"]),
        "track_ids": list(policy["training_ids"]),
        "physics_sha256": scenarios.scenarios[0].factory.physics_config.sha256,
        "reset_sha256": scenarios.scenarios[0].factory.reset_config.sha256,
        "training_authorized": False,
    }


def policy_stochastic_seed(protocol, policy_id, track_id, stratum, episode):
    """Independent policy-action seed; distinct domain from Gym reset stream."""
    evaluation_seed(protocol, policy_id, track_id, stratum, episode)
    return int(digest([
        protocol["sha256"], "policy_action_rng_v1", policy_id, track_id, stratum, episode,
    ])[:8], 16)


def prepare_evaluation_factory(repository_root, protocol, policy_id, track_id, stratum, episode):
    """Construct a registered cell without resetting, evaluating or admitting it."""
    validate_protocol(protocol, repository_root)
    seed = evaluation_seed(protocol, policy_id, track_id, stratum, episode)
    _, seed_index, policy, scenarios = _scenarios_for_policy(
        Path(repository_root).resolve(strict=True), protocol, policy_id
    )
    # Cross-track evaluation is required even for single-track-trained policies.
    factory = next(row.factory for row in scenarios.scenarios if row.track_id == track_id)
    if stratum == "nominal":
        factory = replace(factory, reset_config=None, reset_track_id=None)
    else:
        factory = replace(factory, reset_config=replace(factory.reset_config, seed=seed))
    return factory, {"policy_id": policy_id, "track_id": track_id, "stratum": stratum,
                     "paired_seed_index": seed_index, "training_seed_bundle": dict(policy["seeds"]),
                     "episode": episode, "environment_seed": seed,
                     "policy_rng_seed": policy_stochastic_seed(protocol, policy_id, track_id, stratum, episode),
                     "deterministic_prediction": False, "training_authorized": False}


def capture_runtime_manifest(repository_root, protocol, *, artifacts=None):
    """Capture now, before any future run; this function never starts PPO.

    Admission must require clean source separately. Inspection captures may
    deliberately report dirty source, and can never serve as past run evidence.
    """
    validate_protocol(protocol, repository_root)
    document = {"schema_version": "wp7_runtime_manifest_v1",
        "scope": "current_capture_not_historical_provenance",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": protocol["sha256"], "source": protocol["source"],
        "source_files": protocol["source_files"], "runtime": _runtime(),
        "requested_device": protocol["requested_device"], "policies": protocol["policies"],
        "physics_sha256": protocol["physics_sha256"], "tracks": protocol["tracks"],
        "resets": protocol["resets"], "split_identity": protocol["split_identity"],
        "evaluator": protocol["evaluator"],
        "artifacts": hash_model_artifacts({} if artifacts is None else artifacts),
        "training_executed": False}
    return {**document, "sha256": digest(document)}


def validate_runtime_manifest(manifest, protocol, repository_root, *, artifacts=None):
    """Validate a current capture against actual files/runtime, not historical claims."""
    validate_protocol(protocol, repository_root)
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a mapping")
    expected = capture_runtime_manifest(repository_root, protocol, artifacts=artifacts)
    actual = {k: v for k, v in manifest.items() if k not in ("captured_at_utc", "sha256")}
    reference = {k: v for k, v in expected.items() if k not in ("captured_at_utc", "sha256")}
    stamp = manifest.get("captured_at_utc")
    try:
        captured = datetime.fromisoformat(stamp)
        if captured.tzinfo is None or captured.utcoffset().total_seconds() != 0:
            raise ValueError("UTC capture time required")
    except (ValueError, TypeError) as exc:
        raise ValueError("manifest capture time invalid") from exc
    if digest(actual) != digest(reference) or manifest.get("sha256") != digest({k:v for k,v in manifest.items() if k != "sha256"}):
        raise ValueError("manifest fields, runtime or artifact bytes inconsistent")
