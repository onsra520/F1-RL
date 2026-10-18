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
from importlib import metadata
import os
import platform
import subprocess
import sys

from ..learning.provenance import _capture_git_provenance
from ..learning.pilot_preflight import TRAIN_IDS
from ..learning.reconstruction import RecoveryCandidate, build_training_scenarios
from ..learning.rewards import RewardComposer, RouteProgressReward, CollisionPenalty, TimePenalty
from ..learning.route_navigation import AuthoredRoute
from ..learning.training.config import TrainingRunConfig
from ..learning.training.env_factory import TrainingScenario, TrainingScenarioSet
from ..learning.train_reset import TrainResetConfig
from ..learning.action import ActionContract
from ..learning.wp7_race import WP7RaceConfig
from ..learning.hard_route_outcome import owner_outcome_contract, _validate_owner_parent_identity
from ..sandbox.ballistic_flight import FlightState
from ..runtime.physics import PhysicsConfig
from ..sandbox.geometry.vector import Vec3
from ..sandbox.independent_suspension_dynamics import IndependentSuspensionDynamicsPolicy, SuspensionDynamicsState
from ..sandbox.multi_surface_support import MultiSurfaceSuspensionPolicy
from ..sandbox.ramp_support import WheelContactPoints
from ..sandbox.coupled_longitudinal_dynamics import LongitudinalContactPolicy
from ..sandbox.spatial_contact_dynamics import SpatialContactPolicy
from ..track_assets import load_track_manifest
from .wp7_track_feasibility import BODY


def digest(document):
    return sha256(json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate protocol field")
        result[key] = value
    return result


PROFILES = ('pv5_v1','spatial_ground_v1','spatial_owner_v2','spatial_rotating_owner_v3')


def _registration_path(root,profile):
    if type(profile) is not str or profile not in PROFILES:
        raise ValueError('unregistered execution profile')
    return root / ('configs/research/wp7-rotating-owner-protocol-v3.json' if profile=='spatial_rotating_owner_v3'
                   else 'configs/research/wp7-owner-execution-protocol-v2.json' if profile=='spatial_owner_v2'
                   else 'configs/research/wp7-spatial-ground-protocol-v1.json' if profile=='spatial_ground_v1'
                   else 'configs/research/wp7-pv5-protocol-v1.json')


def protocol_profile(protocol):
    schemas={'wp7_bound_protocol_v1':'pv5_v1','wp7_bound_protocol_v2':'spatial_ground_v1',
             'wp7_bound_protocol_v3':'spatial_owner_v2','wp7_bound_protocol_v4':'spatial_rotating_owner_v3'}
    if not isinstance(protocol,dict) or type(protocol.get('schema_version')) is not str or protocol['schema_version'] not in schemas:
        raise ValueError('unregistered protocol profile schema')
    return schemas[protocol['schema_version']]


def load_registration(root,*,profile='pv5_v1'):
    path = _registration_path(root,profile)
    owner = profile in ('spatial_owner_v2','spatial_rotating_owner_v3')
    spatial = profile in ('spatial_ground_v1','spatial_owner_v2','spatial_rotating_owner_v3')
    doc = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique)
    keys = {"schema_version", "status", "simulator_version", "training_profile", "seed_roster",
            "cohorts", "evaluation", "engineering_suspension", "required_domains", "calibration",
            "source_policy", "approval"}
    if spatial:
        keys.add('engineering_spatial_contact')
    if owner:
        keys.add('owner_outcome_contract')
    if profile=='spatial_rotating_owner_v3':
        keys.add('ground_motion_version')
    if not isinstance(doc, dict) or set(doc) != keys:
        raise ValueError("protocol registration requires exact fields")
    fixed = {
        "schema_version": "wp7_experiment_registration_v4" if profile=='spatial_rotating_owner_v3' else "wp7_experiment_registration_v3" if owner else "wp7_experiment_registration_v2" if spatial else "wp7_experiment_registration_v1", "status": "draft_not_authorized",
        "simulator_version": "f1rl-0.1.0/wp7-spatial-rotating-owner-v3" if profile=='spatial_rotating_owner_v3' else "f1rl-0.1.0/wp7-spatial-owner-v2" if owner else "f1rl-0.1.0/wp7-spatial-ground-v1" if spatial else "f1rl-0.1.0/wp7-pv5", "training_profile": "configs/recovery/wp7-pilot-200k.yml",
        "cohorts": ["singletrack_each_train", "multitrack_uniform_reset"],
        "required_domains": ["grounded", "partial_support", "takeoff", "flight", "landing", "recovery"],
        "calibration": "provisional_engineering_values_require_measured_reference_and_independent_review",
        "source_policy": "exact_commit_clean_tree_and_actual_file_hashes_at_training_start",
        "approval": "separate_independently_verified_owner_contract_required",
    }
    if owner:
        fixed['owner_outcome_contract']='configs/research/wp7-hard-route-outcome-v2.json'
    if profile=='spatial_rotating_owner_v3':
        fixed['ground_motion_version']='spatial_rotating_compliant_lifecycle_v3'
    if any(doc[key] != value for key, value in fixed.items()):
        raise ValueError("unregistered protocol contract")
    if owner:
        owner_outcome_contract(root/doc['owner_outcome_contract'])
        _validate_owner_parent_identity(root/'configs/recovery/wp7-owner-decisions-preregistered-v1.yml')
    seeds = doc["seed_roster"]
    if not isinstance(seeds, list) or len(seeds) < 3:
        raise ValueError("protocol requires at least three paired seeds")
    for row in seeds:
        if not isinstance(row, dict) or set(row) != {"ppo", "sampling", "perturbation"} or any(
            type(s) is not int or not 0 <= s < 2**32 for s in row.values()
        ):
            raise ValueError("invalid protocol seed identity")
    if any(len({row[key] for row in seeds}) != len(seeds) for key in seeds[0]):
        raise ValueError("protocol seed streams must be independent across replicates")
    evaluation = doc["evaluation"]
    expected = {
        "strata": ["nominal", "randomized_train"], "episodes_per_cell": 60,
        "confidence": .95, "minimum_lower_bound": .90, "interval": "exact_one_sided_clopper_pearson",
        "multiplicity": "per_cell_no_familywise_claim_owner_review_required", "policy_actions": "stochastic",
        "episode_seed_protocol": "sha256(protocol,policy,track,stratum,episode)_uint32_v1",
        "checkpoint_selection": "final_model_primary_all_checkpoints_diagnostic_no_best_selection",
        "invalid_episode": "abort_cell_preserve_invalid_evidence_no_redraw_or_success_credit",
    }
    if evaluation != expected or type(evaluation["episodes_per_cell"]) is not int:
        raise ValueError("protocol evaluation fields changed; version/review required")
    suspension = doc["engineering_suspension"]
    if not isinstance(suspension,dict):
        raise ValueError('protocol suspension requires exact parameter fields')
    suspension_keys = {
        "mass_kg", "pitch_inertia_kg_m2", "roll_inertia_kg_m2", "spring_stiffness_npm",
        "damper_nspm", "angular_damping_nmsprad", "max_abs_tilt_rad", "integration_substeps",
    }
    if spatial:
        suspension_keys.add('force_model')
        if suspension.get('force_model')!=('road_relative_unilateral_v3' if owner else 'road_relative_vertical_v2'):
            raise ValueError('spatial registration requires road-relative force model')
        contact = doc['engineering_spatial_contact']
        if (not isinstance(contact,dict) or set(contact)!={'traction','yaw_inertia_kg_m2','lateral_damping_nspm','max_angular_speed_radps'}
                or not isinstance(contact['traction'],dict) or set(contact['traction'])!={
                    'traction_mu','max_drive_power_w','brake_speed_scale_mps','energy_relative_tolerance',
                    'energy_absolute_tolerance_j','max_refinements'}):
            raise ValueError('spatial registration requires exact contact policy fields')
        SpatialContactPolicy(**{**contact,'traction':LongitudinalContactPolicy(**contact['traction'])})
    if not isinstance(suspension, dict) or set(suspension) != suspension_keys:
        raise ValueError("protocol suspension requires exact parameter fields")
    return doc


def prepare_scenarios(root, registration, *, seed_index: int):
    """Build registered TRAIN scenarios for exactly one paired seed replicate.

    This is preparation only. A correct seed binding does not grant PPO admission.
    """
    seeds = registration["seed_roster"]
    if type(seed_index) is not int or not 0 <= seed_index < len(seeds):
        raise ValueError("explicit registered seed_index required for TRAIN scenarios")
    candidate = RecoveryCandidate.from_yaml(root / "configs/recovery/wp7-c2-candidate.yml")
    baseline = build_training_scenarios(candidate)
    entries = {e.track_id: e for e in load_track_manifest(root / "assets/tracks/manifest.json").tracks}
    registry = tuple(entries[identity] for identity in TRAIN_IDS)
    owner = registration['schema_version'] in ('wp7_experiment_registration_v3','wp7_experiment_registration_v4')
    reset = TrainResetConfig(seeds[seed_index]["perturbation"], registry,
        **({'version':'wp7_train_reset_lifecycle_research_v1'} if owner else {}))
    wheels = WheelContactPoints(Vec3(-.75,-.5,1.5), Vec3(.75,-.5,1.5),
                                Vec3(-.75,-.5,-1.5), Vec3(.75,-.5,-1.5))
    contact = MultiSurfaceSuspensionPolicy(wheels, .12, .12)
    suspension = IndependentSuspensionDynamicsPolicy(contact, **registration["engineering_suspension"],
                                                     angular_rate_mapping="yaw_rx_rz_v1")
    spatial = registration['schema_version'] in ('wp7_experiment_registration_v2','wp7_experiment_registration_v3','wp7_experiment_registration_v4')
    spatial_policy = registration.get('engineering_spatial_contact')
    physics = PhysicsConfig("wp7_lifecycle_v1" if owner else "wp7_suspension_v1", BODY, suspension,
        ground_motion_version=registration['ground_motion_version'] if registration['schema_version']=='wp7_experiment_registration_v4'
            else 'spatial_compliant_lifecycle_v1' if owner else 'spatial_contact_v1' if spatial else 'arc_distance_v1',
        spawn_reference='authored_horizontal_road_v1',spatial_contact_policy=(
            SpatialContactPolicy(**{**spatial_policy,'traction':LongitudinalContactPolicy(**spatial_policy['traction'])})
            if spatial else None))
    rewards = RewardComposer((RouteProgressReward(1.), CollisionPenalty(5.), TimePenalty(.1)))
    scenarios = []
    for row in baseline.scenarios:
        initial = row.factory.initial_state
        contact = row.factory.track.surface_at(initial.vehicle.position,0.)
        if contact is None or contact.normal!=Vec3(0.,1.,0.):
            raise ValueError('registered TRAIN spawn requires horizontal authored road')
        position = initial.vehicle.position+Vec3(0.,row.factory.vehicle_parameters.ride_height,0.)
        initial = replace(initial,vehicle=replace(initial.vehicle,position=position),
                          suspension=SuspensionDynamicsState(position.y),
                          **({'flight':FlightState('grounded')} if owner else {}))
        factory = replace(row.factory, offtrack_config=None, physics_config=physics,
            race_config=WP7RaceConfig(version='wp7_directed_lifecycle_owner_v2',hard_violation='terminate_valid_failure')
                if owner else WP7RaceConfig(), reset_config=reset, reset_track_id=row.track_id,
            reward_composer=rewards, initial_state=initial,
            action_contract=ActionContract('steering-throttle-brake-v2') if spatial else None)
        scenarios.append(TrainingScenario(row.track_id, factory))
    return TrainingScenarioSet(tuple(scenarios))


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
    files = sorted([*root.joinpath("f1rl").rglob("*.py"), *root.joinpath("scripts").glob("*.py"),
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


def _cpu_model():
    model = platform.processor().strip() or os.environ.get('PROCESSOR_IDENTIFIER', '').strip()
    if model:
        return model
    if platform.system() == 'Linux':
        try:
            for line in Path('/proc/cpuinfo').read_text(encoding='utf-8', errors='replace').splitlines():
                key, separator, value = line.partition(':')
                if separator and key.strip() == 'model name' and value.strip():
                    return value.strip()
        except OSError:
            pass
    return 'unavailable'


def _runtime():
    packages = {name: metadata.version(name) for name in
                ("numpy", "gymnasium", "stable-baselines3", "torch", "PyYAML")}
    probe = Path(__file__).resolve().parents[2] / "scripts/wp7_capture_gpu.py"
    result = subprocess.run([sys.executable, "-B", str(probe)], capture_output=True, text=True, check=True, timeout=30)
    gpu = json.loads(result.stdout)
    return {"python_version": platform.python_version(), "implementation": platform.python_implementation(),
            "os": platform.system(), "os_release": platform.release(), "packages": packages,
            "cpu": {"architecture": platform.machine(), "model": _cpu_model(),
                    "logical_processors": os.cpu_count()},
            **gpu}


def _artifacts(artifacts):
    if not isinstance(artifacts, dict) or any(not isinstance(k, str) or not k.strip() for k in artifacts):
        raise ValueError("manifest artifact identities required")
    result = {}
    for identity, source in artifacts.items():
        path = Path(source)
        if path.suffix.lower() != ".zip" or path.is_symlink() or not path.is_file():
            raise ValueError("manifest requires explicit existing model/checkpoint ZIP files")
        result[identity] = {"sha256": sha256(path.read_bytes()).hexdigest(), "size_bytes": path.stat().st_size}
    return result


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
        "evaluator": protocol["evaluator"], "artifacts": _artifacts({} if artifacts is None else artifacts),
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
