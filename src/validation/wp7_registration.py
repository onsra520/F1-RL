"""WP7 registration loading and exact schema/profile validation."""

import json

from ..simulation.coupled_longitudinal_dynamics import LongitudinalContactPolicy
from ..simulation.spatial_contact_dynamics import SpatialContactPolicy
from ..rl.hard_route_outcome import (
    owner_outcome_contract,
    _validate_owner_parent_identity,
)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate protocol field")
        result[key] = value
    return result


PROFILES = (
    "pv5_v1",
    "spatial_ground_v1",
    "spatial_owner_v2",
    "spatial_rotating_owner_v3",
)


def _registration_path(root, profile):
    if type(profile) is not str or profile not in PROFILES:
        raise ValueError("unregistered execution profile")
    return root / (
        "configs/research/wp7-rotating-owner-protocol-v3.json"
        if profile == "spatial_rotating_owner_v3"
        else "configs/research/wp7-owner-execution-protocol-v2.json"
        if profile == "spatial_owner_v2"
        else "configs/research/wp7-spatial-ground-protocol-v1.json"
        if profile == "spatial_ground_v1"
        else "configs/research/wp7-pv5-protocol-v1.json"
    )


def protocol_profile(protocol):
    schemas = {
        "wp7_bound_protocol_v1": "pv5_v1",
        "wp7_bound_protocol_v2": "spatial_ground_v1",
        "wp7_bound_protocol_v3": "spatial_owner_v2",
        "wp7_bound_protocol_v4": "spatial_rotating_owner_v3",
    }
    if (
        not isinstance(protocol, dict)
        or type(protocol.get("schema_version")) is not str
        or protocol["schema_version"] not in schemas
    ):
        raise ValueError("unregistered protocol profile schema")
    return schemas[protocol["schema_version"]]


def load_registration(root, *, profile="pv5_v1"):
    path = _registration_path(root, profile)
    owner = profile in ("spatial_owner_v2", "spatial_rotating_owner_v3")
    spatial = profile in (
        "spatial_ground_v1",
        "spatial_owner_v2",
        "spatial_rotating_owner_v3",
    )
    doc = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique)
    keys = {
        "schema_version",
        "status",
        "simulator_version",
        "training_profile",
        "seed_roster",
        "cohorts",
        "evaluation",
        "engineering_suspension",
        "required_domains",
        "calibration",
        "source_policy",
        "approval",
    }
    if spatial:
        keys.add("engineering_spatial_contact")
    if owner:
        keys.add("owner_outcome_contract")
    if profile == "spatial_rotating_owner_v3":
        keys.add("ground_motion_version")
    if not isinstance(doc, dict) or set(doc) != keys:
        raise ValueError("protocol registration requires exact fields")
    fixed = {
        "schema_version": "wp7_experiment_registration_v4"
        if profile == "spatial_rotating_owner_v3"
        else "wp7_experiment_registration_v3"
        if owner
        else "wp7_experiment_registration_v2"
        if spatial
        else "wp7_experiment_registration_v1",
        "status": "draft_not_authorized",
        "simulator_version": "f1rl-0.1.0/wp7-spatial-rotating-owner-v3"
        if profile == "spatial_rotating_owner_v3"
        else "f1rl-0.1.0/wp7-spatial-owner-v2"
        if owner
        else "f1rl-0.1.0/wp7-spatial-ground-v1"
        if spatial
        else "f1rl-0.1.0/wp7-pv5",
        "training_profile": "configs/recovery/wp7-pilot-200k.yml",
        "cohorts": ["singletrack_each_train", "multitrack_uniform_reset"],
        "required_domains": [
            "grounded",
            "partial_support",
            "takeoff",
            "flight",
            "landing",
            "recovery",
        ],
        "calibration": "provisional_engineering_values_require_measured_reference_and_independent_review",
        "source_policy": "exact_commit_clean_tree_and_actual_file_hashes_at_training_start",
        "approval": "separate_independently_verified_owner_contract_required",
    }
    if owner:
        fixed["owner_outcome_contract"] = (
            "configs/research/wp7-hard-route-outcome-v2.json"
        )
    if profile == "spatial_rotating_owner_v3":
        fixed["ground_motion_version"] = "spatial_rotating_compliant_lifecycle_v3"
    if any(doc[key] != value for key, value in fixed.items()):
        raise ValueError("unregistered protocol contract")
    if owner:
        owner_outcome_contract(root / doc["owner_outcome_contract"])
        _validate_owner_parent_identity(
            root / "configs/recovery/wp7-owner-decisions-preregistered-v1.yml"
        )
    seeds = doc["seed_roster"]
    if not isinstance(seeds, list) or len(seeds) < 3:
        raise ValueError("protocol requires at least three paired seeds")
    for row in seeds:
        if (
            not isinstance(row, dict)
            or set(row) != {"ppo", "sampling", "perturbation"}
            or any(type(s) is not int or not 0 <= s < 2**32 for s in row.values())
        ):
            raise ValueError("invalid protocol seed identity")
    if any(len({row[key] for row in seeds}) != len(seeds) for key in seeds[0]):
        raise ValueError("protocol seed streams must be independent across replicates")
    evaluation = doc["evaluation"]
    expected = {
        "strata": ["nominal", "randomized_train"],
        "episodes_per_cell": 60,
        "confidence": 0.95,
        "minimum_lower_bound": 0.90,
        "interval": "exact_one_sided_clopper_pearson",
        "multiplicity": "per_cell_no_familywise_claim_owner_review_required",
        "policy_actions": "stochastic",
        "episode_seed_protocol": "sha256(protocol,policy,track,stratum,episode)_uint32_v1",
        "checkpoint_selection": "final_model_primary_all_checkpoints_diagnostic_no_best_selection",
        "invalid_episode": "abort_cell_preserve_invalid_evidence_no_redraw_or_success_credit",
    }
    if evaluation != expected or type(evaluation["episodes_per_cell"]) is not int:
        raise ValueError("protocol evaluation fields changed; version/review required")
    suspension = doc["engineering_suspension"]
    if not isinstance(suspension, dict):
        raise ValueError("protocol suspension requires exact parameter fields")
    suspension_keys = {
        "mass_kg",
        "pitch_inertia_kg_m2",
        "roll_inertia_kg_m2",
        "spring_stiffness_npm",
        "damper_nspm",
        "angular_damping_nmsprad",
        "max_abs_tilt_rad",
        "integration_substeps",
    }
    if spatial:
        suspension_keys.add("force_model")
        if suspension.get("force_model") != (
            "road_relative_unilateral_v3" if owner else "road_relative_vertical_v2"
        ):
            raise ValueError("spatial registration requires road-relative force model")
        contact = doc["engineering_spatial_contact"]
        if (
            not isinstance(contact, dict)
            or set(contact)
            != {
                "traction",
                "yaw_inertia_kg_m2",
                "lateral_damping_nspm",
                "max_angular_speed_radps",
            }
            or not isinstance(contact["traction"], dict)
            or set(contact["traction"])
            != {
                "traction_mu",
                "max_drive_power_w",
                "brake_speed_scale_mps",
                "energy_relative_tolerance",
                "energy_absolute_tolerance_j",
                "max_refinements",
            }
        ):
            raise ValueError(
                "spatial registration requires exact contact policy fields"
            )
        SpatialContactPolicy(
            **{**contact, "traction": LongitudinalContactPolicy(**contact["traction"])}
        )
    if not isinstance(suspension, dict) or set(suspension) != suspension_keys:
        raise ValueError("protocol suspension requires exact parameter fields")
    return doc
