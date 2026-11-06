"""Fail-closed preregistration reader for the WP1–WP7 owner design decisions.

Owner-approved selections are immutable within this contract version.
Engineering reference pose and wheel/suspension values remain provisional.
No model is trained, HELD-OUT file loaded, or freeze authorized here.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite
from pathlib import Path
from typing import Any

import yaml


OWNER_CODES = ("1B", "2B", "3B", "4B", "5A", "6B", "7B", "8B")
EVIDENCE_GATES = (
    "geometry_feasibility", "multi_patch_road_support",
    "airborne_landing_validation", "continuous_triggers",
    "route_and_termination_review", "training_profile_and_provenance",
    "train_only_perturbation_validation", "independent_ppo_robustness",
    "private_backup_restore", "independent_scientific_review",
)


class _UniqueMappingLoader(yaml.SafeLoader):
    """YAML duplicate keys may not silently override approved thresholds."""


def _construct_unique_mapping(loader: yaml.SafeLoader, node: yaml.MappingNode) -> dict:
    mapping: dict = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        try:
            repeated = key in mapping
        except TypeError as exc:
            raise ValueError("mapping keys must be hashable") from exc
        if repeated:
            raise ValueError(f"duplicate preregistration key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=True)
    return mapping


_UniqueMappingLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def _record(value: object, keys: set[str], where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{where} requires exact keys {sorted(keys)}")
    return value


def _exact(value: object, expected: object, where: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise ValueError(f"{where} must remain {expected!r}")


def _numeric(value: object, where: str, *, positive: bool = False) -> float:
    if type(value) not in (float, int) or not isfinite(value):
        raise ValueError(f"{where} must be a finite real number")
    number = float(value)
    if positive and number <= 0:
        raise ValueError(f"{where} must be strictly positive")
    return number


def _pair(value: object, where: str, expected: tuple[float, float]) -> None:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{where} must contain exactly two endpoints")
    points = [_numeric(x, where) for x in value]
    if not all(isclose(a, b, rel_tol=0, abs_tol=1e-12) for a, b in zip(points, expected, strict=True)):
        raise ValueError(f"{where} owner-approved limits changed")


def _xyz(value: object, where: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{where} must contain three coordinates")
    return tuple(_numeric(x, where) for x in value)  # type: ignore[return-value]


def _optional_float(value: object, where: str) -> None:
    if value is not None:
        _numeric(value, where, positive=True)


@dataclass(frozen=True, slots=True)
class PreregistrationReview:
    codes: tuple[str, ...]
    status: str
    blockers: tuple[str, ...]
    training_permitted: bool = False
    heldout_permitted: bool = False
    scientific_freeze_permitted: bool = False
    wp8_permitted: bool = False


def review_preregistered_contract(path: Path | str) -> PreregistrationReview:
    """Check choices against *owner-approved* values and preregistration gate.

    Intentional hard stop: this module never authorizes training/freeze.
    A separate reviewed future scientific runner must verify signed evidence,
    immutable hashes, backup restoration, split, and model policy identities.
    """
    path = Path(path)
    document = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueMappingLoader)
    root = _record(document, {
        "schema_version", "status", "wp0", "decision_codes",
        "simulation", "driving", "reliability", "backup",
        "evidence_gates", "heldout", "scientific_freeze_permitted",
        "wp8_permitted",
    }, "root")
    _exact(root["schema_version"], 1, "schema_version")
    _exact(root["status"], "design_locked_implementation_pending", "status")
    _exact(root["wp0"], "not_applicable", "wp0")
    _exact(root["decision_codes"], list(OWNER_CODES), "decision_codes")

    simulation = _record(root["simulation"], {
        "mode", "collision_semantics", "body", "suspension", "airborne"
    }, "simulation")
    _exact(simulation["mode"], "full_body_3d", "simulation.mode")
    _exact(simulation["collision_semantics"],
           "swept_obb_conservative_linear_slerp_v1", "simulation.collision_semantics")
    body = _record(simulation["body"], {
        "length_m", "width_m", "height_m", "local_center_xyz_m",
        "wheelbase_m", "ride_height_m", "wheel_points_xyz_m",
    }, "simulation.body")
    for key, value in (("length_m", 4.0), ("width_m", 1.8), ("height_m", 1.2)):
        number = _numeric(body[key], f"simulation.body.{key}", positive=True)
        if number != value:
            raise ValueError(f"owner-approved body.{key} changed")
    wheelbase = _numeric(body["wheelbase_m"], "wheelbase_m", positive=True)
    ride_height = _numeric(body["ride_height_m"], "ride_height_m", positive=True)
    if wheelbase >= body["length_m"]:
        raise ValueError("wheelbase cannot exceed vehicle length")
    center = _xyz(body["local_center_xyz_m"], "body.local_center_xyz_m")
    half = (float(body["width_m"])/2, float(body["height_m"])/2, float(body["length_m"])/2)
    if not (-half[0] < center[0] < half[0] and -half[2] < center[2] < half[2]):
        raise ValueError("body local center reference must be within horizontal chassis bounds")
    if ride_height + center[1] - half[1] <= 0:
        raise ValueError("chassis reference pose would penetrate flat ground")

    points = _record(body["wheel_points_xyz_m"], {
        "front_left", "front_right", "rear_left", "rear_right"
    }, "wheel_points_xyz_m")
    wheel_points = {name: _xyz(value, f"wheel_points_xyz_m.{name}") for name, value in points.items()}
    fl, fr, rl, rr = (
        wheel_points[name] for name in
        ("front_left", "front_right", "rear_left", "rear_right")
    )
    if not (fl[0] < fr[0] and rl[0] < rr[0]
            and fl[2] > rl[2] and fr[2] > rr[2]):
        raise ValueError("wheel contact points must have correct left/right and front/rear ordering")
    if not all(abs(p[0]) <= half[0] and abs(p[2]) <= half[2] for p in wheel_points.values()):
        raise ValueError("wheel points must lie within chassis horizontal envelope")
    if any(abs(p[1]+ride_height) > 1e-9 for p in wheel_points.values()):
        raise ValueError("ground wheel reference must agree with vehicle ride_height")
    if abs(((fl[2] + fr[2]) / 2 - (rl[2] + rr[2]) / 2) - wheelbase) > 1e-9:
        raise ValueError("wheel contact points disagree with fixed legacy wheelbase")

    suspension = _record(simulation["suspension"], {
        "mode", "max_bump_m", "max_droop_m"
    }, "simulation.suspension")
    _exact(suspension["mode"], "simplified_independent_four_wheel", "suspension.mode")
    for key in ("max_bump_m", "max_droop_m"):
        magnitude = _numeric(suspension[key], f"suspension.{key}", positive=True)
        if magnitude >= ride_height:
            raise ValueError("suspension travel must be below ride height")

    airborne = _record(simulation["airborne"], {
        "mode", "max_safe_touchdown_vertical_speed_mps",
        "landing_collision_response",
    }, "simulation.airborne")
    _exact(airborne["mode"], "ballistic_gravity_validated_landing", "airborne.mode")
    _optional_float(airborne["max_safe_touchdown_vertical_speed_mps"], "touchdown speed")
    if airborne["landing_collision_response"] is not None and (
        not isinstance(airborne["landing_collision_response"], str)
        or not airborne["landing_collision_response"].strip()
    ):
        raise ValueError("invalid landing_collision_response")

    driving = _record(root["driving"], {
        "full_body_inside_road_union_required", "route_reference",
        "soft_deviation_m", "hard_deviation_m", "checkpoint_finish",
        "hard_route_violation_semantics",
    }, "driving")
    _exact(driving["full_body_inside_road_union_required"], True, "road_union")
    _exact(driving["route_reference"], "authored_route_v1", "route_reference")
    _exact(_numeric(driving["soft_deviation_m"], "soft deviation"), 2.0, "soft_deviation_m")
    _exact(_numeric(driving["hard_deviation_m"], "hard deviation"), 4.0, "hard_deviation_m")
    triggers = _record(driving["checkpoint_finish"], {
        "mode", "continuous_sweep_required", "ordered_checkpoints_required"
    }, "checkpoint_finish")
    _exact(triggers["mode"], "directed_center_crossing", "checkpoint_finish.mode")
    _exact(triggers["continuous_sweep_required"], True, "continuous_sweep")
    _exact(triggers["ordered_checkpoints_required"], True, "ordered_checkpoints")
    if driving["hard_route_violation_semantics"] is not None and (
        not isinstance(driving["hard_route_violation_semantics"], str)
        or not driving["hard_route_violation_semantics"].strip()
    ):
        raise ValueError("invalid hard route violation semantics")

    reliability = _record(root["reliability"], {
        "minimum_independent_ppo_seed_pairs", "confidence_method",
        "confidence_level", "minimum_lower_success_bound_per_train_track",
        "episodes_per_track_per_model_per_condition", "evaluation_conditions",
        "spawn_lateral_offset_m", "spawn_heading_jitter_deg",
        "perturbation_seed_protocol", "feasible_spawn_filter",
    }, "reliability")
    _exact(reliability["minimum_independent_ppo_seed_pairs"], 3, "ppo seed pairs")
    _exact(reliability["confidence_method"], "exact_one_sided_clopper_pearson", "confidence method")
    _exact(_numeric(reliability["confidence_level"], "confidence level"), 0.95, "confidence level")
    _exact(_numeric(reliability["minimum_lower_success_bound_per_train_track"], "minimum lower bound"),
           0.90, "minimum lower bound")
    n = reliability["episodes_per_track_per_model_per_condition"]
    if type(n) is not int or n < 60:
        raise ValueError("per-track/model/condition sample size must be at least 60")
    _exact(reliability["evaluation_conditions"], ["nominal", "randomized_train"], "evaluation conditions")
    _pair(reliability["spawn_lateral_offset_m"], "spawn lateral range", (-.5, .5))
    _pair(reliability["spawn_heading_jitter_deg"], "heading jitter range", (-5., 5.))
    for key in ("perturbation_seed_protocol", "feasible_spawn_filter"):
        if reliability[key] is not None and (
            not isinstance(reliability[key], str) or not reliability[key].strip()
        ):
            raise ValueError(f"{key} must be a nonempty version identifier or null")

    backup = _record(root["backup"], {
        "provider", "require_sha256_manifest",
        "require_independent_restore_test", "private_folder_id",
    }, "backup")
    _exact(backup["provider"], "private_google_drive", "backup.provider")
    _exact(backup["require_sha256_manifest"], True, "backup.sha256")
    _exact(backup["require_independent_restore_test"], True, "backup.restore")
    if backup["private_folder_id"] is not None and (
        not isinstance(backup["private_folder_id"], str)
        or not backup["private_folder_id"].strip()
    ):
        raise ValueError("private folder id must be nonempty or null")

    gates = _record(root["evidence_gates"], set(EVIDENCE_GATES), "evidence_gates")
    # A future audit may replace 'pending' with hashes/attestations through a
    # separately reviewed schema. Never promote a string to approval here.
    if any(gates[name] != "pending" for name in EVIDENCE_GATES):
        raise ValueError("preregistration evidence gates cannot self-approve")
    heldout = _record(root["heldout"], {"status", "evaluation_permitted"}, "heldout")
    _exact(heldout["status"], "sealed", "heldout.status")
    _exact(heldout["evaluation_permitted"], False, "heldout.evaluation_permitted")
    _exact(root["scientific_freeze_permitted"], False, "scientific_freeze_permitted")
    _exact(root["wp8_permitted"], False, "wp8_permitted")

    pending = list(EVIDENCE_GATES)
    pending.extend([
        "airborne: landing impact and collision response not registered",
        "driving: hard-route violation episode policy not registered",
        "reliability: perturbation seeds and feasible-spawn filter not registered",
        "backup: independent restoration has not been attested",
        "scientific runner: admission disabled until separate peer-reviewed gate",
    ])
    return PreregistrationReview(
        codes=OWNER_CODES,
        status=root["status"],
        blockers=tuple(pending),
    )
