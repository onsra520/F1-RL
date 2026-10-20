"""Deterministic, TRAIN-only simulation consistency probe; not physical validation."""
from __future__ import annotations

from hashlib import sha256
import json
from math import isfinite
from numbers import Real
from pathlib import Path

import numpy as np

from .wp7_protocol import load_registration, prepare_scenarios
from ..learning.train_reset import derive_train_reset_draw_seed


PROTOCOL_PATH = Path("configs/research/simulation-validation-v1.json")


def load_simulation_validation_protocol(root: str | Path) -> tuple[dict, str]:
    path = Path(root).resolve(strict=True) / PROTOCOL_PATH
    value = json.loads(path.read_text(encoding="utf-8"))
    expected_keys = {
        "schema_version", "status", "scope", "source_profile", "split", "registered_tracks",
        "fixture", "tolerances", "invariants", "failure_classes", "physical_calibration_claim",
        "scientific_acceptance_claim",
    }
    if (not isinstance(value, dict) or set(value) != expected_keys
            or value["schema_version"] != "f1rl_simulation_validation_v1"
            or value["status"] != "engineering_protocol_draft"
            or value["scope"] != "simulation_consistency_only_not_physical_calibration"
            or value["source_profile"] != "spatial_rotating_owner_v3"
            or value["split"] != "TRAIN"
            or value["registered_tracks"] != ["straight-sprint-v1", "slalom-v1", "right-turn-v1", "ramp-v1"]
            or value["physical_calibration_claim"] is not False
            or value["scientific_acceptance_claim"] is not False):
        raise ValueError("simulation validation protocol is malformed or scope-escalated")
    fixture = value["fixture"]
    if (not isinstance(fixture, dict) or set(fixture) != {
            "reset_seed", "reset_rank", "reset_index", "action_tape", "repeats"}
            or type(fixture["reset_seed"]) is not int or not 0 <= fixture["reset_seed"] <= 2**32-1
            or fixture["reset_rank"] != 0 or fixture["reset_index"] != 0 or fixture["repeats"] != 2
            or not isinstance(fixture["action_tape"], list) or not fixture["action_tape"]
            or any(not isinstance(action, list) or len(action) != 3
                   or any(type(component) not in (int, float) or not isfinite(component)
                          for component in action) for action in fixture["action_tape"])):
        raise ValueError("simulation validation fixture is malformed")
    tolerances = value["tolerances"]
    if (not isinstance(tolerances, dict) or set(tolerances) != {
            "observation_absolute", "reward_absolute", "state_absolute", "time_absolute_seconds"}
            or any(type(item) not in (int, float) or not isfinite(item) or item < 0.
                   for item in tolerances.values())):
        raise ValueError("simulation validation tolerances are malformed")
    return value, sha256(path.read_bytes()).hexdigest()


def _all_finite(value) -> bool:
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, Real) and not isinstance(value, bool):
        return isfinite(float(value))
    if isinstance(value, np.ndarray):
        return bool(np.isfinite(value).all())
    return True


def _close(left, right, tolerance: float) -> bool:
    return bool(np.allclose(np.asarray(left), np.asarray(right), rtol=0.0, atol=tolerance))


def _episode_probe(factory, track_id: str, *, seed: int, action_tape: list[list[float]]) -> dict:
    env = factory.create(reset_rank=0, reset_sequence=0)
    try:
        observation, reset_info = env.reset(seed=seed)
        expected = factory.reset_config.identity_for_track(track_id)
        reset = reset_info.get("train_reset")
        if (reset is None or reset.get("seed") != expected["seed"]
                or reset.get("rank") != 0 or reset.get("reset_index") != 0
                or reset.get("track_id") != track_id
                or reset.get("configuration_sha256") != expected["configuration_sha256"]
                or reset.get("world_sha256") != expected["world_sha256"]
                or reset.get("draw_seed") != derive_train_reset_draw_seed(
                    version=expected["version"], seed=expected["seed"], rank=0,
                    track_id=track_id, reset_index=0)):
            raise ValueError("reset_identity_mismatch")
        if (not env.observation_space.contains(observation) or not np.isfinite(observation).all()
                or not _all_finite(reset_info)):
            raise ValueError("space_contract_violation_or_non_finite_reset")
        initial_observation = observation.copy()
        rows = []
        for action_values in action_tape:
            action = np.asarray(action_values, dtype=np.float32)
            if not env.action_space.contains(action):
                raise ValueError("space_contract_violation")
            observation, reward, terminated, truncated, info = env.step(action)
            if (not env.observation_space.contains(observation) or not np.isfinite(observation).all()
                    or not isfinite(float(reward)) or not _all_finite(info)):
                raise ValueError("non_finite_state_or_space_contract_violation")
            transition = (info.get("physics") or {}).get("transition")
            if not isinstance(transition, dict) or transition.get("version") != "wp7_transition_audit_v1":
                raise ValueError("physics_transition_error")
            row = {
                "observation": observation.tolist(), "reward": float(reward),
                "terminated": bool(terminated), "truncated": bool(truncated),
                "position": info["telemetry"]["vehicle_position"],
                "velocity": info["telemetry"]["vehicle_velocity"],
                "physics_transition": transition,
                "route": info.get("race"),
                "train_reset": info.get("train_reset"),
                "elapsed_time": float(info["elapsed_time"]),
            }
            rows.append(row)
            if terminated or truncated:
                break
        if len(rows) != len(action_tape):
            raise ValueError("route_or_termination_contract_failure: fixture terminated early")
        return {"reset": reset, "initial_observation": initial_observation.tolist(), "steps": rows}
    finally:
        env.close()


def run_simulation_validation(root: str | Path) -> dict:
    """Run paired actual-environment probes for each registered TRAIN track."""
    root = Path(root).resolve(strict=True)
    protocol, protocol_sha256 = load_simulation_validation_protocol(root)
    registration = load_registration(root, profile=protocol["source_profile"])
    scenarios = prepare_scenarios(root, registration, seed_index=0)
    if [scenario.track_id for scenario in scenarios.scenarios] != protocol["registered_tracks"]:
        raise ValueError("source_or_config_drift: TRAIN track roster differs")
    seed = protocol["fixture"]["reset_seed"]
    results = {}
    for scenario in scenarios.scenarios:
        first = _episode_probe(scenario.factory, scenario.track_id, seed=seed,
                               action_tape=protocol["fixture"]["action_tape"])
        second = _episode_probe(scenario.factory, scenario.track_id, seed=seed,
                                action_tape=protocol["fixture"]["action_tape"])
        tolerance = protocol["tolerances"]
        if (len(first["steps"]) != len(second["steps"])
                or not _close(first["initial_observation"], second["initial_observation"],
                              tolerance["observation_absolute"])):
            raise ValueError(f"nondeterministic_replay: {scenario.track_id} reset differs")
        for left, right in zip(first["steps"], second["steps"], strict=True):
            if (not _close(left["observation"], right["observation"], tolerance["observation_absolute"])
                    or abs(left["reward"] - right["reward"]) > tolerance["reward_absolute"]
                    or not _close(left["position"], right["position"], tolerance["state_absolute"])
                    or not _close(left["velocity"], right["velocity"], tolerance["state_absolute"])
                    or abs(left["elapsed_time"] - right["elapsed_time"])
                    > tolerance["time_absolute_seconds"]
                    or left["terminated"] != right["terminated"] or left["truncated"] != right["truncated"]
                    or left["physics_transition"] != right["physics_transition"]
                    or left["route"] != right["route"]):
                raise ValueError(f"nondeterministic_replay: {scenario.track_id} transition differs")
        digest = sha256(json.dumps(first, sort_keys=True, separators=(",", ":"),
                                  allow_nan=False).encode()).hexdigest()
        results[scenario.track_id] = {"status": "PASS", "paired_replay_sha256": digest,
                                      "steps": len(first["steps"]),
                                      "reset_identity": first["reset"]}
    return {
        "schema_version": "f1rl_simulation_validation_result_v1",
        "status": "SIMULATION_CONSISTENCY_PASS",
        "scope": protocol["scope"], "physical_calibration_claim": False,
        "scientific_acceptance_claim": False, "split": "TRAIN",
        "protocol_sha256": protocol_sha256,
        "track_results": results,
        "registered_4m_boundary_preserved": True,
    }
