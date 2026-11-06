from copy import deepcopy
from pathlib import Path

import numpy as np

from f1rl.research.simulation_parameter_provenance import (
    _within_domain,
    build_parameter_provenance,
)
from f1rl.sandbox.tire_calibration_contract import REQUIRED_WP7_MEASUREMENTS

ROOT = Path(__file__).resolve().parents[2]


def test_parameter_domain_rejects_nonfinite_values_and_enforces_inclusive_bounds():
    assert _within_domain(0.0, 0.0, True, 1.0, True)
    assert not _within_domain(0.0, 0.0, False, 1.0, True)
    assert not _within_domain(1.0, 0.0, True, 1.0, False)
    assert not _within_domain(float("nan"), 0.0, True, None, False)
    assert not _within_domain(float("inf"), 0.0, True, None, False)
    assert not _within_domain(float("-inf"), 0.0, True, None, False)


def test_parameter_manifest_binds_all_wp7_names_to_runtime_and_legal_domains():
    report = build_parameter_provenance(ROOT)

    assert report["schema_version"] == "f1rl_simulation_parameter_provenance_v1"
    assert [row["name"] for row in report["parameters"]] == list(REQUIRED_WP7_MEASUREMENTS)
    assert sum(row["runtime_status"] == "runtime_exact" for row in report["parameters"]) == 6
    assert sum(row["runtime_status"] == "runtime_analogue" for row in report["parameters"]) == 1
    assert sum(row["runtime_status"] == "not_consumed" for row in report["parameters"]) == 5
    assert report["externally_measured"] is False
    assert report["independently_physically_validated"] is False
    assert len(report["configuration_sha256"]) == 64
    assert len(report["runtime_physics_sha256"]) == 64
    for row in report["parameters"]:
        bounds = row["legal_bounds"]
        assert bounds["unit"] == row["unit"]
        assert bounds["source"]
        for value_key in ("configured_value", "runtime_value"):
            value = row[value_key]
            if value is None:
                assert row["runtime_status"] == "not_consumed"
                continue
            assert np.isfinite(value)
            assert value > bounds["minimum"] or (
                bounds["minimum_inclusive"] and value == bounds["minimum"]
            )
            if bounds["maximum"] is not None:
                assert value < bounds["maximum"] or (
                    bounds["maximum_inclusive"] and value == bounds["maximum"]
                )


def test_each_consumed_parameter_has_deterministic_finite_sensitivity():
    from f1rl.research.wp7_protocol import load_registration, prepare_scenarios

    base = load_registration(ROOT, profile="spatial_rotating_owner_v3")
    parameter_paths = {
        "mass_kg": ("engineering_suspension", "mass_kg"),
        "pitch_inertia_kg_m2": ("engineering_suspension", "pitch_inertia_kg_m2"),
        "roll_inertia_kg_m2": ("engineering_suspension", "roll_inertia_kg_m2"),
        "spring_stiffness_npm": ("engineering_suspension", "spring_stiffness_npm"),
        "damper_nspm": ("engineering_suspension", "damper_nspm"),
        "yaw_inertia_kg_m2": ("engineering_spatial_contact", "yaw_inertia_kg_m2"),
        "tire_friction_mu": (
            "engineering_spatial_contact", "traction", "traction_mu"
        ),
    }

    def run(registration):
        factory = prepare_scenarios(ROOT, registration, seed_index=0).scenarios[0].factory
        env = factory.create(reset_rank=0, reset_sequence=0)
        try:
            observation, _ = env.reset(seed=23_012_026)
            states = [observation.copy()]
            for action in (
                [0.0, 1.0, 0.0], [0.5, 1.0, 0.0],
                [-0.5, 1.0, 0.0], [0.0, 1.0, 0.0],
            ):
                observation, reward, terminated, truncated, info = env.step(
                    np.asarray(action, dtype=np.float32)
                )
                assert env.observation_space.contains(observation)
                assert np.isfinite(observation).all() and np.isfinite(reward)
                assert isinstance(info.get("physics", {}).get("transition"), dict)
                assert not terminated and not truncated
                states.append(observation.copy())
            return factory.physics_config.sha256, np.concatenate(states)
        finally:
            env.close()

    baseline_sha, baseline_states = run(base)
    for path in parameter_paths.values():
        perturbed = deepcopy(base)
        value = perturbed
        for key in path[:-1]:
            value = value[key]
        value[path[-1]] *= 1.01
        changed_sha, changed_states = run(perturbed)
        assert changed_sha != baseline_sha
        assert np.isfinite(changed_states).all()
        assert not np.array_equal(changed_states, baseline_states)
