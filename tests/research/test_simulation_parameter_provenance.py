from pathlib import Path
from copy import deepcopy

from f1rl.research.simulation_parameter_provenance import build_parameter_provenance
from f1rl.sandbox.tire_calibration_contract import REQUIRED_WP7_MEASUREMENTS

ROOT = Path(__file__).resolve().parents[2]


def test_parameter_manifest_binds_all_wp7_names_to_live_runtime_or_marks_absent():
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


def test_runtime_parameter_perturbation_reaches_physics_and_finite_train_transition():
    import numpy as np

    from f1rl.research.wp7_protocol import load_registration, prepare_scenarios

    baseline = load_registration(ROOT, profile="spatial_rotating_owner_v3")
    perturbed = deepcopy(baseline)
    perturbed["engineering_suspension"]["spring_stiffness_npm"] *= 1.01
    left = prepare_scenarios(ROOT, baseline, seed_index=0).scenarios[0].factory
    right = prepare_scenarios(ROOT, perturbed, seed_index=0).scenarios[0].factory
    assert left.physics_config.sha256 != right.physics_config.sha256
    assert left.physics_config.independent_suspension_policy.spring_stiffness_npm != (
        right.physics_config.independent_suspension_policy.spring_stiffness_npm)

    transitions = []
    for factory in (left, right):
        env = factory.create(reset_rank=0, reset_sequence=0)
        try:
            observation, _ = env.reset(seed=23_012_026)
            assert env.observation_space.contains(observation)
            row = env.step(np.asarray([0.2, 0.4, 0.0], dtype=np.float32))
            assert env.observation_space.contains(row[0])
            assert np.isfinite(row[0]).all() and np.isfinite(row[1])
            assert isinstance(row[4].get("physics", {}).get("transition"), dict)
            transitions.append(row)
        finally:
            env.close()
    assert not np.array_equal(transitions[0][0], transitions[1][0])
