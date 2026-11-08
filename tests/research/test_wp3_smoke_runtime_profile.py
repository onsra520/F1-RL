from pathlib import Path
import importlib.util

import numpy as np
import pytest

from f1rl.learning.crash_terminal_registered import validate_registered_crash_terminal_info
from f1rl.local_demo import evaluation as evaluation_module
from f1rl.research import wp3_eval_only_coordinator as coordinator
from f1rl.research import wp7_protocol
from f1rl.research.wp7_protocol import load_registration, prepare_scenarios
from tests.integration.test_wp3_applied_crash_end_to_end import crash_factory


ROOT = Path(__file__).resolve().parents[2]


def _registered_v3_factory():
    registration = load_registration(ROOT, profile="spatial_rotating_owner_v3")
    scenarios = prepare_scenarios(ROOT, registration, seed_index=0)
    return next(row.factory for row in scenarios.scenarios
                if row.track_id == "straight-sprint-v1")


def test_full06_runtime_profile_reaches_factory_and_environment_telemetry():
    base = _registered_v3_factory()
    factory, provenance = coordinator.apply_runtime_profile(
        base, coordinator.FULL06_RUNTIME_PROFILE)

    assert base.physics_config.ground_motion_version == "spatial_rotating_compliant_lifecycle_v3"
    assert base.race_config.version == "wp7_directed_lifecycle_owner_v2"
    assert factory.physics_config.ground_motion_version == "spatial_road_edge_partial_lifecycle_v4"
    assert factory.physics_config.independent_suspension_policy.max_abs_tilt_rad == 0.7
    assert factory.race_config.version == "wp7_directed_lifecycle_crash_contract_v4"
    assert provenance["requested_profile"] == coordinator.FULL06_RUNTIME_PROFILE
    assert provenance["effective"]["physics_version"] == factory.physics_config.ground_motion_version
    assert provenance["effective"]["race_version"] == factory.race_config.version
    assert provenance["effective"]["terminal_contract"] == "wp3_physical_crash_terminal_v1"

    env = factory.create()
    try:
        _, info = env.reset(seed=20261106)
        assert info["physics"]["configuration"]["ground_motion_version"] == (
            "spatial_road_edge_partial_lifecycle_v4")
        assert info["race"]["configuration"]["version"] == (
            "wp7_directed_lifecycle_crash_contract_v4")
        assert info["race"]["physical_crash_contract"]["schema_version"] == (
            "wp3_physical_crash_terminal_v1")
    finally:
        env.close()


def test_v3_factory_fails_when_it_does_not_match_requested_v4_profile():
    factory = _registered_v3_factory()

    with pytest.raises(coordinator.RuntimeProfileMismatch, match="runtime profile mismatch"):
        coordinator.validate_runtime_profile(factory, coordinator.FULL06_RUNTIME_PROFILE)


def test_registered_v3_profile_remains_v3_without_explicit_overlay():
    factory = _registered_v3_factory()

    assert factory.physics_config.ground_motion_version == "spatial_rotating_compliant_lifecycle_v3"
    assert factory.race_config.version == "wp7_directed_lifecycle_owner_v2"
    env = factory.create()
    try:
        observation, _ = env.reset(seed=20261106)
        observation, _, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))
    finally:
        env.close()

    assert observation is not None
    assert not terminated and not truncated
    assert info["physics"]["configuration"]["ground_motion_version"] == (
        "spatial_rotating_compliant_lifecycle_v3")
    assert info["race"]["configuration"]["version"] == (
        "wp7_directed_lifecycle_owner_v2")
    with pytest.raises(ValueError, match="unsupported smoke runtime profile"):
        coordinator.apply_runtime_profile(factory, "spatial_rotating_owner_v3")


def test_full06_profile_emits_valid_registered_physical_crash_terminal():
    factory, provenance = coordinator.apply_runtime_profile(
        crash_factory(), coordinator.FULL06_RUNTIME_PROFILE)
    env = factory.create()
    try:
        env.reset()
        _, _, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))
    finally:
        env.close()

    assert terminated and not truncated
    assert info["research_outcome"]["valid_sample"] is True
    assert info["research_outcome"]["reason"] == "physical_crash"
    assert validate_registered_crash_terminal_info(info)
    assert provenance["effective"]["terminal_contract"] == "wp3_physical_crash_terminal_v1"


def test_smoke_cli_propagates_requested_runtime_profile(monkeypatch, tmp_path):
    cli_path = ROOT / "scripts/wp3_eval_only_coordinator.py"
    spec = importlib.util.spec_from_file_location("wp3_eval_only_coordinator_cli", cli_path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    seen = {}

    def fake_run_engineering(root, history, out, run_id, **kwargs):
        seen.update(kwargs)
        return {"invalid": 0}

    monkeypatch.setattr(cli, "run_engineering", fake_run_engineering)
    code = cli.main(["smoke", "--out", str(tmp_path / "run"),
                     "--runtime-profile", coordinator.FULL06_RUNTIME_PROFILE])

    assert code == 0
    assert seen["runtime_profile"] == coordinator.FULL06_RUNTIME_PROFILE


def test_smoke_cli_rejects_v3_request_before_starting_run(monkeypatch, tmp_path):
    cli_path = ROOT / "scripts/wp3_eval_only_coordinator.py"
    spec = importlib.util.spec_from_file_location("wp3_eval_only_coordinator_cli", cli_path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    calls = []
    monkeypatch.setattr(cli, "run_engineering", lambda *args, **kwargs: calls.append(args))

    with pytest.raises(SystemExit) as error:
        cli.main(["smoke", "--out", str(tmp_path / "run"),
                  "--runtime-profile", "spatial_rotating_owner_v3"])

    assert error.value.code == 2
    assert calls == []


def _patch_production_episode(monkeypatch, factory, evaluator):
    monkeypatch.setattr(wp7_protocol, "prepare_evaluation_factory",
                        lambda *args: (factory, {}))
    monkeypatch.setattr(evaluation_module, "_evaluate_episode", evaluator)


def _production_case():
    return {"model_id": "C-seed0", "seed_index": 0, "track": "straight-sprint-v1",
            "stratum": "nominal", "env_seed": 20261106, "policy_seed": 1234}


def test_production_episode_passes_validated_profile_to_evaluator(monkeypatch, tmp_path):
    base = _registered_v3_factory()
    seen = {}

    def evaluate(model, factory, track, env_seed, policy_seed, fixed_dt):
        env = factory.create()
        try:
            _, info = env.reset(seed=env_seed)
        finally:
            env.close()
        seen.update({
            "factory": coordinator.effective_runtime_profile(factory),
            "physics_telemetry": info["physics"]["configuration"]["ground_motion_version"],
            "race_telemetry": info["race"]["configuration"]["version"],
            "terminal_contract": info["race"]["physical_crash_contract"]["schema_version"],
        })
        return {"deterministic_prediction": False, "environment_seed": env_seed,
                "policy_seed": policy_seed, "success": False, "failure_reason": "physical_crash"}

    _patch_production_episode(monkeypatch, base, evaluate)
    metrics = coordinator._production_episode(
        ROOT, tmp_path, _production_case(), {"C-seed0": object(), "protocol": object()})

    assert seen["physics_telemetry"] == "spatial_road_edge_partial_lifecycle_v4"
    assert seen["race_telemetry"] == "wp7_directed_lifecycle_crash_contract_v4"
    assert seen["terminal_contract"] == "wp3_physical_crash_terminal_v1"
    assert metrics["runtime_profile"]["requested_profile"] == coordinator.FULL06_RUNTIME_PROFILE
    assert metrics["runtime_profile"]["effective"] == seen["factory"]


def test_production_episode_stops_before_evaluator_on_effective_profile_mismatch(
        monkeypatch, tmp_path):
    base = _registered_v3_factory()
    evaluator_calls = []
    _patch_production_episode(
        monkeypatch, base, lambda *args: evaluator_calls.append(args))
    monkeypatch.setattr(coordinator, "effective_runtime_profile", lambda factory: {
        "physics_version": "spatial_rotating_compliant_lifecycle_v3",
        "race_version": "wp7_directed_lifecycle_owner_v2",
        "terminal_contract": None,
        "max_abs_tilt_rad": factory.physics_config.independent_suspension_policy.max_abs_tilt_rad,
    })

    with pytest.raises(coordinator.RuntimeProfileMismatch):
        coordinator._production_episode(
            ROOT, tmp_path, _production_case(), {"C-seed0": object(), "protocol": object()})

    assert evaluator_calls == []
